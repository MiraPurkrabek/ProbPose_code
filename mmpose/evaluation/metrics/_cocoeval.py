import copy
import datetime
import sys
import time
import warnings
from collections import defaultdict

import numpy as np

from mmpose.structures.keypoint import fix_bbox_aspect_ratio
from . import _mask as maskUtils


class NullWriter(object):

    def write(self, arg):
        pass

    def flush(self):
        pass


class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # The original Microsoft COCO Toolbox is written
    # # by Piotr Dollar and Tsung-Yi Lin, 2014.
    # # Licensed under the Simplified BSD License [see bsd.txt]
    ######################################################################
    # Updated and renamed to Extended COCO Toolbox (xtcocotool) \
    # by Sheng Jin & Can Wang in 2020. The Extended COCO Toolbox is
    # developed to support multiple pose-related datasets, including COCO,
    # CrowdPose and so on.

    def __init__(
        self,
        cocoGt=None,
        cocoDt=None,
        iouType="keypoints",
        sigmas=None,
        use_area=True,
        extended_oks=False,
        match_by_bbox=False,
        confidence_thr=0.5,
        padding=1.25,
        ignore_near_bbox=False,
    ):
        """
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :param iouType: 'segm', 'bbox' or 'keypoints', 'keypoints_crowd'
        :param sigmas: keypoint labelling sigmas.
        :param use_area (bool): If gt annotations (eg. CrowdPose, AIC)
                                do not have 'area', please set use_area=False.
        :return: None
        """
        if not iouType:
            print("iouType not specified. use default iouType keypoints")
        if sigmas is not None:
            self.sigmas = sigmas
        else:
            # The default sigmas are used for COCO dataset.
            self.sigmas = (
                np.array(
                    [
                        0.26,
                        0.25,
                        0.25,
                        0.35,
                        0.35,
                        0.79,
                        0.79,
                        0.72,
                        0.72,
                        0.62,
                        0.62,
                        1.07,
                        1.07,
                        0.87,
                        0.87,
                        0.89,
                        0.89,
                    ]
                )
                / 10.0
            )

        self.cocoGt = copy.deepcopy(cocoGt)  # ground truth COCO API
        self.cocoDt = copy.deepcopy(cocoDt)  # detections COCO API
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iouType=iouType)  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.stats_names = []  # names of summarized metrics
        self.ious = {}  # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())
            self.anno_file = (None, None)
        self.use_area = use_area
        self.score_key = "score"

        self.extended_oks = extended_oks
        self.confidence_thr = confidence_thr

        self.loc_similarities = []

        self.match_by_bbox = match_by_bbox
        self.padding = padding
        self.ignore_near_bbox = ignore_near_bbox

        self.TMP_COUNTER = 0

    def _prepare(self):
        """
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        """

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann["segmentation"] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == "segm":
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)

        # Find all visibility levels that are present in the dataset
        self.gt_visibilities = set()
        visibilities_counts = defaultdict(int)
        num_pts_above_padding = 1e-8
        num_inst_above_padding = 1e-8
        num_annotated_pts = 1e-8
        num_wrong_3 = 1e-8
        for gt in gts:
            if "keypoints" in p.iouType:
                if p.iouType == "keypoints_wholebody":
                    body_gt = gt["keypoints"]
                    foot_gt = gt["foot_kpts"]
                    face_gt = gt["face_kpts"]
                    lefthand_gt = gt["lefthand_kpts"]
                    righthand_gt = gt["righthand_kpts"]
                    wholebody_gt = body_gt + foot_gt + face_gt + lefthand_gt + righthand_gt
                    g = np.array(wholebody_gt)
                    vis = g[2::3]
                elif p.iouType == "keypoints_foot":
                    g = np.array(gt["foot_kpts"])
                    vis = g[2::3]
                elif p.iouType == "keypoints_face":
                    g = np.array(gt["face_kpts"])
                    vis = g[2::3]
                elif p.iouType == "keypoints_lefthand":
                    g = np.array(gt["lefthand_kpts"])
                    vis = g[2::3]
                elif p.iouType == "keypoints_righthand":
                    g = np.array(gt["righthand_kpts"])
                    vis = g[2::3]
                elif p.iouType == "keypoints_crowd":
                    # 'num_keypoints' in CrowdPose dataset only counts
                    # the visible joints (vis = 2)
                    k = gt["num_keypoints"]
                    self.score_key = "score"
                else:
                    g = np.array(gt["keypoints"])
                    vis = g[2::3]

                num_annotated_pts += np.sum(vis > 0)

                if self.ignore_near_bbox:
                    # Ignore keypoints near the bbox edge
                    bbox_xywh = gt["bbox"]
                    x0, y0, w, h = bbox_xywh
                    x1, y1 = x0 + w, y0 + h
                    tol_x = 0.05 * w
                    tol_y = 0.05 * h
                    x = g[0::3]
                    y = g[1::3]
                    x = np.array(x)
                    y = np.array(y)

                    near_bbox = (
                        ((np.abs(x - x0) < tol_x) & (y > y0 - tol_y) & (y < y1 + tol_y))
                        | ((np.abs(x - x1) < tol_x) & (y > y0 - tol_y) & (y < y1 + tol_y))
                        | ((np.abs(y - y0) < tol_y) & (x > x0 - tol_x) & (x < x1 + tol_x))
                        | ((np.abs(y - y1) < tol_y) & (x > x0 - tol_x) & (x < x1 + tol_x))
                    )
                    vis[near_bbox] = 0

                if not self.extended_oks:
                    # In the original OKS metric, there are only 3 levels of visibility
                    # Set everything else than {1, 2} to zeros

                    # TODO: this behavior is not consistent with the original OKS.
                    # The original does not know v=3, but there are points outside of the AM
                    # and they are evaluated as regular ones. It skews the statistics (gives unreasonable
                    # expectations) and worsen the result for v=1.
                    vis_mask = (vis == 1) | (vis == 2)
                    vis[~vis_mask] = 0

                # Set visibility to 3 if the keypoint is out of the image
                # Do that only for extended_oks. For original OKS, visibility 3 is ignored and would
                # skew results compared to the original OKS.
                elif "pad_to_contain" in gt:
                    pad_to_contain = np.array(gt["pad_to_contain"])
                    pad_to_contain[vis <= 0] = -1.0  # Unannotated keypoints are not considered
                    out_mask = pad_to_contain > self.padding
                    num_wrong_3 += (out_mask & (vis != 3)).sum()
                    vis[(vis > 2) & (~out_mask)] = 1
                    vis[out_mask] = 3

                    num_pts_above_padding += out_mask.sum()
                    num_inst_above_padding += out_mask.any().astype(int)

                unique_vis, vis_counts = np.unique(vis.astype(int), return_counts=True)
                self.gt_visibilities.update(unique_vis)
                for u_vis, count in zip(unique_vis, vis_counts):
                    visibilities_counts[u_vis] += count

                # Update the edited visibilities to the GT
                gt[p.iouType][2::3] = vis.astype(int).tolist()

        self.gt_visibilities = sorted(list(self.gt_visibilities))
        self.gt_visibilities = [vis for vis in self.gt_visibilities if vis > 0]
        print(
            "Number of keypoints above padding: {:d} ({:.2f} %)".format(
                int(num_pts_above_padding), num_pts_above_padding / num_annotated_pts * 100
            )
        )
        print(
            "Number of instances above padding: {:d} ({:.2f} %)".format(
                int(num_inst_above_padding), num_inst_above_padding / len(gts) * 100
            )
        )
        print(
            "Number of keypoints with wrong visibility 3: {:d} ({:.2f} %)".format(
                int(num_wrong_3), num_wrong_3 / num_annotated_pts * 100
            )
        )
        print("Evaluating {:d} levels of visibility: {}".format(len(self.gt_visibilities) + 1, self.gt_visibilities))
        all_kpts = np.array([count for _, count in visibilities_counts.items()]).sum()
        for vis, count in visibilities_counts.items():
            print("\tvisibility {:2d}: {:6d} ({:5.2f} %)".format(vis, count, count / all_kpts * 100))

        _vis_conditions = [lambda x: x > 0]
        for vis in self.gt_visibilities:
            _vis_conditions.append(lambda x, vis=vis: x == vis)

        # for each visibility level, set ignore flag, visibility vector and score key
        for gt in gts:
            gt_ignore = gt["ignore"] if "ignore" in gt else 0
            gt_ignore = gt_ignore and ("iscrowd" in gt and gt["iscrowd"])
            gt["ignore"] = [gt_ignore for _ in range(len(self.gt_visibilities) + 1)]
            if "keypoints" in p.iouType:
                if p.iouType == "keypoints_wholebody":
                    body_gt = gt["keypoints"]
                    foot_gt = gt["foot_kpts"]
                    face_gt = gt["face_kpts"]
                    lefthand_gt = gt["lefthand_kpts"]
                    righthand_gt = gt["righthand_kpts"]
                    wholebody_gt = body_gt + foot_gt + face_gt + lefthand_gt + righthand_gt
                    kpts = np.array(wholebody_gt)
                    vis = kpts[2::3]
                    self.score_key = "wholebody_score"
                elif p.iouType == "keypoints_foot":
                    kpts = np.array(gt["foot_kpts"])
                    vis = kpts[2::3]
                    self.score_key = "foot_score"
                elif p.iouType == "keypoints_face":
                    kpts = np.array(gt["face_kpts"])
                    vis = kpts[2::3]
                    self.score_key = "face_score"
                elif p.iouType == "keypoints_lefthand":
                    kpts = np.array(gt["lefthand_kpts"])
                    vis = kpts[2::3]
                    self.score_key = "lefthand_score"
                elif p.iouType == "keypoints_righthand":
                    kpts = np.array(gt["righthand_kpts"])
                    vis = kpts[2::3]
                    self.score_key = "righthand_score"
                elif p.iouType == "keypoints_crowd":
                    # 'num_keypoints' in CrowdPose dataset only counts
                    # the visible joints (vis = 2)
                    k = gt["num_keypoints"]
                    gt["ignore"] = [gt_ignore or k == 2 for gt_ignore in gt["ignore"]]
                    self.score_key = "score"
                else:
                    kpts = np.array(gt["keypoints"])
                    vis = kpts[2::3]
                    self.score_key = "score"

                if p.iouType != "keypoints_crowd":
                    for i, gt_ig in enumerate(gt["ignore"]):
                        vis_cond = _vis_conditions[i]
                        k = np.count_nonzero(vis_cond(vis))
                        gt["ignore"][i] = gt_ig or (k == 0)

                # Update gt['ignore'] by visibility. If no keypoint with visibility v is annotated,
                # ignore the instance for that visibility level. Do not change gt_ignore[0]
                unique_vis = np.unique(vis[vis > 0].astype(int))
                gt_ignore = np.array(gt["ignore"])
                gt_ignore[1:] = True
                gt_ignore[unique_vis] = False
                gt_ignore[0] = len(unique_vis) <= 0
                gt["ignore"] = gt_ignore.astype(bool).tolist()

        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        flag_no_part_score = False
        for dt in dts:
            # ignore all-zero keypoints and check part score
            if "keypoints" in p.iouType:
                if p.iouType == "keypoints_wholebody":
                    body_dt = dt["keypoints"]
                    foot_dt = dt["foot_kpts"]
                    face_dt = dt["face_kpts"]
                    lefthand_dt = dt["lefthand_kpts"]
                    righthand_dt = dt["righthand_kpts"]
                    wholebody_dt = body_dt + foot_dt + face_dt + lefthand_dt + righthand_dt
                    d = np.array(wholebody_dt)
                    k = np.count_nonzero(d[2::3] > 0)
                    if self.score_key not in dt:
                        dt[self.score_key] = dt["score"]
                        flag_no_part_score = True
                elif p.iouType == "keypoints_foot":
                    d = np.array(dt["foot_kpts"])
                    k = np.count_nonzero(d[2::3] > 0)
                    if self.score_key not in dt:
                        dt[self.score_key] = dt["score"]
                        flag_no_part_score = True
                elif p.iouType == "keypoints_face":
                    d = np.array(dt["face_kpts"])
                    k = np.count_nonzero(d[2::3] > 0)
                    if self.score_key not in dt:
                        dt[self.score_key] = dt["score"]
                        flag_no_part_score = True
                elif p.iouType == "keypoints_lefthand":
                    d = np.array(dt["lefthand_kpts"])
                    k = np.count_nonzero(d[2::3] > 0)
                    if self.score_key not in dt:
                        dt[self.score_key] = dt["score"]
                        flag_no_part_score = True
                elif p.iouType == "keypoints_righthand":
                    d = np.array(dt["righthand_kpts"])
                    k = np.count_nonzero(d[2::3] > 0)
                    if self.score_key not in dt:
                        dt[self.score_key] = dt["score"]
                        flag_no_part_score = True
                else:
                    d = np.array(dt["keypoints"])
                    k = np.count_nonzero(d[2::3] > 0)

                if not "visibilities" in dt:
                    # When visibility is not predicted, take confidence as visibility
                    dt["visibilities"] = d[2::3]

                if k == 0:
                    continue
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        if flag_no_part_score:
            warnings.warn("'{}' not found, use 'score' instead.".format(self.score_key))
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        """
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        """
        tic = time.time()
        print("Running per image evaluation...")
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = "segm" if p.useSegm == 1 else "bbox"
            print("useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType))
        print("Evaluate annotation type *{}*".format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == "segm" or p.iouType == "bbox":
            computeIoU = self.computeIoU
        elif "keypoints" in p.iouType:
            computeIoU = self.computeExtendedOks

        if self.match_by_bbox:
            print("Matching by bbox...")

        if self.extended_oks:
            print("Using extended OKS...")

        self.ious = {
            (imgId, catId): computeIoU(imgId, catId, original=not self.extended_oks)
            for imgId in p.imgIds
            for catId in catIds
        }

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [
            evaluateImg(
                imgId,
                catId,
                areaRng,
                maxDet,
                iou_i=iou_i,
                match_by_bbox=self.match_by_bbox,
                return_matching=False,
            )
            for catId in catIds
            for iou_i in range(len(self.gt_visibilities) + 1)
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]

        self.loc_similarities = np.array(self.loc_similarities)

        print("Loc similarity: {:.4f}".format(self.loc_similarities.mean()))

        self.matched_pairs = []
        for imgId in p.imgIds:
            img_eval = self.evaluateImg(
                imgId,
                1,
                [0, 1e5**2],
                maxDet,
                iou_i=0,
                return_matching=True,
                match_by_bbox=True,
            )
            if img_eval is None or "assigned_pairs" not in img_eval:
                continue
            self.matched_pairs.extend(img_eval["assigned_pairs"])

        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def computeIoU(self, imgId, catId):
        """
        Returns ious - [D x G] array of IoU values for all pairs of detections and gt instances.
        Where D is the number of detections and G is the number of gt intances.
        Detections are sortred from the highest to lowest score before computing `ious`.
        So rows in `ious` are ordered according to detection scores.
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d[self.score_key] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        if p.iouType == "segm":
            g = [g["segmentation"] for g in gt]
            d = [d["segmentation"] for d in dt]
        elif p.iouType == "bbox":
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
        else:
            raise Exception("unknown iouType for iou computation")

        # compute iou between each dt and gt region
        iscrowd = [int(o["iscrowd"]) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def computeExtendedOks(self, imgId, catId, original=False):

        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d[self.score_key] for d in dts], kind="mergesort")
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            print("Truncating detections to maxDets")
            dts = dts[0 : p.maxDets[-1]]
        if len(gts) == 0 or len(dts) == 0:
            return [[] for _ in range(len(self.gt_visibilities) + 1)]
        sigmas = self.sigmas
        vars = (sigmas * 2) ** 2
        k = len(sigmas)

        if original:
            padding = 1.0

        assert self.padding >= 1.0, "Padding must be greater than or equal to 1.0"

        # Prepare ious for each visibility level
        ious = [np.zeros((len(dts), len(gts))) for _ in self.gt_visibilities]
        # Plus the default v > 0 level
        ious.insert(0, np.zeros((len(dts), len(gts))))

        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):

            # Load the GT
            if p.iouType == "keypoints_wholebody":
                body_gt = gt["keypoints"]
                foot_gt = gt["foot_kpts"]
                face_gt = gt["face_kpts"]
                lefthand_gt = gt["lefthand_kpts"]
                righthand_gt = gt["righthand_kpts"]
                wholebody_gt = body_gt + foot_gt + face_gt + lefthand_gt + righthand_gt
                g = np.array(wholebody_gt)
            elif p.iouType == "keypoints_foot":
                g = np.array(gt["foot_kpts"])
            elif p.iouType == "keypoints_face":
                g = np.array(gt["face_kpts"])
            elif p.iouType == "keypoints_lefthand":
                g = np.array(gt["lefthand_kpts"])
            elif p.iouType == "keypoints_righthand":
                g = np.array(gt["righthand_kpts"])
            else:
                g = np.array(gt["keypoints"]).flatten()

            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            gt_in_img = vg < 3  # Visibility 3 means the keypoint is out of image
            self.TMP_COUNTER += (~gt_in_img).sum()

            # Count the number of keypoints visible for each visibility level
            vis_masks = [vg == vis for vis in self.gt_visibilities]
            # Plus the default v > 0 level
            vis_masks.insert(0, vg > 0)

            # create bounds for ignore regions(double the gt bbox)
            bb = gt["bbox"]

            if original:
                x0 = bb[0] - bb[2]
                x1 = bb[0] + bb[2] * 2
                y0 = bb[1] - bb[3]
                y1 = bb[1] + bb[3] * 2
            else:
                bb_xyxy = np.array([bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]])
                x0, y0, x1, y1 = fix_bbox_aspect_ratio(bb_xyxy, padding=self.padding, bbox_format="xyxy")

            for i, dt in enumerate(dts):

                # Load the pred
                if p.iouType == "keypoints_wholebody":
                    body_dt = dt["keypoints"]
                    foot_dt = dt["foot_kpts"]
                    face_dt = dt["face_kpts"]
                    lefthand_dt = dt["lefthand_kpts"]
                    righthand_dt = dt["righthand_kpts"]
                    wholebody_dt = body_dt + foot_dt + face_dt + lefthand_dt + righthand_dt
                    d = np.array(wholebody_dt)
                elif p.iouType == "keypoints_foot":
                    d = np.array(dt["foot_kpts"])
                elif p.iouType == "keypoints_face":
                    d = np.array(dt["face_kpts"])
                elif p.iouType == "keypoints_lefthand":
                    d = np.array(dt["lefthand_kpts"])
                elif p.iouType == "keypoints_righthand":
                    d = np.array(dt["righthand_kpts"])
                else:
                    d = np.array(dt["keypoints"])

                xd = d[0::3]
                yd = d[1::3]
                cd = np.clip(d[2::3], 0, 1)
                if self.confidence_thr is not None:
                    cd[cd < self.confidence_thr] = 0
                    cd[cd >= self.confidence_thr] = 1
                    cd = cd.astype(int)
                vd = np.array(dt["visibilities"])

                # GT visibility is 0/1
                vg = (vg == 2).astype(int)

                for vis_level in range(len(vis_masks)):
                    iou = ious[vis_level]
                    vis_mask = vis_masks[vis_level]

                    k1 = np.count_nonzero(vis_mask)
                    gt_ignore = gt["ignore"][vis_level]

                    assert not (gt_ignore and k1 > 0), "k1 is negative but gt is not ignored"

                    if k1 > 0:

                        # Distance between prediction and GT
                        dx = xd - xg
                        dy = yd - yg
                        dist_sq = dx**2 + dy**2

                        if not original:
                            # Distance of prediction to the closes bbox edge
                            dxe_pred = np.min((xd - x0, x1 - xd), axis=0)
                            dye_pred = np.min((yd - y0, y1 - yd), axis=0)
                            dist_e_pred = dxe_pred**2 + dye_pred**2

                            # Distance of GT to the closest bbox edge
                            dxe_gt = np.min((xg - x0, x1 - xg), axis=0)
                            dye_gt = np.min((yg - y0, y1 - yg), axis=0)
                            dist_e_gt = dxe_gt**2 + dye_gt**2

                            # Pred is in AM, GT is out --> d(pred, e)
                            mask = ~gt_in_img & (cd == 1)
                            dist_sq[mask] = dist_e_pred[mask]

                            # Pred is out of AM, GT is in --> d(gt, e)
                            mask = gt_in_img & (cd == 0)
                            dist_sq[mask] = dist_e_gt[mask]

                            # Both pred and GT are out --> 0
                            mask = ~gt_in_img & (cd == 0)
                            dist_sq[mask] = 0

                    else:
                        # If no GT keypoints for this visibility level, measure distance to
                        # the bbox or extended bbox
                        z = np.zeros((k))
                        dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                        dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
                        dist_sq = dx**2 + dy**2

                    # Normalize by area and sigmas
                    tmparea = gt["bbox"][3] * gt["bbox"][2] * 0.53
                    if self.use_area:
                        tmparea = gt["area"]
                    e = (dist_sq) / vars / (tmparea + np.spacing(1)) / 2

                    if k1 > 0:
                        e = e[vis_mask]

                    loc_oks = np.sum(np.exp(-e)) / e.shape[0]

                    iou[i, j] = loc_oks

        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet, iou_i=0, return_matching=False, match_by_bbox=False):
        """
        perform evaluation for single category and image
        :return: dict (single image results)
        """
        p = self.params
        if return_matching:
            iouThrs = np.array([0.1])
        else:
            iouThrs = p.iouThrs
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if "area" not in g or not self.use_area:
                tmp_area = g["bbox"][2] * g["bbox"][3] * 0.53
            else:
                tmp_area = g["area"]
            if g["ignore"][iou_i] or (tmp_area < aRng[0] or tmp_area > aRng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d[self.score_key] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o["iscrowd"]) for o in gt]

        # Load pre-computed ious
        ious = []
        for i in range(len(self.gt_visibilities) + 1):
            if len(self.ious[imgId, catId][i]) > 0:
                ious.append(self.ious[imgId, catId][i][:, gtind])
            else:
                ious.append(self.ious[imgId, catId][i])

        T = len(iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.ones((T, G), dtype=np.int64) * -1
        dtm = np.ones((T, D), dtype=np.int64) * -1
        gtIg = np.array([g["_ignore"] for g in gt])
        dtIg = np.zeros((T, D))

        # Additional variables for per-keypoint OKS
        assigned_pairs = []

        if return_matching and match_by_bbox:
            for tind, t in enumerate(iouThrs):
                for dind, d in enumerate(dt):
                    d_bbox = np.array(d["bbox"])
                    d_center = d_bbox[:2] + d_bbox[2:] / 2
                    for gind, g in enumerate(gt):
                        g_bbox = np.array(g["bbox"])
                        g_center = g_bbox[:2] + g_bbox[2:] / 2
                        if abs(d_center - g_center).sum() < 2:
                            this_iou = ious[iou_i][dind, gind] if not g["ignore"][iou_i] else np.nan
                            assigned_pairs.append((d, g, this_iou))
                            dtIg[tind, dind] = gtIg[gind]
                            dtm[tind, dind] = gt[gind]["id"]
                            gtm[tind, gind] = d["id"]
                            break
            # set unmatched detections outside of area range to ignore
            a = np.array([d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]).reshape((1, len(dt)))
            dtIg = np.logical_or(dtIg, np.logical_and(dtm < 0, np.repeat(a, T, 0)))
            image_results = {
                "image_id": imgId,
                "category_id": catId,
                "aRng": aRng,
                "maxDet": maxDet,
                "dtIds": [d["id"] for d in dt],
                "gtIds": [g["id"] for g in gt],
                "dtMatches": dtm,
                "gtMatches": gtm,
                "assigned_pairs": assigned_pairs,
                "dtScores": [d[self.score_key] for d in dt],
                "gtIgnore": gtIg,
                "dtIgnore": dtIg,
                "gtIndices": gtind,
            }

        else:
            iou = ious[iou_i]
            gtm = np.ones((T, G), dtype=np.int64) * -1
            dtm = np.ones((T, D), dtype=np.int64) * -1
            gtIg = np.array([g["_ignore"] for g in gt])
            dtIg = np.zeros((T, D))
            assigned_pairs = []
            if len(iou):
                for tind, t in enumerate(iouThrs):

                    for dind, d in enumerate(dt):
                        # information about best match so far (m=-1 -> unmatched)
                        curr_iou = min([t, 1 - 1e-10])
                        m = -1

                        if match_by_bbox:
                            closest_dist = 20
                            d_bbox = np.array(d["bbox"])
                            d_center = d_bbox[:2] + d_bbox[2:] / 2
                            for gind, g in enumerate(gt):
                                g_bbox = np.array(g["bbox"])
                                g_center = g_bbox[:2] + g_bbox[2:] / 2

                                # if this gt already matched, and not a crowd, continue
                                if gtm[tind, gind] >= 0 and not iscrowd[gind]:
                                    continue
                                # if dt matched to reg gt, and on ignore gt, stop
                                # since all the rest of g's are ignored as well because of the prior sorting
                                if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                                    break

                                if iou[dind, gind] < t:
                                    continue

                                abs_dist = abs(d_center - g_center).sum()
                                if abs_dist < closest_dist:
                                    closest_dist = abs_dist
                                    m = gind
                                    curr_iou = iou[dind, gind]

                        else:
                            for gind, g in enumerate(gt):
                                # if this gt already matched, and not a crowd, continue
                                if gtm[tind, gind] >= 0 and not iscrowd[gind]:
                                    continue
                                # if dt matched to reg gt, and on ignore gt, stop
                                # since all the rest of g's are ignored as well because of the prior sorting
                                if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                                    break
                                # continue to next gt unless better match made
                                if iou[dind, gind] < curr_iou:
                                    continue
                                # if match successful and best so far, store appropriately
                                curr_iou = iou[dind, gind]
                                m = gind

                        if return_matching and not match_by_bbox:
                            assigned_pairs.append((d, gt[m], curr_iou if (m != -1 and gtIg[m] != 1) else np.nan))

                        if m == -1:
                            continue
                        # if match made store id of match for both dt and gt
                        self.loc_similarities.append(curr_iou)
                        dtIg[tind, dind] = gtIg[m]
                        dtm[tind, dind] = gt[m]["id"]
                        gtm[tind, m] = d["id"]
            # set unmatched detections outside of area range to ignore
            a = np.array([d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]).reshape((1, len(dt)))
            dtIg = np.logical_or(dtIg, np.logical_and(dtm < 0, np.repeat(a, T, 0)))
            if np.all(gtIg):
                dtIg[:] = True

            # store results for given image and category
            image_results = {
                "image_id": imgId,
                "category_id": catId,
                "aRng": aRng,
                "maxDet": maxDet,
                "dtIds": [d["id"] for d in dt],
                "gtIds": [g["id"] for g in gt],
                "dtMatches": dtm,
                "gtMatches": gtm,
                "assigned_pairs": assigned_pairs,
                "dtScores": [d[self.score_key] for d in dt],
                "gtIgnore": gtIg,
                "dtIgnore": dtIg,
                "gtIndices": gtind,
            }

        return image_results

    def accumulate(self, p=None):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        """
        print("Accumulating evaluation results...")
        tic = time.time()
        if not self.evalImgs:
            print("Please run evaluate() first")
        # allows input customized parameters
        if p is None:
            p = self.params

        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        V = len(self.gt_visibilities) + 1
        precision = -np.ones((T, V, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, V, K, A, M))
        scores = -np.ones((T, V, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        counter = 0

        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0 * V
            for v in range(V):
                Nv = v * A0 * I0
                for a, a0 in enumerate(a_list):
                    Na = a0 * I0
                    for m, maxDet in enumerate(m_list):
                        E = [self.evalImgs[Nk + Nv + Na + i] for i in i_list]
                        E = [e for e in E if not e is None]
                        if len(E) == 0:
                            continue
                        counter += 1

                        dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])
                        # different sorting method generates slightly different results.
                        # mergesort is used to be consistent as Matlab implementation.
                        if self.match_by_bbox:
                            # dtIds = np.concatenate([e['dtIds'][0:maxDet] for e in E])
                            # inds = np.argsort(dtIds, kind='mergesort')
                            inds = np.argsort(-dtScores, kind="mergesort")
                        else:
                            inds = np.argsort(-dtScores, kind="mergesort")
                        dtScoresSorted = dtScores[inds]

                        dtm = np.concatenate([e["dtMatches"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                        dtIg = np.concatenate([e["dtIgnore"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                        gtIg = np.concatenate([e["gtIgnore"] for e in E])
                        npig = np.count_nonzero(gtIg == 0)
                        if npig == 0:
                            continue
                        # https://github.com/cocodataset/cocoapi/pull/332/
                        tps = np.logical_and(dtm >= 0, np.logical_not(dtIg))
                        fps = np.logical_and(dtm < 0, np.logical_not(dtIg))
                        tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float64)
                        fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float64)
                        for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                            tp = np.array(tp)
                            fp = np.array(fp)
                            nd = len(tp)
                            rc = tp / npig
                            pr = tp / (fp + tp + np.spacing(1))
                            q = np.zeros((R,))
                            ss = np.zeros((R,))

                            if nd:
                                recall[t, v, k, a, m] = rc[-1]
                            else:
                                recall[t, v, k, a, m] = 0

                            # numpy is slow without cython optimization for accessing elements
                            # use python array gets significant speed improvement
                            pr = pr.tolist()
                            q = q.tolist()

                            for i in range(nd - 1, 0, -1):
                                if pr[i] > pr[i - 1]:
                                    pr[i - 1] = pr[i]

                            inds = np.searchsorted(rc, p.recThrs, side="left")
                            try:
                                for ri, pi in enumerate(inds):
                                    q[ri] = pr[pi]
                                    ss[ri] = dtScoresSorted[pi]
                            except:
                                pass
                            precision[t, v, :, k, a, m] = np.array(q)
                            scores[t, v, :, k, a, m] = np.array(ss)

        self.eval = {
            "params": p,
            "counts": [T, V, R, K, A, M],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }

        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100, visibility=None):
            assert visibility in self.gt_visibilities + [None]
            p = self.params
            # https://github.com/cocodataset/cocoapi/pull/405
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} | vis={:>5s} ] = {: 0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            v = 0 if visibility is None else self.gt_visibilities.index(visibility) + 1

            if v > 0:
                visStr = "[{:d}]".format(int(visibility))
            elif self.extended_oks:
                visStr = "[>0]"
            else:
                visStr = "[1,2]"

            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, v, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, v, :, aind, mind]

            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, visStr, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stat_names = [None] * stats.shape[0]

            stats[0] = _summarize(1)
            stat_names[0] = "AP"

            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stat_names[1] = "AP .5"

            stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            stat_names[2] = "AP .75"

            stats[3] = _summarize(1, areaRng="small", maxDets=self.params.maxDets[2])
            stat_names[3] = "AP (S)"

            stats[4] = _summarize(1, areaRng="medium", maxDets=self.params.maxDets[2])
            stat_names[4] = "AP (M)"

            stats[5] = _summarize(1, areaRng="large", maxDets=self.params.maxDets[2])
            stat_names[5] = "AP (L)"

            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stat_names[6] = "AR (maxDets={})".format(self.params.maxDets[0])

            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stat_names[7] = "AR (maxDets={})".format(self.params.maxDets[1])

            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stat_names[8] = "AR (maxDets={})".format(self.params.maxDets[2])

            stats[9] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
            stat_names[9] = "AR (S)"

            stats[10] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
            stat_names[10] = "AR (M)"

            stats[11] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
            stat_names[11] = "AR (L)"

            return stats, stat_names

        def _summarizeKps_crowd():
            # Adapted from https://github.com/Jeff-sjtu/CrowdPose
            # @article{li2018crowdpose,
            #   title={CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark},
            #   author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
            #   journal={arXiv preprint arXiv:1812.00324},
            #   year={2018}
            # }
            stats = np.zeros((9,))
            stat_names = [None] * stats.shape[0]
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=0.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=0.75)
            stats[3] = _summarize(0, maxDets=20)
            stats[4] = _summarize(0, maxDets=20, iouThr=0.5)
            stats[5] = _summarize(0, maxDets=20, iouThr=0.75)
            type_result = self.get_type_result(first=0.2, second=0.8)

            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | type={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision"
            typeStr = "(AP)"
            iouStr = "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
            print(iStr.format(titleStr, typeStr, iouStr, "easy", 20, type_result[0]))
            print(iStr.format(titleStr, typeStr, iouStr, "medium", 20, type_result[1]))
            print(iStr.format(titleStr, typeStr, iouStr, "hard", 20, type_result[2]))
            stats[6] = type_result[0]
            stats[7] = type_result[1]
            stats[8] = type_result[2]

            return stats, stat_names

        def _summarizeKps(eval=None):
            num_vis = len(self.gt_visibilities)
            stats = np.zeros((11 + num_vis,))
            stat_names = [None] * stats.shape[0]

            stats[0] = _summarize(1, maxDets=20)
            stat_names[0] = "AP"

            for vi, v in enumerate(self.gt_visibilities):
                stats[1 + vi] = _summarize(1, maxDets=20, visibility=v)
                stat_names[1 + vi] = "AP (v={:d})".format(v)

            stats[1 + num_vis] = _summarize(1, maxDets=20, iouThr=0.5)
            stat_names[1 + num_vis] = "AP .5"

            stats[2 + num_vis] = _summarize(1, maxDets=20, iouThr=0.75)
            stat_names[2 + num_vis] = "AP .75"

            stats[3 + num_vis] = _summarize(1, maxDets=20, areaRng="medium")
            stat_names[3 + num_vis] = "AP (M)"

            stats[4 + num_vis] = _summarize(1, maxDets=20, areaRng="large")
            stat_names[4 + num_vis] = "AP (L)"

            stats[5 + num_vis] = _summarize(0, maxDets=20)
            stat_names[5 + num_vis] = "AR"

            stats[6 + num_vis] = _summarize(0, maxDets=20, iouThr=0.5)
            stat_names[6 + num_vis] = "AR .5"

            stats[7 + num_vis] = _summarize(0, maxDets=20, iouThr=0.75)
            stat_names[7 + num_vis] = "AR .75"

            stats[8 + num_vis] = _summarize(0, maxDets=20, areaRng="medium")
            stat_names[8 + num_vis] = "AR (M)"

            stats[9 + num_vis] = _summarize(0, maxDets=20, areaRng="large")
            stat_names[9 + num_vis] = "AR (L)"

            stats[10 + num_vis] = np.mean(self.loc_similarities)
            stat_names[10 + num_vis] = "OKS"

            return stats, stat_names

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets
        elif iouType == "keypoints_crowd":
            summarize = _summarizeKps_crowd
        elif "keypoints" in iouType:
            summarize = _summarizeKps

        self.stats, self.stats_names = summarize()

    def __str__(self):
        self.summarize()

    def get_type_result(self, first=0.01, second=0.85):
        gt_file, resfile = self.anno_file
        easy, mid, hard = self.split(gt_file, first, second)
        res = []
        nullwrite = NullWriter()
        oldstdout = sys.stdout
        sys.stdout = nullwrite
        for curr_type in [easy, mid, hard]:
            curr_list = curr_type
            self.params.imgIds = curr_list
            self.evaluate()
            self.accumulate()
            score = self.eval["precision"][:, :, :, 0, :]
            res.append(round(np.mean(score), 4))
        sys.stdout = oldstdout
        return res

    def split(serlf, gt_file, first=0.01, second=0.85):
        import json

        data = json.load(open(gt_file, "r"))
        easy = []
        mid = []
        hard = []
        for item in data["images"]:
            if item["crowdIndex"] < first:
                easy.append(item["id"])
            elif item["crowdIndex"] < second:
                mid.append(item["id"])
            else:
                hard.append(item["id"])
        return easy, mid, hard


class Params:
    """
    Params for coco evaluation api
    """

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        self.recThrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2], [96**2, 1e5**2]]
        self.areaRngLbl = ["all", "small", "medium", "large"]
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        self.recThrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0**2, 1e5**2], [32**2, 96**2], [96**2, 1e5**2]]
        self.areaRngLbl = ["all", "medium", "large"]
        # self.areaRng = [[0 ** 2, 1e5 ** 2]]
        # self.areaRngLbl = ['all']
        self.useCats = 1

    def __init__(self, iouType="segm"):
        if iouType == "segm" or iouType == "bbox":
            self.setDetParams()
        elif "keypoints" in iouType:
            self.setKpParams()
        else:
            raise Exception("iouType not supported")
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
