# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict
from typing import Dict, Optional, Sequence
import traceback

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MessageHub, MMLogger, print_log
from xtcocotools.coco import COCO
# from xtcocotools.cocoeval import COCOeval
from ._cocoeval import COCOeval

from mmpose.registry import METRICS
from mmpose.structures.bbox import bbox_xyxy2xywh
from mmpose.structures.keypoint import find_min_padding_exact, fix_bbox_aspect_ratio
from ..functional import (oks_nms, soft_oks_nms, transform_ann, transform_pred,
                          transform_sigmas)

from . import _mask as maskUtils


EVAL_HEATMAPS = False
EVAL_CALIBRATION = False


@METRICS.register_module()
class CocoMetric(BaseMetric):
    """COCO pose estimation task evaluation metric.

    Evaluate AR, AP, and mAP for keypoint detection tasks. Support COCO
    dataset and other datasets in COCO format. Please refer to
    `COCO keypoint evaluation <https://cocodataset.org/#keypoints-eval>`__
    for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None
        use_area (bool): Whether to use ``'area'`` message in the annotations.
            If the ground truth annotations (e.g. CrowdPose, AIC) do not have
            the field ``'area'``, please set ``use_area=False``.
            Defaults to ``True``
        iou_type (str): The same parameter as `iouType` in
            :class:`xtcocotools.COCOeval`, which can be ``'keypoints'``, or
            ``'keypoints_crowd'`` (used in CrowdPose dataset).
            Defaults to ``'keypoints'``
        score_mode (str): The mode to score the prediction results which
            should be one of the following options:

                - ``'bbox'``: Take the score of bbox as the score of the
                    prediction results.
                - ``'bbox_keypoint'``: Use keypoint score to rescore the
                    prediction results.
                - ``'bbox_rle'``: Use rle_score to rescore the
                    prediction results.

            Defaults to ``'bbox_keypoint'`
        keypoint_score_thr (float): The threshold of keypoint score. The
            keypoints with score lower than it will not be included to
            rescore the prediction results. Valid only when ``score_mode`` is
            ``bbox_keypoint``. Defaults to ``0.2``
        nms_mode (str): The mode to perform Non-Maximum Suppression (NMS),
            which should be one of the following options:

                - ``'oks_nms'``: Use Object Keypoint Similarity (OKS) to
                    perform NMS.
                - ``'soft_oks_nms'``: Use Object Keypoint Similarity (OKS)
                    to perform soft NMS.
                - ``'none'``: Do not perform NMS. Typically for bottomup mode
                    output.

            Defaults to ``'oks_nms'`
        nms_thr (float): The Object Keypoint Similarity (OKS) threshold
            used in NMS when ``nms_mode`` is ``'oks_nms'`` or
            ``'soft_oks_nms'``. Will retain the prediction results with OKS
            lower than ``nms_thr``. Defaults to ``0.9``
        format_only (bool): Whether only format the output results without
            doing quantitative evaluation. This is designed for the need of
            test submission when the ground truth annotations are absent. If
            set to ``True``, ``outfile_prefix`` should specify the path to
            store the output results. Defaults to ``False``
        pred_converter (dict, optional): Config dictionary for the prediction
            converter. The dictionary has the same parameters as
            'KeypointConverter'. Defaults to None.
        gt_converter (dict, optional): Config dictionary for the ground truth
            converter. The dictionary has the same parameters as
            'KeypointConverter'. Defaults to None.
        outfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., ``'a/b/prefix'``.
            If not specified, a temp file will be created. Defaults to ``None``
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Defaults to ``'cpu'``
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Defaults to ``None``
    """
    default_prefix: Optional[str] = 'coco'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 use_area: bool = True,
                 iou_type: str = 'keypoints',
                 score_mode: str = 'bbox_keypoint',
                 score_thresh_type: str = 'score',
                 keypoint_score_thr: float = 0.2,
                 nms_mode: str = 'oks_nms',
                 nms_thr: float = 0.9,
                 format_only: bool = False,
                 pred_converter: Dict = None,
                 gt_converter: Dict = None,
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 extended: list = [False],
                 match_by_bbox: list = [False],
                 ignore_border_points: list = [False],
                 ignore_stats: list = [],
                 padding: float = 1.25) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        # initialize coco helper with the annotation json file
        # if ann_file is not specified, initialize with the converted dataset
        if ann_file is not None:
            with get_local_path(ann_file) as local_path:
                self.coco = COCO(local_path)
        else:
            self.coco = None

        self.use_area = use_area
        self.iou_type = iou_type

        allowed_score_modes = ['bbox', 'bbox_keypoint', 'bbox_rle', 'keypoint']
        if score_mode not in allowed_score_modes:
            raise ValueError(
                "`score_mode` should be one of 'bbox', 'bbox_keypoint', "
                f"'bbox_rle', but got {score_mode}")
        self.score_mode = score_mode
        self.keypoint_score_thr = keypoint_score_thr
        if score_thresh_type not in ['score', 'prob']:
            raise ValueError(
                "'score_thresh_type' should be one of 'score' or 'prob'"
            )
        self.score_thresh_type = score_thresh_type

        allowed_nms_modes = ['oks_nms', 'soft_oks_nms', 'none']
        if nms_mode not in allowed_nms_modes:
            raise ValueError(
                "`nms_mode` should be one of 'oks_nms', 'soft_oks_nms', "
                f"'none', but got {nms_mode}")
        self.nms_mode = nms_mode
        self.nms_thr = nms_thr

        if format_only:
            assert outfile_prefix is not None, '`outfile_prefix` can not be '\
                'None when `format_only` is True, otherwise the result file '\
                'will be saved to a temp directory which will be cleaned up '\
                'in the end.'
        elif ann_file is not None:
            # do evaluation only if the ground truth annotations exist
            assert 'annotations' in load(ann_file), \
                'Ground truth annotations are required for evaluation '\
                'when `format_only` is False.'

        self.format_only = format_only
        self.outfile_prefix = outfile_prefix
        self.pred_converter = pred_converter
        self.gt_converter = gt_converter

        len_params = max(len(extended), len(match_by_bbox))
        if len(extended) == 1 and len_params > 1:
            extended = extended * len_params
        if len(match_by_bbox) == 1 and len_params > 1:
            match_by_bbox = match_by_bbox * len_params
        assert len(extended) == len(match_by_bbox), \
            'The length of `extended` and `match_by_bbox` should be the same.'
        assert len(extended) >= 1, \
            'The length of `extended` and `match_by_bbox` should be at least 1.'
        self.extended = extended
        self.match_by_bbox = match_by_bbox
        self.ignore_border_points = ignore_border_points

        self.ignore_stats = ignore_stats
        self.prob_thr = -1
        self.has_probability = True
        self.padding = padding

        self._compute_min_padding_in_coco()

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the metric."""
        if self.gt_converter is not None:
            dataset_meta['sigmas'] = transform_sigmas(
                dataset_meta['sigmas'], self.gt_converter['num_keypoints'],
                self.gt_converter['mapping'])
            dataset_meta['num_keypoints'] = len(dataset_meta['sigmas'])
        self._dataset_meta = dataset_meta

        if self.coco is None:
            message = MessageHub.get_current_instance()
            ann_file = message.get_info(
                f"{dataset_meta['dataset_name']}_ann_file", None)
            if ann_file is not None:
                with get_local_path(ann_file) as local_path:
                    self.coco = COCO(local_path)
                print_log(
                    f'CocoMetric for dataset '
                    f"{dataset_meta['dataset_name']} has successfully "
                    f'loaded the annotation file from {ann_file}', 'current')

    def _compute_min_padding_in_coco(self):
        """Compute the minimum padding in COCO format."""
        if self.coco is None:
            return
        
        for _, ann in self.coco.anns.items():
            if 'pad_to_contain' in ann.keys():
                continue

            kpts = np.array(ann['keypoints']).reshape(-1, 3)
            bbox = np.array(ann['bbox']).flatten()
            min_padding = find_min_padding_exact(bbox, kpts)
            ann['pad_to_contain'] = min_padding

        return

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model, each of which has the following keys:

                - 'id': The id of the sample
                - 'img_id': The image_id of the sample
                - 'pred_instances': The prediction results of instance(s)
        """
        self.results_len = len(self.results)
        for data_sample in data_samples:
            if 'pred_instances' not in data_sample:
                raise ValueError(
                    '`pred_instances` are required to process the '
                    f'predictions results in {self.__class__.__name__}. ')

            # keypoints.shape: [N, K, 2],
            # N: number of instances, K: number of keypoints
            # for topdown-style output, N is usually 1, while for
            # bottomup-style output, N is the number of instances in the image
            keypoints = data_sample['pred_instances']['keypoints']
            N, K, _ = keypoints.shape
            # [N, K], the scores for all keypoints of all instances
            keypoint_scores = data_sample['pred_instances']['keypoint_scores']
            assert keypoint_scores.shape == keypoints.shape[:2]
            
            if 'keypoints_visible' in data_sample['pred_instances']:
                keypoints_visible = data_sample['pred_instances']['keypoints_visible']
            else:
                keypoints_visible = keypoint_scores.copy()
            
            if 'keypoints_probs' in data_sample['pred_instances']:
                keypoints_probs = data_sample['pred_instances']['keypoints_probs']
                # keypoints_probs = keypoint_scores.copy()
            else:
                self.has_probability = False
                keypoints_probs = keypoint_scores.copy()

            if 'keypoints_oks' in data_sample['pred_instances']:
                keypoints_oks = data_sample['pred_instances']['keypoints_oks']
                # keypoints_oks = keypoint_scores.copy()
            else:
                keypoints_oks = keypoint_scores.copy()

            if 'keypoints_error' in data_sample['pred_instances']:
                keypoints_error = data_sample['pred_instances']['keypoints_error']
            else:
                keypoints_error = keypoint_scores.copy()

            if EVAL_HEATMAPS:
                if 'pred_fields' in data_sample and 'heatmaps' in data_sample['pred_fields']:
                    heatmaps = data_sample['pred_fields']['heatmaps']
                else:
                    heatmaps = np.ones((N, K, 64, 48)) * np.nan
                heatmaps = heatmaps.reshape((-1, K, 64, 48))

            # parse prediction results
            pred = dict()
            pred['id'] = data_sample['id']
            pred['img_id'] = data_sample['img_id']

            pred['keypoints'] = keypoints
            pred['keypoint_scores'] = keypoint_scores
            pred['keypoints_visible'] = keypoints_visible
            pred['keypoint_probs'] = keypoints_probs
            pred['keypoint_oks'] = keypoints_oks
            pred['keypoint_error'] = keypoints_error
            if EVAL_HEATMAPS:
                pred['heatmaps'] = heatmaps
            pred['category_id'] = data_sample.get('category_id', 1)
            if 'bboxes' in data_sample['pred_instances']:
                pred['bbox'] = bbox_xyxy2xywh(
                    data_sample['pred_instances']['bboxes'])

            if 'bbox_scores' in data_sample['pred_instances']:
                # some one-stage models will predict bboxes and scores
                # together with keypoints
                bbox_scores = data_sample['pred_instances']['bbox_scores']
            elif ('bbox_scores' not in data_sample['gt_instances']
                  or len(data_sample['gt_instances']['bbox_scores']) !=
                  len(keypoints)):
                # bottom-up models might output different number of
                # instances from annotation
                bbox_scores = np.ones(len(keypoints))
            else:
                # top-down models use detected bboxes, the scores of which
                # are contained in the gt_instances
                bbox_scores = data_sample['gt_instances']['bbox_scores']
            pred['bbox_scores'] = bbox_scores

            # get area information
            if 'bbox_scales' in data_sample['gt_instances']:
                pred['areas'] = np.prod(
                    data_sample['gt_instances']['bbox_scales'], axis=1)

            # parse gt
            gt = dict()
            if self.coco is None:
                gt['width'] = data_sample['ori_shape'][1]
                gt['height'] = data_sample['ori_shape'][0]
                gt['img_id'] = data_sample['img_id']
                if self.iou_type == 'keypoints_crowd':
                    assert 'crowd_index' in data_sample, \
                        '`crowd_index` is required when `self.iou_type` is ' \
                        '`keypoints_crowd`'
                    gt['crowd_index'] = data_sample['crowd_index']
                assert 'raw_ann_info' in data_sample, \
                    'The row ground truth annotations are required for ' \
                    'evaluation when `ann_file` is not provided'
                anns = data_sample['raw_ann_info']
                gt['raw_ann_info'] = anns if isinstance(anns, list) else [anns]

            # add converted result to the results list
            self.results.append((pred, gt))
        processed_len = len(self.results) - self.results_len
        if processed_len != len(data_samples):
            print(f'Warning: {processed_len} samples are processed, ')
            print(f'but {len(data_samples)} samples are provided.')
        
    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset. Each dict
                contains the ground truth information about the data sample.
                Required keys of the each `gt_dict` in `gt_dicts`:
                    - `img_id`: image id of the data sample
                    - `width`: original image width
                    - `height`: original image height
                    - `raw_ann_info`: the raw annotation information
                Optional keys:
                    - `crowd_index`: measure the crowding level of an image,
                        defined in CrowdPose dataset
                It is worth mentioning that, in order to compute `CocoMetric`,
                there are some required keys in the `raw_ann_info`:
                    - `id`: the id to distinguish different annotations
                    - `image_id`: the image id of this annotation
                    - `category_id`: the category of the instance.
                    - `bbox`: the object bounding box
                    - `keypoints`: the keypoints cooridinates along with their
                        visibilities. Note that it need to be aligned
                        with the official COCO format, e.g., a list with length
                        N * 3, in which N is the number of keypoints. And each
                        triplet represent the [x, y, visible] of the keypoint.
                    - `iscrowd`: indicating whether the annotation is a crowd.
                        It is useful when matching the detection results to
                        the ground truth.
                There are some optional keys as well:
                    - `area`: it is necessary when `self.use_area` is `True`
                    - `num_keypoints`: it is necessary when `self.iou_type`
                        is set as `keypoints_crowd`.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".
        Returns:
            str: The filename of the json file.
        """
        image_infos = []
        annotations = []
        img_ids = []
        ann_ids = []

        for gt_dict in gt_dicts:
            # filter duplicate image_info
            if gt_dict['img_id'] not in img_ids:
                image_info = dict(
                    id=gt_dict['img_id'],
                    width=gt_dict['width'],
                    height=gt_dict['height'],
                )
                if self.iou_type == 'keypoints_crowd':
                    image_info['crowdIndex'] = gt_dict['crowd_index']

                image_infos.append(image_info)
                img_ids.append(gt_dict['img_id'])

            # filter duplicate annotations
            for ann in gt_dict['raw_ann_info']:
                if ann is None:
                    # during evaluation on bottom-up datasets, some images
                    # do not have instance annotation
                    continue

                annotation = dict(
                    id=ann['id'],
                    image_id=ann['image_id'],
                    category_id=ann['category_id'],
                    bbox=ann['bbox'],
                    keypoints=ann['keypoints'],
                    iscrowd=ann['iscrowd'],
                )
                if self.use_area:
                    assert 'area' in ann, \
                        '`area` is required when `self.use_area` is `True`'
                    annotation['area'] = ann['area']

                if self.iou_type == 'keypoints_crowd':
                    assert 'num_keypoints' in ann, \
                        '`num_keypoints` is required when `self.iou_type` ' \
                        'is `keypoints_crowd`'
                    annotation['num_keypoints'] = ann['num_keypoints']

                annotations.append(annotation)
                ann_ids.append(ann['id'])

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmpose CocoMetric.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=self.dataset_meta['CLASSES'],
            licenses=None,
            annotations=annotations,
        )
        converted_json_path = f'{outfile_prefix}.gt.json'
        dump(coco_json, converted_json_path, sort_keys=True, indent=4)
        return converted_json_path

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split prediction and gt list
        preds, gts = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self.coco is None:
            # use converted gt json file to initialize coco helper
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self.coco = COCO(coco_json_path)
        if self.gt_converter is not None:
            for id_, ann in self.coco.anns.items():
                self.coco.anns[id_] = transform_ann(
                    ann, self.gt_converter['num_keypoints'],
                    self.gt_converter['mapping'])

        kpts = defaultdict(list)

        # group the preds by img_id
        for pred in preds:
            img_id = pred['img_id']

            if self.pred_converter is not None:
                pred = transform_pred(pred,
                                      self.pred_converter['num_keypoints'],
                                      self.pred_converter['mapping'])

            for idx, keypoints in enumerate(pred['keypoints']):

                instance = {
                    'id': pred['id'],
                    'img_id': pred['img_id'],
                    'category_id': pred['category_id'],
                    'keypoints': keypoints,
                    'keypoint_scores': pred['keypoint_scores'][idx],
                    'bbox_score': pred['bbox_scores'][idx],
                    'keypoints_visible': pred['keypoints_visible'][idx],
                    'keypoint_probs': pred['keypoint_probs'][idx],
                    'keypoint_oks': pred['keypoint_oks'][idx],
                    'keypoint_error': pred['keypoint_error'][idx],
                }
                if EVAL_HEATMAPS:
                    instance['heatmaps'] = pred['heatmaps'][idx]
                if 'bbox' in pred:
                    instance['bbox'] = pred['bbox'][idx]
                    diagonal = np.sqrt(
                        instance['bbox'][2]**2 + instance['bbox'][3]**2)
                if 'areas' in pred:
                    instance['area'] = pred['areas'][idx]
                    diagonal = np.sqrt(instance['area'])
                else:
                    # use keypoint to calculate bbox and get area
                    area = (
                        np.max(keypoints[:, 0]) - np.min(keypoints[:, 0])) * (
                            np.max(keypoints[:, 1]) - np.min(keypoints[:, 1]))
                    instance['area'] = area
                    diagonal = np.sqrt(area)
                
                kpts[img_id].append(instance)

        # sort keypoint results according to id and remove duplicate ones
        kpts = self._sort_and_unique_bboxes(kpts, key='id')

        # score the prediction results according to `score_mode`
        # and perform NMS according to `nms_mode`
        valid_kpts = defaultdict(list)
        if self.pred_converter is not None:
            num_keypoints = self.pred_converter['num_keypoints']
        else:
            num_keypoints = self.dataset_meta['num_keypoints']
        for img_id, instances in kpts.items():
            for instance in instances:
                # concatenate the keypoint coordinates and scores
                instance['keypoints'] = np.concatenate([
                    instance['keypoints'], instance['keypoint_probs'][:, None]
                ],
                                                       axis=-1)
                if self.score_mode == 'bbox':
                    instance['score'] = instance['bbox_score']
                elif self.score_mode == 'keypoint':
                    instance['score'] = np.mean(instance['keypoint_scores'])
                else:
                    bbox_score = instance['bbox_score']
                    if self.score_mode == 'bbox_rle':
                        keypoint_scores = instance['keypoint_scores']
                        instance['score'] = float(bbox_score +
                                                  np.mean(keypoint_scores) +
                                                  np.max(keypoint_scores))

                    else:  # self.score_mode == 'bbox_keypoint':
                        mean_kpt_score = 0
                        valid_num = 0
                        for kpt_idx in range(num_keypoints):
                            kpt_score = instance['keypoint_scores'][kpt_idx]
                            kpt_prob = instance['keypoint_probs'][kpt_idx]
                            kpt_thresh = kpt_score if self.score_thresh_type == 'score' else kpt_prob
                            if kpt_thresh > self.keypoint_score_thr:
                                mean_kpt_score += kpt_score
                                valid_num += 1
                        if valid_num != 0:
                            mean_kpt_score /= valid_num
                        instance['score'] = bbox_score * mean_kpt_score
            # perform nms
            if self.nms_mode == 'none':
                valid_kpts[img_id] = instances
            else:
                nms = oks_nms if self.nms_mode == 'oks_nms' else soft_oks_nms
                keep = nms(
                    instances,
                    self.nms_thr,
                    sigmas=self.dataset_meta['sigmas'])
                valid_kpts[img_id] = [instances[_keep] for _keep in keep]

        # convert results to coco style and dump into a json file
        self.results2json(valid_kpts, outfile_prefix=outfile_prefix)

        # only format the results without doing quantitative evaluation
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return {}

        eval_results = OrderedDict()
        
        # mAP evaluation results
        logger.info(f'Evaluating {self.__class__.__name__}...')
        # self.prob_thr = 0.45

        # Classification evaluation results
        try:
            info_str = self._do_python_variables_eval(
                gts if self.coco is None else self.coco.anns,
                valid_kpts
            )
            name_value = OrderedDict(info_str)
            eval_results.update(name_value)
            
            info_str = self._do_python_vector_fields_eval(
                gts if self.coco is None else self.coco.anns,
                valid_kpts
            )

        except Exception:
            print("Error in classification evaluation")
            traceback.print_exc()
            pass

        try:
            self._do_oks_to_iou_eval(
                gts if self.coco is None else self.coco.anns,
                valid_kpts
            )
        except Exception:
            print("Error in oks to iou evaluation")
            traceback.print_exc()
            pass
        
        try:
            self._do_kpts_conf_eval(
                gts if self.coco is None else self.coco.anns,
                valid_kpts
            )
        except Exception:
            print("Error in kpts conf evaluation")
            traceback.print_exc()
            pass

        # Localization evaluation results
        info_str = self._do_python_keypoint_eval(outfile_prefix)
        name_value = OrderedDict(info_str)
        eval_results.update(name_value)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def results2json(self, keypoints: Dict[int, list],
                     outfile_prefix: str) -> str:
        """Dump the keypoint detection results to a COCO style json file.

        Args:
            keypoints (Dict[int, list]): Keypoint detection results
                of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json",

        Returns:
            str: The json file name of keypoint results.
        """
        # the results with category_id
        cat_results = []

        for _, img_kpts in keypoints.items():
            _keypoints = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            num_keypoints = self.dataset_meta['num_keypoints']
            # collect all the person keypoints in current image
            _keypoints = _keypoints.reshape(-1, num_keypoints * 3)

            result = []
            for img_kpt, keypoint in zip(img_kpts, _keypoints):
                res = {
                    'image_id': img_kpt['img_id'],
                    'category_id': img_kpt['category_id'],
                    'keypoints': keypoint.tolist(),
                    'score': float(img_kpt['score']),
                }
                if 'bbox' in img_kpt:
                    res['bbox'] = img_kpt['bbox'].tolist()
                if 'keypoints_visible' in img_kpt:
                    res['visibility'] = img_kpt['keypoints_visible'].tolist()
                result.append(res)

            cat_results.extend(result)

        res_file = f'{outfile_prefix}.keypoints.json'
        dump(cat_results, res_file, sort_keys=True, indent=4)

    def _do_python_keypoint_eval(self, outfile_prefix: str) -> list:
        """Do keypoint evaluation using COCOAPI.

        Args:
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json",

        Returns:
            list: a list of tuples. Each tuple contains the evaluation stats
            name and corresponding stats value.
        """
        res_file = f'{outfile_prefix}.keypoints.json'
        coco_det = self.coco.loadRes(res_file)
        sigmas = self.dataset_meta['sigmas']

        info_str = []
        for extended_oks, match_by_bbox, ignore_border_points in zip(
            self.extended, self.match_by_bbox, self.ignore_border_points
        ):
            prefix = ""
            suffix = ""
            if match_by_bbox:
                prefix = "bbox_" + prefix
            if extended_oks:
                prefix = "Ex_" + prefix
            if ignore_border_points:
                suffix = suffix + "_NoBrd"

            conf_thr = self.prob_thr
            print("+"*80)
            print("COCO Eval params: Bbox {:5s}, ExOKS {:5s}".format(
                str(match_by_bbox), str(extended_oks)
            ), end="")
            if extended_oks:
                print(" with conf_thr: {:.2f} (has probability: {})".format(conf_thr, self.has_probability), end="")
            print()

            coco_eval = COCOeval(
                self.coco,
                coco_det,
                iouType=self.iou_type,
                sigmas=sigmas,
                use_area=self.use_area,
                extended_oks=extended_oks,
                match_by_bbox=match_by_bbox,
                confidence_thr=conf_thr,
                padding=self.padding,
                ignore_near_bbox=ignore_border_points
            )
            coco_eval.params.useSegm = None
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            try:
                stats_names = coco_eval.stats_names
            except AttributeError:
                if self.iou_type == 'keypoints_crowd':
                    stats_names = [
                        'AP', 'AP .5', 'AP .75', 'AR', 'AR .5', 'AR .75',
                        'AP(E)', 'AP(M)', 'AP(H)'
                    ]
                else:
                    stats_names = [
                        'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR',
                        'AR .5', 'AR .75', 'AR (M)', 'AR (L)'
                    ]
            i_str = list(zip(stats_names, coco_eval.stats))
            ignore_stats = self.ignore_stats
            i_str = [(k, v) for k, v in i_str if k not in self.ignore_stats]
            i_str = [(f'{prefix}{k}', v) for k, v in i_str]
            i_str = [(f'{k}{suffix}', v) for k, v in i_str]

            info_str.extend(i_str)

        return info_str

    def _do_python_vector_fields_eval(self, gts, dts):
        # Match GT and DT by 'image_id' and 'id'
        gt_dict = {}
        for _, g in gts.items():
            kpts = np.array(g['keypoints'])
            if np.allclose(kpts, 0):
                continue
            dict_key = (g['image_id'], g['id'])
            gt_dict[dict_key] = g
        dt_dict = {}
        for _, img_d in dts.items():
            for d in img_d:
                dict_key = (d['img_id'], d['id'])
                dt_dict[dict_key] = d

        gt_kpts = []
        gt_bboxes = []
        gt_aboxes = []
        dt_kpts = []
        mask = []
        for key in gt_dict.keys():
            try:
                dtk = np.array(dt_dict[key]['keypoints']).reshape((-1, 3))
            except KeyError:
                continue
            gt_bbox = np.array(gt_dict[key]['bbox']).flatten()
            gt_abox = fix_bbox_aspect_ratio(gt_bbox, aspect_ratio=3/4, padding=1.25, bbox_format='xywh')
            gt_bboxes.append(gt_bbox)
            gt_aboxes.append(gt_abox)
            gtk = np.array(gt_dict[key]['keypoints']).reshape((-1, 3))
            gt_kpts.append(gtk)
            dt_kpts.append(dtk)
            mask.append((gtk[:, 2] > 0).flatten())
        gt_kpts = np.array(gt_kpts).astype(np.float32)
        dt_kpts = np.array(dt_kpts).astype(np.float32)
        mask = np.array(mask)
        gt_bboxes = np.array(gt_bboxes).astype(np.float32)
        gt_aboxes = np.array(gt_aboxes).astype(np.float32)

        out_abox = (
            (gt_kpts[:, :, 0] < gt_aboxes[:, None, 0]) |
            (gt_kpts[:, :, 0] > (gt_aboxes[:, None, 0] + gt_aboxes[:, None, 2])) |
            (gt_kpts[:, :, 1] < gt_aboxes[:, None, 1]) |
            (gt_kpts[:, :, 1] > (gt_aboxes[:, None, 1] + gt_aboxes[:, None, 3]))
        )

        out_bbox = (
            (dt_kpts[:, :, 0] < gt_bboxes[:, None, 0]) |
            (dt_kpts[:, :, 0] > (gt_bboxes[:, None, 0] + gt_bboxes[:, None, 2])) |
            (dt_kpts[:, :, 1] < gt_bboxes[:, None, 1]) |
            (dt_kpts[:, :, 1] > (gt_bboxes[:, None, 1] + gt_bboxes[:, None, 3]))
        )
        
        # Compute vectors
        vec = dt_kpts[:, :, :2] - gt_kpts[:, :, :2]
        # Normalize vectors by bbox size
        vec[:, :, 0] /= gt_bboxes[:, None, 2]
        vec[:, :, 1] /= gt_bboxes[:, None, 3]
        gt_kpts[:, :, 0] -= gt_bboxes[:, None, 0]
        gt_kpts[:, :, 0] /= gt_bboxes[:, None, 2]
        gt_kpts[:, :, 1] -= gt_bboxes[:, None, 1]
        gt_kpts[:, :, 1] /= gt_bboxes[:, None, 3]

        # Select only annotated keypoints
        gt_kpts = gt_kpts[mask, :2]
        vec = vec[mask, :2]


    def _do_python_variables_eval(self, gts, dts):
        """Do evaluation for classification.

        Args:
            gt_anns (dict): The ground truth annotations.
            valid_kpts (dict): The valid keypoint detection results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info('Evaluating classification...')

        # Match GT and DT by 'image_id' and 'id'
        gt_dict = {}
        for _, g in gts.items():
            kpts = np.array(g['keypoints'])
            if np.allclose(kpts, 0):
                continue
            dict_key = (g['image_id'], g['id'])
            gt_dict[dict_key] = g
        dt_dict = {}
        for _, img_d in dts.items():
            for d in img_d:
                dict_key = (d['img_id'], d['id'])
                dt_dict[dict_key] = d
        
        # Get predictions and GT
        gt_vis = []
        dt_scores = []
        dt_vis = []
        dt_probs = []
        gt_names = []
        gt_kpts_list = []
        gt_aboxes = []
        dt_heatmaps = []
        has_heatmaps = True
        fp_count = 0
        fn_count = 0
        wa_count = 0
        wbox_count = 0
        fp_imgs_count = 0
        fn_imgs_count = 0
        wa_imgs_count = 0
        wbox_imgs_count = 0
        num_pts_out = 0
        for key in dt_dict.keys():
            
            gtv = np.array(gt_dict[key]['keypoints'][2::3])
            
            # Check if the padding is enough - recompute the annotation
            gt_kpts = np.array(gt_dict[key]['keypoints']).reshape((-1, 3))
            gt_kpts_list.append(gt_kpts)
            gt_bbox_xywh = np.array(gt_dict[key]['bbox'])
            gt_abox_xywh = fix_bbox_aspect_ratio(gt_bbox_xywh, aspect_ratio=3/4, padding=self.padding, bbox_format='xywh')
            gt_aboxes.append(gt_abox_xywh)
            min_padding = find_min_padding_exact(gt_bbox_xywh, gt_kpts)
            kpts_out = min_padding > self.padding
            gtv[(gtv > 2) & (~kpts_out)] = 1    # If something is annotated as out but isn't, change it to invisible 
            gtv[kpts_out] = 3                    # Everything out is v=3
            num_pts_out += np.sum(kpts_out)
            gt_kpts[:, -1] = gtv.flatten()

            gt_vis.append(gtv)
            image_id = key[0]
            ann_name = "_".join(list(map(str, key)))
            gt_names.append([ann_name]*17)
            dt_vis.append(dt_dict[key]['keypoints_visible'])
            dt_probs.append(dt_dict[key]['keypoint_probs'])
            dt_scores.append(dt_dict[key]['keypoint_scores'])
            kpts = np.array(dt_dict[key]['keypoints']).reshape((-1, 3))

            if EVAL_HEATMAPS:
                dt_heatmap = np.array(dt_dict[key]['heatmaps'])
                has_heatmaps = has_heatmaps and not np.isnan(dt_heatmap).all()
                dt_heatmaps.append(dt_heatmap)

        print("FP count: {:d} kpts, {:d} imgs ({:.2f} %)".format(fp_count, fp_imgs_count, fp_imgs_count / len(dt_dict) * 100))
        print("FN count: {:d} kpts, {:d} imgs ({:.2f} %)".format(fn_count, fn_imgs_count, fn_imgs_count / len(dt_dict) * 100))
        print("WA count: {:d} kpts, {:d} imgs ({:.2f} %)".format(wa_count, wa_imgs_count, wa_imgs_count / len(dt_dict) * 100))
        print("WBOX count: {:d} kpts, {:d} imgs ({:.2f} %)".format(wbox_count, wbox_imgs_count, wbox_imgs_count / len(dt_dict) * 100))
        print("Num points out: ", num_pts_out)

        gt_vis = np.array(gt_vis).flatten().astype(np.float32)
        dt_vis = np.array(dt_vis).flatten().astype(np.float32)
        dt_probs = np.array(dt_probs).flatten().astype(np.float32)
        dt_scores = np.array(dt_scores).flatten().astype(np.float32)
        
        dt_heatmaps = np.array(dt_heatmaps).astype(np.float32)
        gt_kpts = np.array(gt_kpts_list).astype(np.float32)
        gt_aboxes = np.array(gt_aboxes).astype(np.float32)

        if has_heatmaps and EVAL_HEATMAPS:
            self._do_heatmap_calibrations_eval(gt_kpts, dt_heatmaps, gt_aboxes)
        print("\nHas heatmaps: ", has_heatmaps, "\n", "EVAL_HEATMAPS: ", EVAL_HEATMAPS)

        gt_names = np.array(gt_names).squeeze().reshape((-1, 17))
        joint_names = [
            "_nose", "_leye", "_reye", "_lear", "_rear", "_lsho", "_rsho",
            "_lelb", "_relb", "_lwri", "_rwri", "_lhip", "_rhip", "_lkne",
            "_rkne", "_lank", "_rank"
        ]
        joint_names = np.array(joint_names).reshape((1, 17))
        joint_names = np.tile(joint_names, (gt_names.shape[0], 1))
        gt_names = np.char.add(gt_names, joint_names)
        gt_names = gt_names.flatten()
        
        gt_probs = gt_vis.copy()
        gt_probs[gt_probs == 0] = np.nan
        gt_probs[gt_probs == 1] = 1
        gt_probs[gt_probs == 2] = 1
        gt_probs[gt_probs == 3] = 0

        gt_vis[gt_vis == 0] = np.nan
        gt_vis[gt_vis == 1] = 0
        gt_vis[gt_vis == 2] = 1
        gt_vis[gt_vis == 3] = np.nan

        info_str = []
        vis_acc, vis_thr = self._do_classification_eval(gt_vis, dt_vis, force_balance=True, classification_name="Visibility", verbose=True)
        info_str.extend([
            ('vis_acc', float(vis_acc)),
            ('vis_thr', float(vis_thr)),
        ])
        _, _ = self._do_classification_eval(gt_vis, dt_vis, force_balance=True, classification_name="Visibility", verbose=True)

        unique_gt_probs = np.unique(gt_probs[~np.isnan(gt_probs)])
        if len(unique_gt_probs) > 1:
            prob_acc, prob_thr = self._do_classification_eval(gt_probs, dt_probs, force_balance=False, verbose=True, plot_name="prob")
            info_str.extend([
                ('prob_acc', float(prob_acc)),
                ('prob_thr', float(prob_thr)),
            ])
            score_acc, score_thr = self._do_classification_eval(gt_probs, dt_scores, force_balance=False, verbose=True, plot_name="score")
            info_str.extend([
                ('score_acc', float(score_acc)),
                ('score_thr', float(score_thr)),
            ])
            if self.has_probability:
                self.prob_thr = prob_thr
            else:
                self.prob_thr = score_thr
        else:
            print("All GT probs have the same value ({}), skipping probability evaluation".format(unique_gt_probs))
            
        return info_str

    def _do_heatmap_calibrations_eval(self, gt_kpts, dt_heatmaps, gt_aboxes):

        B, C, H, W = dt_heatmaps.shape
        assert gt_kpts.shape[0] == B
        assert gt_kpts.shape[1] == C
        assert gt_aboxes.shape[0] == B

        # Transform gt_kpts to heatmap space
        kpts_visibility = gt_kpts[:, :, 2].flatten()
        for b in range(B):
            abox = gt_aboxes[b, :].astype(np.float32)
            kpts = gt_kpts[b, :, :2].astype(np.float32)
            scale_factor = ((abox[2:] - 1) /
                            (np.array([W, H], dtype=np.float32) - 1)).astype(np.float32)
            transformed_kpts = (kpts - abox[:2]) / scale_factor
            gt_kpts[b, :, :2] = transformed_kpts

        gt_kpts = gt_kpts.reshape((-1, 3)).astype(int)
        dt_heatmaps = dt_heatmaps.reshape((B*C, H, W))
        kpts_mask = (kpts_visibility > 0) & (kpts_visibility < 3)
        
        in_mask = (gt_kpts[:, 0] >= 0) & (gt_kpts[:, 0] <= W-1) & (gt_kpts[:, 1] >= 0) & (gt_kpts[:, 1] <= H-1)
        kpts_mask = kpts_mask & in_mask
        
        gt_kpts = gt_kpts[kpts_mask, :2]
        dt_heatmaps = dt_heatmaps[kpts_mask, :, :]

        assert gt_kpts[:, 0].max() <= W-1
        assert gt_kpts[:, 1].max() <= H-1
        assert gt_kpts[:, 0].min() >= 0
        assert gt_kpts[:, 1].min() >= 0

        bar_width = 0.05
        thresholds = np.linspace(0, 1.0+1e-10, int(1/bar_width)+1, endpoint=True)
        avg_areas = np.zeros_like(thresholds)[:-1]
        binned_ratios = np.zeros_like(thresholds)[:-1]

        for kpt, htm in zip(gt_kpts, dt_heatmaps):
            kpt_int = kpt.astype(int)
            kpt_flat = np.ravel_multi_index(kpt_int[::-1], htm.shape)
            htm_flat = htm.flatten()
            sort_idx = np.argsort(htm_flat, kind='stable')[::-1]
            htm_sorted = htm_flat[sort_idx]
            htm_cumsum = np.cumsum(htm_sorted)
            htm_cumsum = np.clip(htm_cumsum, 0, 1)
            
            # Create the lower and upper bounds for each array value
            lower_bounds = np.hstack(([0], htm_cumsum[:-1]))
            upper_bounds = htm_cumsum

            # Now for each bin defined by thresholds, we calculate how much of each array element falls into that bin
            # We can broadcast the lower and upper bounds against the thresholds
            bin_starts = np.maximum(lower_bounds[:, None], thresholds[:-1])  # Max between lower bounds and bin starts
            bin_ends = np.minimum(upper_bounds[:, None], thresholds[1:])    # Min between upper bounds and bin ends

            # Calculate the overlaps, ensuring no negative values
            weights = np.clip(bin_ends - bin_starts, 0, None)
            
            # Compute number of zero pixels
            zero_pixels = weights.sum(axis=1) < 1e-10

            
            weights[zero_pixels, -1] = 1

            norm_weights = weights / weights.sum(axis=1, keepdims=True)
            avg_areas += norm_weights.sum(axis=0)

            kpt_sort_idx = sort_idx.argsort()[kpt_flat]
            binned_ratios += norm_weights[kpt_sort_idx, :]

        binned_ratios /= gt_kpts.shape[0]
        avg_areas /= (gt_kpts.shape[0] * H * W)

        binned_ratios = binned_ratios[::-1]
        avg_areas = avg_areas[::-1]

        x_labels = (thresholds[1:] + thresholds[:-1]) / 2


    def _do_oks_to_iou_eval(self, gts, dts):

        sigmas = np.array(
            [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        sigmas = (sigmas * 2)**2

        # Format GT such that for each image, you have a set of annotations
        gt_dict = {}
        for _, g in gts.items():
            dict_key = g['image_id']

            if dict_key not in gt_dict:
                gt_dict[dict_key] = []
            gt_dict[dict_key].append(g)

        oks_by_iou = [[], []]
        
        for image_id in gt_dict.keys():
            image_gt = gt_dict[image_id]
            image_dt = dts[image_id]

            if len(image_dt) == 0 or len(image_gt) == 0:
                continue

            dt_bboxes = np.array([d['bbox'] for d in image_dt])
            dt_conf = np.array([d['score'] for d in image_dt])
            gt_bboxes = np.array([g['bbox'] for g in image_gt])

            # Sort DT by confidence; descending
            sorted_idx = np.argsort(dt_conf)[::-1]
            dt_bboxes = dt_bboxes[sorted_idx]

            # IoUs in dimensions DT x GT
            ious = np.array(maskUtils.iou(dt_bboxes, gt_bboxes, np.zeros((len(gt_bboxes)))))
            if ious.size == 1:
                ious = ious.reshape((1, 1))
                dt_argmax_ious = np.array([0])
                dt_max_ious = np.array([0])
                dt_to_gt_matching = np.array([0]) if ious > 0.5 else np.array([-1])
            else:
                # Save the highest IoU for each DT

                value_matrix = ious.copy()
                value_matrix[value_matrix < 0.5] = 0
                dt_to_gt_matching = np.ones(value_matrix.shape[0]) * -1
                for dti in range(value_matrix.shape[0]):
                    max_iou = np.max(value_matrix[dti, :])
                    if max_iou > 0:
                        gti = np.argmax(value_matrix[dti, :])
                        dt_to_gt_matching[dti] = gti
                        value_matrix[:, gti] = 0
                        ious[dti, gti] = -1
                    else:
                        ious[dti, :] = -1
                
                # Each DT is characterized by the second highest IoU with GT (first highest is its own GT)
                dt_max_ious = np.max(ious, axis=1)
            
            # For each DT, compute its OKS with its GT
            for i, dt in enumerate(image_dt):
                gti = int(dt_to_gt_matching[i])
                if gti == -1:
                    continue
                gt = image_gt[gti]
                area = gt['area']

                # Compute OKS
                dkpts = np.array(dt['keypoints']).reshape((-1, 3))
                gkpts = np.array(gt['keypoints']).reshape((-1, 3))
                mkpts = gkpts[:, 2] > 0
                if mkpts.sum() == 0:
                    continue
                dx = dkpts[:, 0] - gkpts[:, 0]
                dy = dkpts[:, 1] - gkpts[:, 1]
                dist = dx**2 + dy**2
                e = dist / area / 2.0 / sigmas
                e = e[mkpts] 

                nan_mask = np.isnan(e)
                if nan_mask.all():
                    oks = 0
                    breakpoint()
                else:
                    oks = np.nanmean(np.exp(-e))

                oks_by_iou[0].append(oks)
                oks_by_iou[1].append(dt_max_ious[i])

    
    def _do_kpts_conf_eval(self, gts, dts):

        sigmas = np.array(
            [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        sigmas = (sigmas * 2)**2

        # Format GT such that for each image, you have a set of annotations
        gt_dict = {}
        for _, g in gts.items():
            dict_key = g['image_id']

            if dict_key not in gt_dict:
                gt_dict[dict_key] = []
            gt_dict[dict_key].append(g)

        conf_eval = []
        
        for image_id in gt_dict.keys():
            image_gt = gt_dict[image_id]
            image_dt = dts[image_id]

            if len(image_dt) == 0 or len(image_gt) == 0:
                continue

            dt_bboxes = np.array([d['bbox'] for d in image_dt])
            dt_conf = np.array([d['score'] for d in image_dt])
            gt_bboxes = np.array([g['bbox'] for g in image_gt])

            # Sort DT by confidence; descending
            sorted_idx = np.argsort(dt_conf)[::-1]
            dt_bboxes = dt_bboxes[sorted_idx]

            # IoUs in dimensions DT x GT
            ious = np.array(maskUtils.iou(dt_bboxes, gt_bboxes, np.zeros((len(gt_bboxes)))))
            if ious.size == 1:
                ious = ious.reshape((1, 1))
                dt_argmax_ious = np.array([0])
                dt_max_ious = np.array([0])
                dt_to_gt_matching = np.array([0]) if ious > 0.5 else np.array([-1])
            else:
                # Save the highest IoU for each DT

                value_matrix = ious.copy()
                value_matrix[value_matrix < 0.5] = 0
                dt_to_gt_matching = np.ones(value_matrix.shape[0]) * -1
                for dti in range(value_matrix.shape[0]):
                    max_iou = np.max(value_matrix[dti, :])
                    if max_iou > 0:
                        gti = np.argmax(value_matrix[dti, :])
                        dt_to_gt_matching[dti] = gti
                        value_matrix[:, gti] = 0
                        ious[dti, gti] = -1
                    else:
                        ious[dti, :] = -1
                
                # Each DT is characterized by the second highest IoU with GT (first highest is its own GT)
                dt_max_ious = np.max(ious, axis=1)
            
            # For each DT, compute its OKS with its GT
            for i, dt in enumerate(image_dt):
                gti = int(dt_to_gt_matching[i])
                if gti == -1:
                    continue
                gt = image_gt[gti]
                area = gt['area']

                # Compute OKS
                dkpts = np.array(dt['keypoints']).reshape((-1, 3))
                dvis = np.array(dt.get('keypoints_visible', np.ones_like(dkpts[:, 2])))
                gkpts = np.array(gt['keypoints']).reshape((-1, 3))

                # conf_eval.append([dkpts[:, 2], gkpts[:, 2]])
                conf_eval.append([dvis, gkpts[:, 2]])

        conf_eval = np.array(conf_eval)
        conf_pts = conf_eval.transpose((0, 2, 1)).reshape((-1, 2))

               
    def _do_classification_eval(self, gts, dts, force_balance=False, verbose=False, names=None, plot_name="", classification_name="Classification"):
        
        mask = ~np.isnan(gts)
        gts = gts[mask].astype(bool)
        dts = dts[mask]
        if names is not None:
            names = names[mask]
        
        # Verify callibration
        if EVAL_CALIBRATION:
            n_bins = 10
            thresholds = np.linspace(0, 1.00, n_bins+1, endpoint=True)
            ratios_by_threshold = np.zeros(n_bins)
            bins = np.digitize(dts, thresholds) -1
            unique_bins = np.unique(bins)
            bins_x_labels = (thresholds[1:] + thresholds[:-1]) / 2
            
            for b in unique_bins:
                selected_gts = gts[bins == b] 
                if len(selected_gts) == 0:
                    ratios_by_threshold[b] = 1.0
                else:
                    ratios_by_threshold[b] = np.sum(selected_gts) / len(selected_gts)

        if force_balance:
            pos_num = np.sum(gts)
            neg_num = np.sum(1 - gts)
            num = min(pos_num, neg_num)
            if num == 0:
                return -1, -1
            pos_idx = np.where(gts == 1)[0]
            neg_idx = np.where(gts == 0)[0]
            np.random.shuffle(pos_idx)
            np.random.shuffle(neg_idx)
            idx = np.concatenate([pos_idx[:num], neg_idx[:num]])
            
            gts = gts[idx]
            dts = dts[idx]
            if names is not None:
                names = names[idx]
        
        n_samples = len(gts)
        thresholds = np.linspace(0, 1.00, 21, endpoint=True)
        preds = (dts[:, None] > thresholds)
        correct = preds == gts[:, None]
        counts = np.sum(correct, axis=0)
        acc = counts / n_samples

        best_idx = np.argmax(acc)
        best_thr = thresholds[best_idx]
        best_acc = acc[best_idx]

        return best_acc, best_thr

    def _sort_and_unique_bboxes(self,
                                kpts: Dict[int, list],
                                key: str = 'id') -> Dict[int, list]:
        """Sort keypoint detection results in each image and remove the
        duplicate ones. Usually performed in multi-batch testing.

        Args:
            kpts (Dict[int, list]): keypoint prediction results. The keys are
                '`img_id`' and the values are list that may contain
                keypoints of multiple persons. Each element in the list is a
                dict containing the ``'key'`` field.
                See the argument ``key`` for details.
            key (str): The key name in each person prediction results. The
                corresponding value will be used for sorting the results.
                Default: ``'id'``.

        Returns:
            Dict[int, list]: The sorted keypoint detection results.
        """
        for img_id, persons in kpts.items():
            # deal with bottomup-style output
            if isinstance(kpts[img_id][0][key], Sequence):
                return kpts
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts

    def error_to_OKS(self, error, area=1.0):
        """Convert the error to OKS."""
        sigmas = np.array(
                [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        
        norm_error = error**2 / sigmas**2 / area / 2.0
        return np.exp(-norm_error)
