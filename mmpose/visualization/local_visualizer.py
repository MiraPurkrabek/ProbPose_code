# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.structures import InstanceData, PixelData

from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.registry import VISUALIZERS
from mmpose.structures import PoseDataSample
from .opencv_backend_visualizer import OpencvBackendVisualizer
from .simcc_vis import SimCCVisualizer
from mmpose.structures.keypoint import fix_bbox_aspect_ratio

def _get_adaptive_scales(areas: np.ndarray,
                         min_area: int = 800,
                         max_area: int = 30000) -> np.ndarray:
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``min_area``, the scale is 0.5 while the area is larger than
    ``max_area``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Defaults to 800.
        max_area (int): Upper bound areas for adaptive scales.
            Defaults to 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


@VISUALIZERS.register_module()
class PoseLocalVisualizer(OpencvBackendVisualizer):
    """MMPose Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to ``None``
        vis_backends (list, optional): Visual backend config list. Defaults to
            ``None``
        save_dir (str, optional): Save file dir for all storage backends.
            If it is ``None``, the backend storage will not save any data.
            Defaults to ``None``
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to ``'green'``
        kpt_color (str, tuple(tuple(int)), optional): Color of keypoints.
            The tuple of color should be in BGR order. Defaults to ``'red'``
        link_color (str, tuple(tuple(int)), optional): Color of skeleton.
            The tuple of color should be in BGR order. Defaults to ``None``
        line_width (int, float): The width of lines. Defaults to 1
        radius (int, float): The radius of keypoints. Defaults to 4
        show_keypoint_weight (bool): Whether to adjust the transparency
            of keypoints according to their score. Defaults to ``False``
        alpha (int, float): The transparency of bboxes. Defaults to ``1.0``

    Examples:
        >>> import numpy as np
        >>> from mmengine.structures import InstanceData
        >>> from mmpose.structures import PoseDataSample
        >>> from mmpose.visualization import PoseLocalVisualizer

        >>> pose_local_visualizer = PoseLocalVisualizer(radius=1)
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> gt_instances = InstanceData()
        >>> gt_instances.keypoints = np.array([[[1, 1], [2, 2], [4, 4],
        ...                                          [8, 8]]])
        >>> gt_pose_data_sample = PoseDataSample()
        >>> gt_pose_data_sample.gt_instances = gt_instances
        >>> dataset_meta = {'skeleton_links': [[0, 1], [1, 2], [2, 3]]}
        >>> pose_local_visualizer.set_dataset_meta(dataset_meta)
        >>> pose_local_visualizer.add_datasample('image', image,
        ...                         gt_pose_data_sample)
        >>> pose_local_visualizer.add_datasample(
        ...                       'image', image, gt_pose_data_sample,
        ...                        out_file='out_file.jpg')
        >>> pose_local_visualizer.add_datasample(
        ...                        'image', image, gt_pose_data_sample,
        ...                         show=True)
        >>> pred_instances = InstanceData()
        >>> pred_instances.keypoints = np.array([[[1, 1], [2, 2], [4, 4],
        ...                                       [8, 8]]])
        >>> pred_instances.score = np.array([0.8, 1, 0.9, 1])
        >>> pred_pose_data_sample = PoseDataSample()
        >>> pred_pose_data_sample.pred_instances = pred_instances
        >>> pose_local_visualizer.add_datasample('image', image,
        ...                         gt_pose_data_sample,
        ...                         pred_pose_data_sample)
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = 'green',
                 kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = 'red',
                 link_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (255, 255, 255),
                 skeleton: Optional[Union[List, Tuple]] = None,
                 line_width: Union[int, float] = 1,
                 radius: Union[int, float] = 3,
                 show_keypoint_weight: bool = False,
                 backend: str = 'opencv',
                 alpha: float = 1.0):

        warnings.filterwarnings(
            'ignore',
            message='.*please provide the `save_dir` argument.*',
            category=UserWarning)

        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            backend=backend)

        self.bbox_color = bbox_color
        self.kpt_color = kpt_color
        self.link_color = link_color
        self.line_width = line_width
        self.text_color = text_color
        self.skeleton = skeleton
        self.radius = radius
        self.alpha = alpha
        self.show_keypoint_weight = show_keypoint_weight
        # Set default value. When calling
        # `PoseLocalVisualizer().set_dataset_meta(xxx)`,
        # it will override the default value.
        self.dataset_meta = {}

    def set_dataset_meta(self,
                         dataset_meta: Dict,
                         skeleton_style: str = 'mmpose'):
        """Assign dataset_meta to the visualizer. The default visualization
        settings will be overridden.

        Args:
            dataset_meta (dict): meta information of dataset.
        """
        if skeleton_style == 'openpose':
            dataset_name = dataset_meta['dataset_name']
            if dataset_name == 'coco':
                dataset_meta = parse_pose_metainfo(
                    dict(from_file='configs/_base_/datasets/coco_openpose.py'))
            elif dataset_name == 'coco_wholebody':
                dataset_meta = parse_pose_metainfo(
                    dict(from_file='configs/_base_/datasets/'
                         'coco_wholebody_openpose.py'))
            else:
                raise NotImplementedError(
                    f'openpose style has not been '
                    f'supported for {dataset_name} dataset')

        if isinstance(dataset_meta, dict):
            self.dataset_meta = dataset_meta.copy()
            self.bbox_color = dataset_meta.get('bbox_color', self.bbox_color)
            self.kpt_color = dataset_meta.get('keypoint_colors',
                                              self.kpt_color)
            self.link_color = dataset_meta.get('skeleton_link_colors',
                                               self.link_color)
            self.skeleton = dataset_meta.get('skeleton_links', self.skeleton)
        # sometimes self.dataset_meta is manually set, which might be None.
        # it should be converted to a dict at these times
        if self.dataset_meta is None:
            self.dataset_meta = {}

    def _draw_instances_bbox(self, image: np.ndarray,
                             instances: InstanceData) -> np.ndarray:
        """Draw bounding boxes and corresponding labels of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if 'bboxes' in instances:
            bboxes = instances.bboxes
            self.draw_bboxes(
                bboxes,
                edge_colors=self.bbox_color,
                alpha=self.alpha,
                line_widths=self.line_width)
        else:
            return self.get_image()

        if 'labels' in instances and self.text_color is not None:
            classes = self.dataset_meta.get('classes', None)
            labels = instances.labels

            positions = bboxes[:, :2]
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                bboxes[:, 2] - bboxes[:, 0])
            scales = _get_adaptive_scales(areas)

            for i, (pos, label) in enumerate(zip(positions, labels)):
                label_text = classes[
                    label] if classes is not None else f'class {label}'

                if isinstance(self.bbox_color,
                              tuple) and max(self.bbox_color) > 1:
                    facecolor = [c / 255.0 for c in self.bbox_color]
                else:
                    facecolor = self.bbox_color

                self.draw_texts(
                    label_text,
                    pos,
                    colors=self.text_color,
                    font_sizes=int(13 * scales[i]),
                    vertical_alignments='bottom',
                    bboxes=[{
                        'facecolor': facecolor,
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

        return self.get_image()

    def _draw_instances_kpts(self,
                             image: np.ndarray,
                             instances: InstanceData,
                             kpt_thr: float = 0.3,
                             show_kpt_idx: bool = False,
                             skeleton_style: str = 'mmpose'):
        """Draw keypoints and skeletons (optional) of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``
            skeleton_style (str): Skeleton style selection. Defaults to
                ``'mmpose'``

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        if skeleton_style == 'openpose':
            return self._draw_instances_kpts_openpose(image, instances,
                                                      kpt_thr)

        self.set_image(image)
        img_h, img_w, _ = image.shape

        if 'keypoints' in instances:
            keypoints = instances.get('transformed_keypoints',
                                      instances.keypoints)

            if 'keypoints_visible' in instances:
                keypoints_visible = instances.keypoints_visible
            else:
                keypoints_visible = np.ones(keypoints.shape[:-1])

            for kpts, visible in zip(keypoints, keypoints_visible):
                kpts = np.array(kpts, copy=False)

                if self.kpt_color is None or isinstance(self.kpt_color, str):
                    kpt_color = [self.kpt_color] * len(kpts)
                elif len(self.kpt_color) == len(kpts):
                    kpt_color = self.kpt_color
                else:
                    raise ValueError(
                        f'the length of kpt_color '
                        f'({len(self.kpt_color)}) does not matches '
                        f'that of keypoints ({len(kpts)})')

                # draw links
                if self.skeleton is not None and self.link_color is not None:
                    if self.link_color is None or isinstance(
                            self.link_color, str):
                        link_color = [self.link_color] * len(self.skeleton)
                    elif len(self.link_color) == len(self.skeleton):
                        link_color = self.link_color
                    else:
                        raise ValueError(
                            f'the length of link_color '
                            f'({len(self.link_color)}) does not matches '
                            f'that of skeleton ({len(self.skeleton)})')

                    for sk_id, sk in enumerate(self.skeleton):
                        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                        if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                                or pos1[1] >= img_h or pos2[0] <= 0
                                or pos2[0] >= img_w or pos2[1] <= 0
                                or pos2[1] >= img_h or visible[sk[0]] < kpt_thr
                                or visible[sk[1]] < kpt_thr
                                or link_color[sk_id] is None):
                            # skip the link that should not be drawn
                            continue

                        X = np.array((pos1[0], pos2[0]))
                        Y = np.array((pos1[1], pos2[1]))
                        color = link_color[sk_id]
                        if not isinstance(color, str):
                            color = tuple(int(c) for c in color)
                        transparency = self.alpha
                        if self.show_keypoint_weight:
                            transparency *= max(
                                0,
                                min(1,
                                    0.5 * (visible[sk[0]] + visible[sk[1]])))

                        self.draw_lines(
                            X, Y, color, line_widths=self.line_width)

                # draw each point on image
                for kid, kpt in enumerate(kpts):
                    if visible[kid] < kpt_thr or kpt_color[kid] is None:
                        # skip the point that should not be drawn
                        continue

                    color = kpt_color[kid]
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(0, min(1, visible[kid]))
                    self.draw_circles(
                        kpt,
                        radius=np.array([self.radius]),
                        face_colors=color,
                        edge_colors=color,
                        alpha=transparency,
                        line_widths=self.radius)
                    if show_kpt_idx:
                        kpt_idx_coords = kpt + [self.radius, -self.radius]
                        self.draw_texts(
                            str(kid),
                            kpt_idx_coords,
                            colors=color,
                            font_sizes=self.radius * 3,
                            vertical_alignments='bottom',
                            horizontal_alignments='center')

        return self.get_image()

    def _draw_instances_kpts_openpose(self,
                                      image: np.ndarray,
                                      instances: InstanceData,
                                      kpt_thr: float = 0.3):
        """Draw keypoints and skeletons (optional) of GT or prediction in
        openpose style.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        self.set_image(image)
        img_h, img_w, _ = image.shape

        if 'keypoints' in instances:
            keypoints = instances.get('transformed_keypoints',
                                      instances.keypoints)

            if 'keypoints_visible' in instances:
                keypoints_visible = instances.keypoints_visible
            else:
                keypoints_visible = np.ones(keypoints.shape[:-1])

            keypoints_info = np.concatenate(
                (keypoints, keypoints_visible[..., None]), axis=-1)
            # compute neck joint
            neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
            # neck score when visualizing pred
            neck[:, 2:3] = np.logical_and(
                keypoints_info[:, 5, 2:3] > kpt_thr,
                keypoints_info[:, 6, 2:3] > kpt_thr).astype(int)
            new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)

            mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
            openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
            new_keypoints_info[:, openpose_idx] = \
                new_keypoints_info[:, mmpose_idx]
            keypoints_info = new_keypoints_info

            keypoints, keypoints_visible = keypoints_info[
                ..., :2], keypoints_info[..., 2]

            for kpts, visible in zip(keypoints, keypoints_visible):
                kpts = np.array(kpts, copy=False)

                if self.kpt_color is None or isinstance(self.kpt_color, str):
                    kpt_color = [self.kpt_color] * len(kpts)
                elif len(self.kpt_color) == len(kpts):
                    kpt_color = self.kpt_color
                else:
                    raise ValueError(
                        f'the length of kpt_color '
                        f'({len(self.kpt_color)}) does not matches '
                        f'that of keypoints ({len(kpts)})')

                # draw links
                if self.skeleton is not None and self.link_color is not None:
                    if self.link_color is None or isinstance(
                            self.link_color, str):
                        link_color = [self.link_color] * len(self.skeleton)
                    elif len(self.link_color) == len(self.skeleton):
                        link_color = self.link_color
                    else:
                        raise ValueError(
                            f'the length of link_color '
                            f'({len(self.link_color)}) does not matches '
                            f'that of skeleton ({len(self.skeleton)})')

                    for sk_id, sk in enumerate(self.skeleton):
                        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                        if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                                or pos1[1] >= img_h or pos2[0] <= 0
                                or pos2[0] >= img_w or pos2[1] <= 0
                                or pos2[1] >= img_h or visible[sk[0]] < kpt_thr
                                or visible[sk[1]] < kpt_thr
                                or link_color[sk_id] is None):
                            # skip the link that should not be drawn
                            continue

                        X = np.array((pos1[0], pos2[0]))
                        Y = np.array((pos1[1], pos2[1]))
                        color = link_color[sk_id]
                        if not isinstance(color, str):
                            color = tuple(int(c) for c in color)
                        transparency = self.alpha
                        if self.show_keypoint_weight:
                            transparency *= max(
                                0,
                                min(1,
                                    0.5 * (visible[sk[0]] + visible[sk[1]])))

                        if sk_id <= 16:
                            # body part
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                            transparency = 0.6
                            angle = math.degrees(
                                math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            polygons = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(self.line_width)),
                                int(angle), 0, 360, 1)

                            self.draw_polygons(
                                polygons,
                                edge_colors=color,
                                face_colors=color,
                                alpha=transparency)

                        else:
                            # hand part
                            self.draw_lines(X, Y, color, line_widths=2)

                # draw each point on image
                for kid, kpt in enumerate(kpts):
                    if visible[kid] < kpt_thr or kpt_color[
                            kid] is None or kpt_color[kid].sum() == 0:
                        # skip the point that should not be drawn
                        continue

                    color = kpt_color[kid]
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(0, min(1, visible[kid]))

                    # draw smaller dots for face & hand keypoints
                    radius = self.radius // 2 if kid > 17 else self.radius

                    self.draw_circles(
                        kpt,
                        radius=np.array([radius]),
                        face_colors=color,
                        edge_colors=color,
                        alpha=transparency,
                        line_widths=radius)

        return self.get_image()

    def _draw_instance_heatmap(
        self,
        fields: PixelData,
        overlaid_image: Optional[np.ndarray] = None,
        draw_type = "featmap",
    ):
        """Draw heatmaps of GT or prediction.

        Args:
            fields (:obj:`PixelData`): Data structure for
                pixel-level annotations or predictions.
            overlaid_image (np.ndarray): The image to draw.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        
        assert draw_type in ["featmap", "p_area", "contours"]
        
        if 'heatmaps' not in fields:
            return None
        heatmaps = fields.heatmaps
        if isinstance(heatmaps, np.ndarray):
            heatmaps = torch.from_numpy(heatmaps)
        
        
        if draw_type == "featmap":
            # The original jet-colored featuremap

            if heatmaps.dim() == 3:
                heatmaps, _ = heatmaps.max(dim=0)
            heatmaps = heatmaps.unsqueeze(0)
            out_image = self.draw_featmap(heatmaps, overlaid_image)
        
        elif draw_type == "p_area":
            # Probability area for a given threshold

            colors_rgb = np.array([
                [230, 25, 75],   # red
                [60, 180, 75],   # green
                [255, 225, 25],  # yellow
                [0, 130, 200],   # blue
                [245, 130, 48],  # orange
                [145, 30, 180],  # purple
                [70, 240, 240],  # cyan
                [240, 50, 230],  # magenta
                [210, 245, 60],  # lime
                [250, 190, 212], # pink
                [0, 128, 128],   # teal
                [220, 190, 255], # lavender
                [255, 250, 200], # beige
                [128, 0, 0],     # maroon
                [170, 255, 195], # mint
                [128, 128, 0],   # olive
                [255, 215, 180], # apricot

                [255, 255, 255], # white
                [170, 110, 40],  # brown
                [0, 0, 128],     # navy
                [128, 128, 128], # grey
                [0, 0, 0],       # black
            ])

            # Create an empty image (float for accumulation)
            acc_heatmaps = np.zeros((heatmaps.shape[1], heatmaps.shape[2], 3), dtype=np.float32)

            painted_img = overlaid_image.copy()
            
            # Process each heatmap
            for heatmap, color in zip(heatmaps, colors_rgb):
                heatmap = heatmap.numpy().squeeze().astype(np.float32)
                prob_thr = 0.75

                if heatmap.sum() < prob_thr:
                    continue

                htm_flat = heatmap.flatten()
                htm_sort = np.sort(htm_flat)[::-1]
                htm_cusum = np.cumsum(htm_sort)

                # k_thr = htm_sort[np.searchsorted(htm_cusum, prob_thr)]
                k_thr = htm_sort[np.searchsorted(htm_cusum, prob_thr * htm_cusum[-1])]
                
                binary_map = np.where(heatmap > k_thr, 1, 0)
                binary_map = binary_map.astype(np.uint8)

                contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                probmap_i = cv2.drawContours(overlaid_image.copy(), contours, -1, color.tolist(), thickness=-1, lineType=cv2.LINE_4)
                probmap_i_transparent = cv2.addWeighted(probmap_i, 0.7, painted_img, 0.3, 0)
                painted_img = np.where(binary_map[..., None], probmap_i_transparent, painted_img)

                painted_img = cv2.drawContours(painted_img, contours, -1, color.tolist(), thickness=1, lineType=cv2.LINE_4)

            out_image = painted_img

        elif draw_type == "contours":
            # Probability contours -- each color is 10% area

            overlaid_image = cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2GRAY)
            overlaid_image = cv2.cvtColor(overlaid_image, cv2.COLOR_GRAY2BGR)

            colors_rgb = np.array([
                [230, 25, 75],   # red
                [60, 180, 75],   # green
                [255, 225, 25],  # yellow
                [0, 130, 200],   # blue
                [245, 130, 48],  # orange
                [145, 30, 180],  # purple
                [70, 240, 240],  # cyan
                [240, 50, 230],  # magenta
                [210, 245, 60],  # lime
                [250, 190, 212], # pink
                [0, 128, 128],   # teal
                [220, 190, 255], # lavender
                [255, 250, 200], # beige
                [128, 0, 0],     # maroon
                [170, 255, 195], # mint
                [128, 128, 0],   # olive
                [255, 215, 180], # apricot

                [255, 255, 255], # white
                [170, 110, 40],  # brown
                [0, 0, 128],     # navy
                [128, 128, 128], # grey
                [0, 0, 0],       # black
            ])

            # Create an empty image (float for accumulation)
            
            painted_img = overlaid_image.copy()
            
            # Process each heatmap
            for heatmap, color in zip(heatmaps, colors_rgb):
                heatmap = heatmap.numpy().squeeze().astype(np.float32)
                
                htm_flat = heatmap.flatten()
                htm_sort = np.sort(htm_flat)[::-1]
                htm_cusum = np.cumsum(htm_sort)
                
                for prob_i, prob_thr in enumerate(np.linspace(0.9, 0.1, 9, endpoint=True)):
                                    
                    color = colors_rgb[prob_i]

                    if heatmap.sum() < 0.5:
                        continue

                    if heatmap.sum() < prob_thr:
                        continue

                    k_thr_up   = htm_sort[min(np.searchsorted(htm_cusum, (prob_thr - 0.1), side='right'), len(htm_sort)-1)]
                    k_thr_down = htm_sort[min(np.searchsorted(htm_cusum, prob_thr, side='right'), len(htm_sort)-1)]

                    binary_map = np.where((heatmap >= k_thr_down) & (heatmap < k_thr_up), 1, 0)
                    binary_map = binary_map.astype(np.uint8)

                    if (binary_map.sum() / binary_map.size) > 0.005:
                        continue

                    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    probmap_i = cv2.drawContours(overlaid_image.copy(), contours, -1, color.tolist(), thickness=-1, lineType=cv2.LINE_4)
                    probmap_i_transparent = cv2.addWeighted(probmap_i, 0.6, painted_img, 0.4, 0)
                    painted_img = np.where(binary_map[..., None], probmap_i_transparent, painted_img)
                
            out_image = painted_img

        return out_image

    def _draw_instance_xy_heatmap(
        self,
        fields: PixelData,
        overlaid_image: Optional[np.ndarray] = None,
        n: int = 20,
    ):
        """Draw heatmaps of GT or prediction.

        Args:
            fields (:obj:`PixelData`): Data structure for
            pixel-level annotations or predictions.
            overlaid_image (np.ndarray): The image to draw.
            n (int): Number of keypoint, up to 20.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        if 'heatmaps' not in fields:
            return None
        heatmaps = fields.heatmaps
        _, h, w = heatmaps.shape
        if isinstance(heatmaps, np.ndarray):
            heatmaps = torch.from_numpy(heatmaps)
        out_image = SimCCVisualizer().draw_instance_xy_heatmap(
            heatmaps, overlaid_image, n)
        out_image = cv2.resize(out_image[:, :, ::-1], (w, h))
        return out_image

    @master_only
    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       data_sample: PoseDataSample,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       draw_heatmap: bool = False,
                       draw_bbox: bool = False,
                       show_kpt_idx: bool = False,
                       skeleton_style: str = 'mmpose',
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       kpt_thr: float = 0.3,
                       step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier
            image (np.ndarray): The image to draw
            data_sample (:obj:`PoseDataSample`, optional): The data sample
                to visualize
            draw_gt (bool): Whether to draw GT PoseDataSample. Default to
                ``True``
            draw_pred (bool): Whether to draw Prediction PoseDataSample.
                Defaults to ``True``
            draw_bbox (bool): Whether to draw bounding boxes. Default to
                ``False``
            draw_heatmap (bool): Whether to draw heatmaps. Defaults to
                ``False``
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``
            skeleton_style (str): Skeleton style selection. Defaults to
                ``'mmpose'``
            show (bool): Whether to display the drawn image. Default to
                ``False``
            wait_time (float): The interval of show (s). Defaults to 0
            out_file (str): Path to output file. Defaults to ``None``
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            step (int): Global step value to record. Defaults to 0
        """

        gt_img_data = None
        pred_img_data = None

        dt_kpts = data_sample.pred_instances.keypoints.reshape((-1, 2))
        dt_vis = data_sample.pred_instances.keypoint_scores.reshape(-1)[:17]
        try:
            gt_kpts = data_sample.gt_instances.keypoints.reshape((-1, 2))
            gt_vis = data_sample.gt_instances.keypoints_visible.reshape(-1)[:17] * 2
        except AttributeError:
            gt_kpts = np.zeros_like(dt_kpts)
            gt_vis = np.zeros_like(dt_vis)

        gt_kpts = gt_kpts[:17, :]
        dt_kpts = dt_kpts[:17, :]

        if draw_gt:
            gt_img_data = image.copy()
            gt_img_heatmap = None

            # draw bboxes & keypoints
            if 'gt_instances' in data_sample:
                gt_img_data = self._draw_instances_kpts(
                    gt_img_data, data_sample.gt_instances, kpt_thr,
                    show_kpt_idx, skeleton_style)
                if draw_bbox:
                    gt_img_data = self._draw_instances_bbox(
                        gt_img_data, data_sample.gt_instances)
                
            # draw heatmaps
            if 'gt_fields' in data_sample and draw_heatmap:
                gt_img_heatmap = self._draw_instance_heatmap(
                    data_sample.gt_fields, image)
                
                # Draw abox over heatmap
                bbox_xyxy = data_sample.gt_instances.bboxes.squeeze()
                abox_xyxy = fix_bbox_aspect_ratio(bbox_xyxy, aspect_ratio=3/4, padding=1.25, bbox_format='xyxy')
                abox_xyxy = abox_xyxy.flatten().astype(int)
                gt_img_heatmap = cv2.rectangle(gt_img_heatmap, (abox_xyxy[0], abox_xyxy[1]), (abox_xyxy[2], abox_xyxy[3]), (0, 255, 0), 2)

                if gt_img_heatmap is not None:
                    gt_img_data = np.concatenate((gt_img_data, gt_img_heatmap),
                                                 axis=0)

        if draw_pred:
            pred_img_data = image.copy()
            pred_img_heatmap = None

            # draw bboxes & keypoints
            if 'pred_instances' in data_sample:
                pred_img_data = self._draw_instances_kpts(
                    pred_img_data, data_sample.pred_instances, kpt_thr,
                    show_kpt_idx, skeleton_style)
                if draw_bbox:
                    pred_img_data = self._draw_instances_bbox(
                        pred_img_data, data_sample.pred_instances)

            # draw heatmaps
            if 'pred_fields' in data_sample and draw_heatmap:
                if 'keypoint_x_labels' in data_sample.pred_instances:
                    pred_img_heatmap = self._draw_instance_xy_heatmap(
                        data_sample.pred_fields, image)
                else:

                    max_image_pad = [0, 0, 0, 0]
                    for input_center, aw_scale in zip(data_sample.input_center, data_sample.input_scale):
                        img_pad = [
                            int(max(aw_scale[0] / 2 - input_center[0] + 10, 0)),
                            int(max(aw_scale[1] / 2 - input_center[1] + 10, 0)),
                            int(max(input_center[0] + aw_scale[0] / 2 - data_sample.ori_shape[1] + 10, 0)),
                            int(max(input_center[1] + aw_scale[1] / 2 - data_sample.ori_shape[0] + 10, 0))
                        ]
                        max_image_pad = np.maximum(max_image_pad, img_pad)

                    padded_img = cv2.copyMakeBorder(
                        image.copy(),
                        max_image_pad[1],
                        max_image_pad[3],
                        max_image_pad[0],
                        max_image_pad[2],
                        cv2.BORDER_CONSTANT,
                        value=(80, 80, 80),
                    ) 
                    
                    # Normalize heatmaps and compute its posterior
                    # ProbMaps are not normalized due to the resize to original image size
                    # Heatmaps are never normalized by design
                    heatmaps = data_sample.pred_fields.heatmaps
                    heatmaps = heatmaps / heatmaps.sum(axis=(1,2), keepdims=True)
                    presence_prob = data_sample.pred_instances.keypoints_probs.reshape(-1, 17)

                    # Take mean across all instances. Even though the operation does not make sense mathematically, it is only for visualization
                    # To make the probmaps mathematically corect, the multiplication should be made in mmpose/structures/utils.py - merge_data_samples()
                    # But that would not be backward compatible with previous methods
                    presence_prob = presence_prob.mean(axis=0)
                    posterior = heatmaps * presence_prob[:, None, None]
                    data_sample.pred_fields.heatmaps = posterior
                    
                    pred_img_heatmap = self._draw_instance_heatmap(
                        data_sample.pred_fields, padded_img, draw_type="p_area")
                    
                    # Draw abox over heatmap
                    if draw_bbox:
                        for bbox_xyxy in data_sample.gt_instances.bboxes:   
                            bbox_xyxy[:2] += np.array(max_image_pad[:2])
                            bbox_xyxy[2:] += np.array(max_image_pad[:2])
                            abox_xyxy = fix_bbox_aspect_ratio(bbox_xyxy, aspect_ratio=3/4, padding=1.25, bbox_format='xyxy')
                            abox_xyxy = abox_xyxy.flatten().astype(int)
                            pred_img_heatmap = cv2.rectangle(pred_img_heatmap, (abox_xyxy[0], abox_xyxy[1]), (abox_xyxy[2], abox_xyxy[3]), (0, 255, 0), 1)

                if pred_img_heatmap is not None:
                    pred_img_heatmap = cv2.resize(pred_img_heatmap, (pred_img_data.shape[:2][::-1]))
                    pred_img_data = np.concatenate(
                        (pred_img_data, pred_img_heatmap), axis=0)

        # merge visualization results
        if gt_img_data is not None and pred_img_data is not None:
            if gt_img_heatmap is None and pred_img_heatmap is not None:
                gt_img_data = np.concatenate((gt_img_data, image), axis=0)
            elif gt_img_heatmap is not None and pred_img_heatmap is None:
                pred_img_data = np.concatenate((pred_img_data, image), axis=0)

            # Resize GT img to the same height as pred img while keeping the aspect ratio
            new_height = pred_img_data.shape[0]
            new_width = int(gt_img_data.shape[1] * new_height / gt_img_data.shape[0])
            gt_img_data = cv2.resize(
                gt_img_data,
                (new_width, new_height),
            )
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)

        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            # save drawn_img to backends
            self.add_image(name, drawn_img, step)

        return self.get_image()
