# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
from mmengine.fileio import get_local_path

from mmpose.registry import DATASETS
from ..body3d import Human36mDataset


@DATASETS.register_module()
class H36MWholeBodyDataset(Human36mDataset):
    METAINFO: dict = dict(from_file="configs/_base_/datasets/h3wb.py")
    """Human3.6M 3D WholeBody Dataset.

    "H3WB: Human3.6M 3D WholeBody Dataset and Benchmark", ICCV'2023.
    More details can be found in the `paper
    <https://arxiv.org/abs/2211.15692>`__.

    H36M-WholeBody keypoints::

        0-16: 17 body keypoints,
        17-22: 6 foot keypoints,
        23-90: 68 face keypoints,
        91-132: 42 hand keypoints

        In total, we have 133 keypoints for wholebody pose estimation.

    Args:
        ann_file (str): Annotation file path. Default: ''.
        seq_len (int): Number of frames in a sequence. Default: 1.
        seq_step (int): The interval for extracting frames from the video.
            Default: 1.
        multiple_target (int): If larger than 0, merge every
            ``multiple_target`` sequence together. Default: 0.
        multiple_target_step (int): The interval for merging sequence. Only
            valid when ``multiple_target`` is larger than 0. Default: 0.
        pad_video_seq (bool): Whether to pad the video so that poses will be
            predicted for every frame in the video. Default: ``False``.
        causal (bool): If set to ``True``, the rightmost input frame will be
            the target frame. Otherwise, the middle input frame will be the
            target frame. Default: ``True``.
        subset_frac (float): The fraction to reduce dataset size. If set to 1,
            the dataset size is not reduced. Default: 1.
        keypoint_2d_src (str): Specifies 2D keypoint information options, which
            should be one of the following options:

            - ``'gt'``: load from the annotation file
            - ``'detection'``: load from a detection
              result file of 2D keypoint
            - 'pipeline': the information will be generated by the pipeline

            Default: ``'gt'``.
        keypoint_2d_det_file (str, optional): The 2D keypoint detection file.
            If set, 2d keypoint loaded from this file will be used instead of
            ground-truth keypoints. This setting is only when
            ``keypoint_2d_src`` is ``'detection'``. Default: ``None``.
        factor_file (str, optional): The projection factors' file. If set,
            factor loaded from this file will be used instead of calculated
            factors. Default: ``None``.
        camera_param_file (str): Cameras' parameters file. Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data.
            Default: ``dict(img='')``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    def __init__(self, test_mode: bool = False, **kwargs):

        self.camera_order_id = ["54138969", "55011271", "58860488", "60457274"]
        if not test_mode:
            self.subjects = ["S1", "S5", "S6"]
        else:
            self.subjects = ["S7"]

        super().__init__(test_mode=test_mode, **kwargs)

    def _load_ann_file(self, ann_file: str) -> dict:
        with get_local_path(ann_file) as local_path:
            data = np.load(local_path, allow_pickle=True)

        self.ann_data = data["train_data"].item()
        self.camera_data = data["metadata"].item()

    def get_sequence_indices(self) -> List[List[int]]:
        return []

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:

        instance_list = []
        image_list = []

        instance_id = 0
        for subject in self.subjects:
            actions = self.ann_data[subject].keys()
            for act in actions:
                for cam in self.camera_order_id:
                    if cam not in self.ann_data[subject][act]:
                        continue
                    keypoints_2d = self.ann_data[subject][act][cam]["pose_2d"]
                    keypoints_3d = self.ann_data[subject][act][cam]["camera_3d"]
                    num_keypoints = keypoints_2d.shape[1]

                    camera_param = self.camera_data[subject][cam]
                    camera_param = {
                        "K": camera_param["K"][0, :2, ...],
                        "R": camera_param["R"][0],
                        "T": camera_param["T"].reshape(3, 1),
                        "Distortion": camera_param["Distortion"][0],
                    }

                    seq_step = 1
                    _len = (self.seq_len - 1) * seq_step + 1
                    _indices = list(range(len(self.ann_data[subject][act]["frame_id"])))
                    seq_indices = [
                        _indices[i : (i + _len) : seq_step] for i in list(range(0, len(_indices) - _len + 1))
                    ]

                    for idx, frame_ids in enumerate(seq_indices):
                        expected_num_frames = self.seq_len
                        if self.multiple_target:
                            expected_num_frames = self.multiple_target

                        assert len(frame_ids) == (expected_num_frames), (
                            f"Expected `frame_ids` == {expected_num_frames}, but "  # noqa
                            f"got {len(frame_ids)} "
                        )

                        _kpts_2d = keypoints_2d[frame_ids]
                        _kpts_3d = keypoints_3d[frame_ids]

                        target_idx = [-1] if self.causal else [int(self.seq_len) // 2]
                        if self.multiple_target > 0:
                            target_idx = list(range(self.multiple_target))

                        instance_info = {
                            "num_keypoints": num_keypoints,
                            "keypoints": _kpts_2d,
                            "keypoints_3d": _kpts_3d / 1000,
                            "keypoints_visible": np.ones_like(_kpts_2d[..., 0], dtype=np.float32),
                            "keypoints_3d_visible": np.ones_like(_kpts_2d[..., 0], dtype=np.float32),
                            "scale": np.zeros((1, 1), dtype=np.float32),
                            "center": np.zeros((1, 2), dtype=np.float32),
                            "factor": np.zeros((1, 1), dtype=np.float32),
                            "id": instance_id,
                            "category_id": 1,
                            "iscrowd": 0,
                            "camera_param": camera_param,
                            "img_paths": [f"{subject}/{act}/{cam}/{i:06d}.jpg" for i in frame_ids],
                            "img_ids": frame_ids,
                            "lifting_target": _kpts_3d[target_idx] / 1000,
                            "lifting_target_visible": np.ones_like(_kpts_2d[..., 0], dtype=np.float32)[target_idx],
                        }
                        instance_list.append(instance_info)

                        if self.data_mode == "bottomup":
                            for idx, img_name in enumerate(instance_info["img_paths"]):
                                img_info = self.get_img_info(idx, img_name)
                                image_list.append(img_info)

                        instance_id += 1

        return instance_list, image_list
