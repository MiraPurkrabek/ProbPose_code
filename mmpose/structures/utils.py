# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List

import cv2
import numpy as np
import torch
from mmengine.structures import InstanceData, PixelData
from mmengine.utils import is_list_of
from pycocotools import mask as Mask

from .bbox.transforms import get_warp_matrix
from .pose_data_sample import PoseDataSample


def merge_data_samples(data_samples: List[PoseDataSample]) -> PoseDataSample:
    """Merge the given data samples into a single data sample.

    This function can be used to merge the top-down predictions with
    bboxes from the same image. The merged data sample will contain all
    instances from the input data samples, and the identical metainfo with
    the first input data sample.

    Args:
        data_samples (List[:obj:`PoseDataSample`]): The data samples to
            merge

    Returns:
        PoseDataSample: The merged data sample.
    """

    if not is_list_of(data_samples, PoseDataSample):
        raise ValueError("Invalid input type, should be a list of " ":obj:`PoseDataSample`")

    if len(data_samples) == 0:
        warnings.warn("Try to merge an empty list of data samples.")
        return PoseDataSample()

    metadata = data_samples[0].metainfo
    metadata["input_center"] = np.array([ds.input_center for ds in data_samples])
    metadata["input_scale"] = np.array([ds.input_scale for ds in data_samples])
    merged = PoseDataSample(metainfo=metadata)

    if "gt_instances" in data_samples[0]:
        merged.gt_instances = InstanceData.cat([d.gt_instances for d in data_samples])

    if "pred_instances" in data_samples[0]:
        merged.pred_instances = InstanceData.cat([d.pred_instances for d in data_samples])

    if "pred_fields" in data_samples[0] and "heatmaps" in data_samples[0].pred_fields:

        # We maintain two sets of heatmaps for different purposes:
        # 1. reverted_heatmaps: Maps heatmaps back to original image space without padding,
        #    useful for precise keypoint localization within the image bounds
        # 2. padded_heatmaps: Maps heatmaps to a padded image space, which is particularly
        #    beneficial for:
        #    - Visualizing heatmap predictions that extend beyond image boundaries
        #    - Better handling of probability maps near image edges
        #    - Improved visualization of keypoint predictions that might be partially
        #      outside the original image frame
        # The padded version helps maintain the full context of the pose estimation,
        # especially for poses where keypoints might be predicted outside the original
        # image boundaries, while the original version ensures accurate keypoint
        # localization within the image.

        # Initialize lists to store both original and padded heatmaps
        reverted_heatmaps = []
        padded_heatmaps = []

        # Compute padding for the whole image
        max_image_pad = [0, 0, 0, 0]
        for data_sample in data_samples:
            aw_scale = data_sample.input_scale
            # Calculate padding to ensure the person is centered in the image
            # Format: [left, top, right, bottom] padding
            img_pad = [
                int(max(aw_scale[0] / 2 - data_sample.input_center[0] + 10, 0)),  # left padding
                int(max(aw_scale[1] / 2 - data_sample.input_center[1] + 10, 0)),  # top padding
                int(
                    max(data_sample.input_center[0] + aw_scale[0] / 2 - data_sample.ori_shape[1] + 10, 0)
                ),  # right padding
                int(
                    max(data_sample.input_center[1] + aw_scale[1] / 2 - data_sample.ori_shape[0] + 10, 0)
                ),  # bottom padding
            ]
            max_image_pad = np.maximum(max_image_pad, img_pad)

        for data_sample in data_samples:
            # Calculate aspect-ratio aware scaling
            aw_scale = data_sample.input_scale

            # Adjust center point based on padding
            aw_center = data_sample.input_center + np.array([max_image_pad[0], max_image_pad[1]])

            # Calculate new image shape after padding
            padded_img_shape = (
                data_sample.ori_shape[0] + max_image_pad[1] + max_image_pad[3],
                data_sample.ori_shape[1] + max_image_pad[0] + max_image_pad[2],
            )

            # Store original and padded transformations for debugging
            data_sample.input_center, aw_center
            data_sample.input_scale, aw_scale
            data_sample.ori_shape, padded_img_shape

            # Generate heatmaps with original parameters
            reverted_heatmaps.append(
                revert_heatmap(
                    data_sample.pred_fields.heatmaps,
                    data_sample.input_center,
                    data_sample.input_scale,
                    data_sample.ori_shape,
                )
            )

            # Generate heatmaps with padded parameters
            padded_heatmaps.append(
                revert_heatmap(data_sample.pred_fields.heatmaps, aw_center, aw_scale, padded_img_shape)
            )

        # Merge heatmaps using maximum values
        merged_heatmaps = np.max(reverted_heatmaps, axis=0)
        merged_padded_heatmaps = np.max(padded_heatmaps, axis=0)

        # Store the padded heatmaps in the prediction fields
        pred_fields = PixelData()
        pred_fields.set_data(dict(heatmaps=merged_padded_heatmaps))
        merged.pred_fields = pred_fields

    if "gt_fields" in data_samples[0] and "heatmaps" in data_samples[0].gt_fields:
        reverted_heatmaps = [
            revert_heatmap(
                data_sample.gt_fields.heatmaps, data_sample.input_center, data_sample.input_scale, data_sample.ori_shape
            )
            for data_sample in data_samples
        ]

        merged_heatmaps = np.max(reverted_heatmaps, axis=0)
        gt_fields = PixelData()
        gt_fields.set_data(dict(heatmaps=merged_heatmaps))
        merged.gt_fields = gt_fields

    return merged


def revert_heatmap(heatmap, input_center, input_scale, img_shape):
    """Revert predicted heatmap on the original image.

    Args:
        heatmap (np.ndarray or torch.tensor): predicted heatmap.
        input_center (np.ndarray): bounding box center coordinate.
        input_scale (np.ndarray): bounding box scale.
        img_shape (tuple or list): size of original image.
    """
    if torch.is_tensor(heatmap):
        heatmap = heatmap.cpu().detach().numpy()

    ndim = heatmap.ndim
    # [K, H, W] -> [H, W, K]
    if ndim == 3:
        heatmap = heatmap.transpose(1, 2, 0)

    hm_h, hm_w = heatmap.shape[:2]
    img_h, img_w = img_shape
    warp_mat = get_warp_matrix(
        input_center.reshape((2,)), input_scale.reshape((2,)), rot=0, output_size=(hm_w, hm_h), inv=True
    )

    heatmap = cv2.warpAffine(heatmap, warp_mat, (img_w, img_h), flags=cv2.INTER_LINEAR)

    # [H, W, K] -> [K, H, W]
    if ndim == 3:
        heatmap = heatmap.transpose(2, 0, 1)

    return heatmap


def split_instances(instances: InstanceData) -> List[InstanceData]:
    """Convert instances into a list where each element is a dict that contains
    information about one instance."""
    results = []

    # return an empty list if there is no instance detected by the model
    if instances is None:
        return results

    for i in range(len(instances.keypoints)):
        result = dict(
            keypoints=instances.keypoints[i].tolist(),
            keypoint_scores=instances.keypoint_scores[i].tolist(),
        )
        if "bboxes" in instances:
            # Flatten bbox coordinates and convert to list format
            result["bbox"] = instances.bboxes[i].flatten().tolist()
            if "bbox_scores" in instances:
                result["bbox_score"] = instances.bbox_scores[i]
        if "masks" in instances:
            # Convert binary mask to COCO polygon format
            mask = instances.masks[i].astype(np.uint8)

            # Find contours in the binary mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Convert contours to COCO polygon format
            segmentation = []
            for contour in contours:
                # Only include contours with sufficient points (>= 3 points, 6 coordinates)
                if contour.size >= 6:
                    segmentation.append(contour.flatten().tolist())
            result["segmentation"] = segmentation
        results.append(result)

    return results
