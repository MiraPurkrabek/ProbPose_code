# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import cv2
import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .utils import (generate_offset_heatmap, generate_udp_gaussian_heatmaps,
                    get_heatmap_maximum, refine_keypoints_dark_udp, generate_probmaps, get_heatmap_expected_value)


@KEYPOINT_CODECS.register_module()
class DoubleProbMap(BaseKeypointCodec):
    r"""Generate two probability maps for keypoint detection using expected OKS scores.
    See the paper: `ProbPose: A Probabilistic Approach to 2D Human Pose Estimation`_ 
    by Purkrabek et al. (2025) for details.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]
        - heatmap size: [W, H]

    Encoded:

        - heatmaps (np.ndarray): The inner probability map in shape (K, H, W)
            where [W, H] is the `heatmap_size`, and K is the number of keypoints.
            Each pixel value represents the expected OKS score if that pixel is
            predicted as the keypoint location, given the ground truth location.
            Generated using standard padding.
        - out_heatmaps (np.ndarray): The outer probability map with same shape
            as heatmaps but generated using larger padding to better handle
            keypoints near and beyond image boundaries.
        - keypoint_weights (np.ndarray): The target weights in shape (K,)

    Args:
        input_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [W, H]
        in_heatmap_padding (float): Padding factor for inner probability map.
            A value of 1.0 means no padding. Defaults to 1.0
        out_heatmap_padding (float): Padding factor for outer probability map.
            Should be larger than in_heatmap_padding. Defaults to 1.0
        heatmap_type (str): The heatmap type to encode the keypoints. Options
            are:

            - ``'gaussian'``: Gaussian heatmap
            - ``'combined'``: Combination of a binary label map and offset
                maps for X and Y axes.

        sigma (float): The sigma value of the Gaussian heatmap when
            ``heatmap_type=='gaussian'``. Defaults to 2.0
        radius_factor (float): The radius factor of the binary label
            map when ``heatmap_type=='combined'``. The positive region is
            defined as the neighbor of the keypoint with the radius
            :math:`r=radius_factor*max(W, H)`. Defaults to 0.0546875
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation in DarkPose. Defaults to 11

    .. _`ProbPose: A Probabilistic Approach to 2D Human Pose Estimation`: 
        https://arxiv.org/abs/2412.02254
    """

    label_mapping_table = dict(keypoint_weights='keypoint_weights', )
    field_mapping_table = dict(heatmaps='heatmaps', )

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 in_heatmap_padding: float = 1.0,
                 out_heatmap_padding: float = 1.0,
                 heatmap_type: str = 'gaussian',
                 sigma: float = 2.,
                 radius_factor: float = 0.0546875,
                 blur_kernel_size: int = 11,
                 ) -> None:
        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.radius_factor = radius_factor
        self.heatmap_type = heatmap_type
        self.blur_kernel_size = blur_kernel_size
        
        self.input_center = np.array(input_size) / 2
        self.input_wh = np.array(input_size)
        self.heatmap_center = np.array(heatmap_size) / 2
        self.heatmap_wh = np.array(heatmap_size)

        self.in_heatmap_padding = in_heatmap_padding
        self.out_heatmap_padding = out_heatmap_padding

        self.in_activation_map_wh = self.input_wh * in_heatmap_padding
        self.out_activation_map_wh = self.input_wh * out_heatmap_padding
        self.in_activation_map_tl = self.input_center - self.in_activation_map_wh / 2
        self.out_activation_map_tl = self.input_center - self.out_activation_map_wh / 2

        self.in_scale_factor = ((self.in_activation_map_wh - 1) /
                                (np.array(heatmap_size)-1)).astype(np.float32)
        self.out_scale_factor = ((self.out_activation_map_wh - 1) /
                                (np.array(heatmap_size)-1)).astype(np.float32)

        if self.heatmap_type not in {'gaussian', 'combined'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `heatmap_type` value'
                f'{self.heatmap_type}. Should be one of '
                '{"gaussian", "combined"}')
        
    def _kpts_to_activation_pts(self, keypoints: np.ndarray, htm_type: str = "in") -> np.ndarray:
        """
        Transform the keypoint coordinates to the activation space.
        In the original UDPHeatmap, activation map is the same as the input image space with
        different resolution but in this case we allow the activation map to have different
        size (padding) than the input image space.
        Centers of activation map and input image space are aligned.
        """
        assert htm_type in ["in", "out"]
        top_left = self.in_activation_map_tl if htm_type == "in" else self.out_activation_map_tl
        scale_factor = self.in_scale_factor if htm_type == "in" else self.out_scale_factor
        transformed_keypoints = keypoints - top_left
        transformed_keypoints = transformed_keypoints / scale_factor
        return transformed_keypoints
    
    def activation_pts_to_kpts(self, keypoints: np.ndarray, htm_type: str = "in") -> np.ndarray:
        """
        Transform the points in activation map to the keypoint coordinates.
        In the original UDPHeatmap, activation map is the same as the input image space with
        different resolution but in this case we allow the activation map to have different
        size (padding) than the input image space.
        Centers of activation map and input image space are aligned.
        """
        assert htm_type in ["in", "out"]
        top_left = self.in_activation_map_tl if htm_type == "in" else self.out_activation_map_tl
        input_size = self.in_activation_map_wh if htm_type == "in" else self.out_activation_map_wh
        W, H = self.heatmap_size
        transformed_keypoints = keypoints / [W - 1, H - 1] * input_size
        transformed_keypoints += top_left
        return transformed_keypoints

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None,
               id_similarity: Optional[float] = 0.0,
               keypoints_visibility: Optional[np.ndarray] = None) -> dict:
        """Encode keypoints into heatmaps. Note that the original keypoint
        coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)
            id_similarity (float): The usefulness of the identity information
                for the whole pose. Defaults to 0.0
            keypoints_visibility (np.ndarray): The visibility bit for each
                keypoint (N, K). Defaults to None

        Returns:
            dict:
            - heatmap (np.ndarray): The generated heatmap in shape
                (C_out, H, W) where [W, H] is the `heatmap_size`, and the
                C_out is the output channel number which depends on the
                `heatmap_type`. If `heatmap_type=='gaussian'`, C_out equals to
                keypoint number K; if `heatmap_type=='combined'`, C_out
                equals to K*3 (x_offset, y_offset and class label)
            - keypoint_weights (np.ndarray): The target weights in shape
                (K,)
        """
        assert keypoints.shape[0] == 1, (
            f'{self.__class__.__name__} only support single-instance '
            'keypoint encoding')
        
        if keypoints_visibility is None:
            keypoints_visibility = np.zeros(keypoints.shape[:2], dtype=np.float32)

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        if self.heatmap_type == 'gaussian':
            heatmaps, keypoint_weights = generate_probmaps(
                heatmap_size=self.heatmap_size,
                keypoints=self._kpts_to_activation_pts(keypoints, htm_type="in"),
                keypoints_visible=keypoints_visible,
                sigma=self.sigma,
                keypoints_visibility=keypoints_visibility,)
            out_heatmaps, out_kpt_weights = generate_probmaps(
                heatmap_size=self.heatmap_size,
                keypoints=self._kpts_to_activation_pts(keypoints, htm_type="out"),
                keypoints_visible=keypoints_visible,
                sigma=self.sigma,
                keypoints_visibility=keypoints_visibility,)
        
        elif self.heatmap_type == 'combined':
            heatmaps, keypoint_weights = generate_offset_heatmap(
                heatmap_size=self.heatmap_size,
                keypoints=self._kpts_to_activation_pts(keypoints, htm_type="in"),
                keypoints_visible=keypoints_visible,
                radius_factor=self.radius_factor)
            out_heatmaps, out_kpt_weights = generate_offset_heatmap(
                heatmap_size=self.heatmap_size,
                keypoints=self._kpts_to_activation_pts(keypoints, htm_type="out"),
                keypoints_visible=keypoints_visible,
                radius_factor=self.radius_factor)
        else:
            raise ValueError(
                f'{self.__class__.__name__} got invalid `heatmap_type` value'
                f'{self.heatmap_type}. Should be one of '
                '{"gaussian", "combined"}')

        annotated = keypoints_visible > 0
        
        out_heatmap_keypoints = self._kpts_to_activation_pts(keypoints, htm_type="out")

        in_image = np.logical_and(
            out_heatmap_keypoints[:, :, 0] >= 0,
            out_heatmap_keypoints[:, :, 0] < self.heatmap_size[0],
        )
        in_image = np.logical_and(
            in_image,
            out_heatmap_keypoints[:, :, 1] >= 0,
        )
        in_image = np.logical_and(
            in_image,
            out_heatmap_keypoints[:, :, 1] < self.heatmap_size[1],
        )

        # Add zero-th dimension to out_heatmaps
        out_htms = np.expand_dims(out_heatmaps, axis=0)        
                
        encoded = dict(
            heatmaps=heatmaps,
            keypoint_weights=keypoint_weights,
            out_heatmaps=out_htms,
            out_kpt_weights=out_kpt_weights,
            annotated=annotated,
            in_image=in_image,
            keypoints_scaled=keypoints,
            identification_similarity=id_similarity,
        )

        return encoded

    def decode(self, encoded: np.ndarray, htm_type: str = "out") -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from heatmaps. The decoded keypoint
        coordinates are in the input image space.

        Args:
            encoded (np.ndarray): Heatmaps in shape (K, H, W)
            htm_type (str): The type of heatmap to decode. Options are:
                    - ``'in'``: Decode the input heatmap
                    - ``'out'``: Decode the output heatmap
                Defaults to 'in'.

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded keypoint coordinates in shape
                (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K). It
                usually represents the confidence of the keypoint prediction
        """
        assert htm_type in ["in", "out"]
        heatmaps = encoded.copy()

        if self.heatmap_type == 'gaussian':
            keypoints, scores = get_heatmap_expected_value(heatmaps)
            # unsqueeze the instance dimension for single-instance results
            keypoints = keypoints[None]
            scores = scores[None]

        elif self.heatmap_type == 'combined':
            _K, H, W = heatmaps.shape
            K = _K // 3

            for cls_heatmap in heatmaps[::3]:
                # Apply Gaussian blur on classification maps
                ks = 2 * self.blur_kernel_size + 1
                cv2.GaussianBlur(cls_heatmap, (ks, ks), 0, cls_heatmap)

            # valid radius
            radius = self.radius_factor * max(W, H)

            x_offset = heatmaps[1::3].flatten() * radius
            y_offset = heatmaps[2::3].flatten() * radius
            keypoints, scores = get_heatmap_maximum(heatmaps=heatmaps[::3])
            index = (keypoints[..., 0] + keypoints[..., 1] * W).flatten()
            index += W * H * np.arange(0, K)
            index = index.astype(int)
            keypoints += np.stack((x_offset[index], y_offset[index]), axis=-1)
            # unsqueeze the instance dimension for single-instance results
            keypoints = keypoints[None].astype(np.float32)
            scores = scores[None]

        keypoints = self.activation_pts_to_kpts(keypoints, htm_type=htm_type)

        return keypoints, scores
