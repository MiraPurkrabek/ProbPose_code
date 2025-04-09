# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import torch
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.evaluation.functional import pose_pck_accuracy, keypoint_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions, InstanceList, InstanceData)
from ..base_head import BaseHead

import numpy as np
import json

OptIntSeq = Optional[Sequence[int]]

KEYPOINT_NAMES = [
    'nose', 'leye', 'reye', 'lear', 'rear', 'lsho', 'rsho', 'lelb', 'relb',
    'lwri', 'rwri', 'lhip', 'rhip', 'lkne', 'rkne', 'lank', 'rank'
]

@MODELS.register_module()
class DoubleProbMapHead(BaseHead):
    """Multi-variate head predicting all information about keypoints. Apart 
    from the heatmap, it also predicts:
        1) Heatmap for each keypoint
        2) Probability of keypoint being in the heatmap
        3) Visibility of each keypoint
        4) Predicted OKS per keypoint
        5) Predictd euclidean error per keypoint
    The heatmap predicting part is the same as HeatmapHead introduced in
    in `Simple Baselines`_ by Xiao et al (2018).

    Args:
        in_channels (int | Sequence[int]): Number of channels in the input
            feature map
        out_channels (int): Number of channels in the output heatmap
        deconv_out_channels (Sequence[int], optional): The output channel
            number of each deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``
        conv_out_channels (Sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        final_layer_dict (dict): Arguments of the final Conv2d layer.
            Defaults to ``dict(kernel_size=1)``
        keypoint_loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KeypointMSELoss`
        probability_loss (Config): Config of the probability loss. Defaults to use
            :class:`BCELoss`
        visibility_loss (Config): Config of the visibility loss. Defaults to use
            :class:`BCELoss`
        oks_loss (Config): Config of the oks loss. Defaults to use
            :class:`MSELoss`
        error_loss (Config): Config of the error loss. Defaults to use
            :class:`L1LogLoss`
        normalize (bool): Whether to normalize values in the heatmaps between 
            0 and 1 with sigmoid. Defaults to ``False``
        detach_probability (bool): Whether to detach the probability
            from gradient computation. Defaults to ``True``
        detach_visibility (bool): Whether to detach the visibility
            from gradient computation. Defaults to ``True``
        learn_heatmaps_from_zeros (bool): Whether to learn the
            heatmaps from zeros. Defaults to ``False``
        freeze_heatmaps (bool): Whether to freeze the heatmaps prediction.
            Defaults to ``False``
        freeze_probability (bool): Whether to freeze the probability prediction.
            Defaults to ``False``
        freeze_visibility (bool): Whether to freeze the visibility prediction.
            Defaults to ``False``
        freeze_oks (bool): Whether to freeze the oks prediction.
            Defaults to ``False``
        freeze_error (bool): Whether to freeze the error prediction.
            Defaults to ``False``
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings


    .. _`Simple Baselines`: https://arxiv.org/abs/1804.06208
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 final_layer_dict: dict = dict(kernel_size=1),
                 keypoint_loss: ConfigType = dict(
                     type='KeypointMSELoss', use_target_weight=True),
                 probability_loss: ConfigType = dict(
                     type='BCELoss', use_target_weight=True),
                 visibility_loss: ConfigType = dict(
                     type='BCELoss', use_target_weight=True),
                 oks_loss: ConfigType = dict(
                     type='MSELoss', use_target_weight=True),
                 error_loss: ConfigType = dict(
                     type='L1LogLoss', use_target_weight=True),
                 normalize: bool = False,
                 detach_probability: bool = True,
                 detach_visibility: bool = True,
                 detach_second_heatmaps: bool = False,
                 learn_heatmaps_from_zeros: bool = False,
                 split_heatmaps_by: str = 'in/all',
                 freeze_heatmaps: bool = False,
                 freeze_second_heatmaps: bool = False,
                 freeze_probability: bool = False,
                 freeze_visibility: bool = False,
                 freeze_oks: bool = False,
                 freeze_error: bool = False,
                 decoder: OptConfigType = dict(
                    type='DoubleProbMap', input_size=(192, 256),
                    heatmap_size=(48, 64),  sigma=2, in_heatmap_padding=1.0, out_heatmap_padding=1.25),
                 init_cfg: OptConfigType = None,
        ):

        split_heatmaps_by = split_heatmaps_by.lower()
        assert split_heatmaps_by in [
            "visibility",
            "in/out",
            "in/all"            
        ], "'split_heatmaps_by' must be one of 'visibility', 'in/out', 'in/all'"
        self.split_heatmaps_by = split_heatmaps_by

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.tmp_count=0
        self.tmp_sum=0
        self.in_out_stats = {
            "dataset_name": 0,
            "p(in)": 0,
            "p(out)": 0,

            "p_in(in|in)": 0,
            "p_in(in|out)": 0,
            "p_in(out|in)": 0,
            "p_in(out|out)": 0,
            
            "p_out(in|in)": 0,
            "p_out(in|out)": 0,
            "p_out(out|in)": 0,
            "p_out(out|out)": 0,
            
            "dist_in(in|in)": 0,
            "dist_in(out|in)": 0,
            "dist_in(out|out)": 0,
            "dist_in(in|out)": 0,
            "dist_out(in|in)": 0,
            "dist_out(out|in)": 0,
            "dist_out(out|out)": 0,
            "dist_out(in|out)": 0,
        }
        self.results_log = {}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.keypoint_loss_module = MODELS.build(keypoint_loss)
        self.probability_loss_module = MODELS.build(probability_loss)
        self.visibility_loss_module = MODELS.build(visibility_loss)
        self.oks_loss_module = MODELS.build(oks_loss)
        self.error_loss_module = MODELS.build(error_loss)

        self.decoder = KEYPOINT_CODECS.build(decoder)
        self.nonlinearity = nn.ReLU(inplace=True)
        self.learn_heatmaps_from_zeros = learn_heatmaps_from_zeros
        
        self.detach_all = freeze_heatmaps and detach_second_heatmaps and detach_probability and detach_visibility
        self._build_first_heatmap_head(
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            final_layer_dict=final_layer_dict,
            normalize=normalize,
            freeze=freeze_heatmaps)
        self.detach_second_heatmaps = detach_second_heatmaps
        self._build_second_heatmap_head(
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            final_layer_dict=final_layer_dict,
            normalize=normalize,
            freeze=freeze_second_heatmaps)

        self.detach_probability = detach_probability
        self._build_probability_head(
            in_channels=in_channels,
            out_channels=out_channels,
            freeze=freeze_probability)
        
        self.detach_visibility = detach_visibility
        self._build_visibility_head(
            in_channels=in_channels,
            out_channels=out_channels,
            freeze=freeze_visibility)
        
        self._build_oks_head(
            in_channels=in_channels,
            out_channels=out_channels,
            freeze=freeze_oks)
        self.freeze_oks = freeze_oks

        self._build_error_head(
            in_channels=in_channels,
            out_channels=out_channels,
            freeze=freeze_error)
        self.freeze_error = freeze_error

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _build_first_heatmap_head(self, in_channels: int, out_channels: int,
                            deconv_out_channels: Sequence[int],
                            deconv_kernel_sizes: Sequence[int],
                            conv_out_channels: Sequence[int],
                            conv_kernel_sizes: Sequence[int],
                            final_layer_dict: dict,
                            normalize: bool = False,
                            freeze: bool = False) -> None:
        """Build the original heatmap head module."""
        deconv, conv, final, normalize = self._build_heatmap_head(
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            final_layer_dict=final_layer_dict,
            normalize=normalize,
            freeze=freeze)
        self.deconv_layers = deconv
        self.conv_layers = conv
        self.final_layer = final
        self.normalize_layer = normalize

    def _build_second_heatmap_head(self, in_channels: int, out_channels: int,
                            deconv_out_channels: Sequence[int],
                            deconv_kernel_sizes: Sequence[int],
                            conv_out_channels: Sequence[int],
                            conv_kernel_sizes: Sequence[int],
                            final_layer_dict: dict,
                            normalize: bool = False,
                            freeze: bool = False) -> None:
        """Build the original heatmap head module."""
        deconv, conv, final, normalize = self._build_heatmap_head(
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            final_layer_dict=final_layer_dict,
            normalize=normalize,
            freeze=freeze)
        self.second_head = nn.Sequential(deconv, conv, final, normalize)

    def _build_heatmap_head(self, in_channels: int, out_channels: int,
                            deconv_out_channels: Sequence[int],
                            deconv_kernel_sizes: Sequence[int],
                            conv_out_channels: Sequence[int],
                            conv_kernel_sizes: Sequence[int],
                            final_layer_dict: dict,
                            normalize: bool = False,
                            freeze: bool = False) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
        """Build the heatmap head module."""
        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            deconv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                    conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {conv_out_channels} and '
                    f'{conv_kernel_sizes}')

            conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            conv_layers = nn.Identity()

        if final_layer_dict is not None:
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)
            cfg.update(final_layer_dict)
            final_layer = build_conv_layer(cfg)
        else:
            final_layer = nn.Identity()
        normalize_layer = nn.Sigmoid() if normalize else nn.Identity()

        if freeze:
            for param in deconv_layers.parameters():
                param.requires_grad = False
            for param in conv_layers.parameters():
                param.requires_grad = False
            for param in final_layer.parameters():
                param.requires_grad = False
            for param in normalize_layer.parameters():
                param.requires_grad = False
        return deconv_layers, conv_layers, final_layer, normalize_layer

    def _build_probability_head(self, in_channels: int, out_channels: int,
                                freeze: bool = False) -> nn.Module:
        """Build the probability head module."""
        ppb_layers = []
        kernel_sizes = [(4, 3), (2, 2), (2, 2)]
        for i in range(len(kernel_sizes)):
            ppb_layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            ppb_layers.append(
                nn.BatchNorm2d(num_features=in_channels))
            ppb_layers.append(
                nn.MaxPool2d(kernel_size=kernel_sizes[i], stride=kernel_sizes[i], padding=0))
            ppb_layers.append(self.nonlinearity)
        ppb_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        ppb_layers.append(nn.Sigmoid())
        self.probability_layers = nn.Sequential(*ppb_layers)

        if freeze:
            for param in self.probability_layers.parameters():
                param.requires_grad = False

    def _build_visibility_head(self, in_channels: int, out_channels: int,
                                 freeze: bool = False) -> nn.Module:
        """Build the visibility head module."""
        vis_layers = []
        kernel_sizes = [(4, 3), (2, 2), (2, 2)]
        for i in range(len(kernel_sizes)):
            vis_layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            vis_layers.append(
                nn.BatchNorm2d(num_features=in_channels))
            vis_layers.append(
                nn.MaxPool2d(kernel_size=kernel_sizes[i], stride=kernel_sizes[i], padding=0))
            vis_layers.append(self.nonlinearity)
        vis_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        vis_layers.append(nn.Sigmoid())
        self.visibility_layers = nn.Sequential(*vis_layers)

        if freeze:
            for param in self.visibility_layers.parameters():
                param.requires_grad = False

    def _build_oks_head(self, in_channels: int, out_channels: int,
                        freeze: bool = False) -> nn.Module:
        """Build the oks head module."""
        oks_layers = []
        kernel_sizes = [(4, 3), (2, 2), (2, 2)]
        for i in range(len(kernel_sizes)):
            oks_layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            oks_layers.append(
                nn.BatchNorm2d(num_features=in_channels))
            oks_layers.append(
                nn.MaxPool2d(kernel_size=kernel_sizes[i], stride=kernel_sizes[i], padding=0))
            oks_layers.append(self.nonlinearity)
        oks_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        oks_layers.append(nn.Sigmoid())
        self.oks_layers = nn.Sequential(*oks_layers)

        if freeze:
            for param in self.oks_layers.parameters():
                param.requires_grad = False

    def _build_error_head(self, in_channels: int, out_channels: int,
                        freeze: bool = False) -> nn.Module:
        """Build the error head module."""
        error_layers = []
        kernel_sizes = [(4, 3), (2, 2), (2, 2)]
        for i in range(len(kernel_sizes)):
            error_layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            error_layers.append(
                nn.BatchNorm2d(num_features=in_channels))
            error_layers.append(
                nn.MaxPool2d(kernel_size=kernel_sizes[i], stride=kernel_sizes[i], padding=0))
            error_layers.append(self.nonlinearity)
        error_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        error_layers.append(self.nonlinearity)
        self.error_layers = nn.Sequential(*error_layers)

        if freeze:
            for param in self.error_layers.parameters():
                param.requires_grad = False

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
            layers.append(build_conv_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(self.nonlinearity)
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(build_upsample_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(self.nonlinearity)
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _error_from_heatmaps(self, gt_heatmaps: Tensor, dt_heatmaps: Tensor) -> Tensor:
        """Calculate the error from heatmaps.

        Args:
            heatmaps (Tensor): The predicted heatmaps.

        Returns:
            Tensor: The predicted error.
        """
        # Transform to numpy
        gt_heatmaps = to_numpy(gt_heatmaps)
        dt_heatmaps = to_numpy(dt_heatmaps)

        # Get locations from heatmaps
        B, C, H, W = gt_heatmaps.shape
        gt_coords = np.zeros((B, C, 2))
        dt_coords = np.zeros((B, C, 2))
        for i, (gt_htm, dt_htm) in enumerate(zip(gt_heatmaps, dt_heatmaps)):
            coords, score = self.decoder.decode(gt_htm)
            coords = coords.squeeze()
            gt_coords[i, :, :] = coords
            
            coords, score = self.decoder.decode(dt_htm)
            coords = coords.squeeze()
            dt_coords[i, :, :] = coords
        
        # NaN coordinates mean empty heatmaps -> set them to -1
        # as the error will be ignored by weight
        gt_coords[np.isnan(gt_coords)] = -1

        # Calculate the error
        target_errors = np.linalg.norm(gt_coords - dt_coords, axis=2)
        assert (target_errors >= 0).all(), "Euclidean distance cannot be negative"

        return target_errors
    
    def _oks_from_heatmaps(self, gt_heatmaps: Tensor, dt_heatmaps: Tensor, weight: Tensor) -> Tensor:
        """Calculate the OKS from heatmaps.

        Args:
            heatmaps (Tensor): The predicted heatmaps.

        Returns:
            Tensor: The predicted OKS.
        """
        C = dt_heatmaps.shape[1]

        # Transform to numpy
        gt_heatmaps = to_numpy(gt_heatmaps)
        dt_heatmaps = to_numpy(dt_heatmaps)
        B, C, H, W = gt_heatmaps.shape
        weight = to_numpy(weight).squeeze().reshape((B, C, 1))

        # Get locations from heatmaps
        gt_coords = np.zeros((B, C, 2))
        dt_coords = np.zeros((B, C, 2))
        for i, (gt_htm, dt_htm) in enumerate(zip(gt_heatmaps, dt_heatmaps)):
            coords, score = self.decoder.decode(gt_htm)
            coords = coords.squeeze()
            gt_coords[i, :, :] = coords
            
            coords, score = self.decoder.decode(dt_htm)
            coords = coords.squeeze()
            dt_coords[i, :, :] = coords

        # NaN coordinates mean empty heatmaps -> set them to 0
        gt_coords[np.isnan(gt_coords)] = 0

        # Add probability as visibility
        gt_coords = gt_coords * weight
        dt_coords = dt_coords * weight
        gt_coords = np.concatenate((gt_coords, weight*2), axis=2)
        dt_coords = np.concatenate((dt_coords, weight*2), axis=2)

        # Calculate the oks
        target_oks = []
        oks_weights = []
        for i in range(len(gt_coords)):
            gt_kpts = gt_coords[i]
            dt_kpts = dt_coords[i]
            valid_gt_kpts = gt_kpts[:, 2] > 0
            if not valid_gt_kpts.any():
                # Changed for per-keypoint OKS
                target_oks.append(np.zeros(C))
                oks_weights.append(0)
                continue

            gt_bbox = np.array([
                0, 0,
                64, 48,
            ])
            gt = {
                'keypoints': gt_kpts,
                'bbox': gt_bbox,
                'area': gt_bbox[2] * gt_bbox[3],
            }
            dt = {
                'keypoints': dt_kpts,
                'bbox': gt_bbox,
                'area': gt_bbox[2] * gt_bbox[3],
            }
            # Changed for per-keypoint OKS
            oks = compute_oks(gt, dt, use_area=False, per_kpt=True)
            target_oks.append(oks)
            oks_weights.append(1)

        target_oks = np.array(target_oks)

        oks_weights = np.array(oks_weights)

        return target_oks, oks_weights

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def merge_heatmaps(self, heatmaps1: Tensor, heatmaps2: Tensor, bbox_masks: Tensor, return_stats: bool = False) -> Tensor:
        """Merge the heatmaps.

        Args:
            heatmaps1 (Tensor): The first heatmap.
            heatmaps2 (Tensor): The second heatmap.

        Returns:
            Tensor: The merged heatmaps.
        """
        B, C, H, W = heatmaps1.shape
        hin_in = torch.ones((B*C), device=heatmaps1.device)
        hout_in = torch.zeros((B*C), device=heatmaps1.device)

        # If decoder is DoubleProbMap, we must reshape and pad heatmaps to the input size
        if self.decoder.__class__.__name__ == "DoubleProbMap":
            htm1_pad = self.decoder.in_heatmap_padding
            htm2_pad = self.decoder.out_heatmap_padding
            in_wh = self.decoder.in_activation_map_wh
            out_wh = self.decoder.out_activation_map_wh
            input_size = np.array(self.decoder.input_size)
            max_pad = max(htm1_pad, htm2_pad)
            
            if max_pad < 1.0:
                raise NotImplementedError("Max padding < 1.0 not implemented yet")
            
            # Pad the smaller heatmap such that it cover the same area as the larger one
            if htm1_pad < htm2_pad:
                pad = ((htm2_pad / htm1_pad) * in_wh).astype(int)
                heatmaps1 = nn.functional.pad(heatmaps1, (pad[0], pad[0], pad[1], pad[1]), mode='constant', value=0)
                heatmaps1 = nn.functional.interpolate(heatmaps1, size=(H, W), mode='bilinear')
                
                # Resize bbox_max accordingly
                bbox_pad = ((htm2_pad - 1) * input_size).astype(int)
                bbox_masks = nn.functional.pad(bbox_masks, (bbox_pad[0], bbox_pad[0], bbox_pad[1], bbox_pad[1]), mode='constant', value=0)
                bbox_masks = nn.functional.interpolate(bbox_masks, size=(H, W), mode='nearest')
            else:
                pad = ((htm1_pad / htm2_pad) * out_wh).astype(int)
                heatmaps2 = nn.functional.pad(heatmaps2, (pad[0], pad[0], pad[1], pad[1]), mode='constant', value=0)
                heatmaps2 = nn.functional.interpolate(heatmaps2, size=(H, W), mode='bilinear')
                
                # Resize bbox_max accordingly
                bbox_pad = ((htm1_pad - 1) * input_size).astype(int)
                bbox_masks = nn.functional.pad(bbox_masks, (bbox_pad[0], bbox_pad[0], bbox_pad[1], bbox_pad[1]), mode='constant', value=0)
                bbox_masks = nn.functional.interpolate(bbox_masks, size=(H, W), mode='nearest')

        if self.split_heatmaps_by == "visibility":
            # This is OK apart from nonlinearity on the 'visibility' border
            heatmaps = heatmaps1 + heatmaps2
        elif self.split_heatmaps_by in ["in/all", "in/out"]:
            if bbox_masks is not None:

                # heatmaps = heatmaps2

                # Sophisticated way to merge heatmaps
                # value_thr = 0.2
                bbox_masks = bbox_masks.repeat(1, C, 1, 1)
                bbox_masks = nn.functional.interpolate(bbox_masks, size=(H, W), mode='nearest')
                bbox_masks = bbox_masks.to(heatmaps1.device)
                
                hin = heatmaps1.reshape((B*C, H*W))
                hout = heatmaps2.reshape((B*C, H*W))
                bbox_masks_flat = bbox_masks.reshape((B*C, H*W))

                # Find if the max value of the Out-heatmap is in the bbox
                hout_amax = hout.argmax(dim=-1)
                hin_amax = hin.argmax(dim=-1)
                row_idx = torch.arange(0, B*C, device=bbox_masks_flat.device)
                hin_in = bbox_masks_flat[row_idx, hin_amax] > 0.5
                hout_in = bbox_masks_flat[row_idx, hout_amax] > 0.5
                heatmaps = torch.zeros_like(hin)
                heatmaps[hout_in, :] = hin[hout_in, :]
                heatmaps[~hout_in, :] = hout[~hout_in, :]
                
                # hin_masked = (heatmaps1*bbox_masks).reshape((B*C, H*W))
                # hout_masked = (heatmaps2*(1-bbox_masks)).reshape((B*C, H*W))
                # hin_value = hin_masked.max(dim=-1).values > value_thr
                # hout_value = hout_masked.max(dim=-1).values > value_thr

                # disagree = (hin_value & hout_value)# | (~hin_value & ~hout_value)
                # agree = ~disagree
                # heatmaps = torch.zeros_like(hin)
                # heatmaps[~hout_value] = hin[~hout_value]
                # heatmaps[hout_value] = hout[hout_value]
                
                # preds1 = self.decode(heatmaps1)
                # preds2 = self.decode(heatmaps2)
                # kpts1 = np.stack([p['keypoints'] for p in preds1], axis=0)
                # kpts2 = np.stack([p['keypoints'] for p in preds2], axis=0)
                # kpts1_score = np.stack([p['keypoint_scores'] for p in preds1], axis=0)
                # kpts2_score = np.stack([p['keypoint_scores'] for p in preds2], axis=0)
                # kpts1[kpts1_score < value_thr, :] = 0
                # kpts2[kpts2_score < value_thr, :] = 0
                # dists = np.linalg.norm(kpts1 - kpts2, axis=-1).squeeze()
                # htm_diagonal = np.sqrt(H**2 + W**2)
                # close_thr = 0.05 * htm_diagonal
                # heatmaps = torch.zeros_like(heatmaps1)
                # agree = torch.from_numpy(dists < close_thr).to(heatmaps1.device)
                # heatmaps[agree, :, :] = heatmaps1[agree, :, :]
                # heatmaps[~agree, :, :] = heatmaps2[~agree, :, :]

                # # Weighted way to merge heatmaps
                # hin_masked = (heatmaps1*bbox_masks).reshape((B*C, H*W))
                # hout_masked = (heatmaps2*(1-bbox_masks)).reshape((B*C, H*W))
                # hin_value = hin_masked.max(dim=-1).values
                # hout_value = hout_masked.max(dim=-1).values
                # heatmaps = hin_masked*hin_value[:, None] + hout_masked*hout_value[:, None]

                self.tmp_count = 1
                self.tmp_sum = 1
                # self.tmp_count = agree.sum().item()
                # self.tmp_sum = agree.size().numel()
                heatmaps = heatmaps.reshape((B, C, H, W))

                # # Masked way to merge heatmaps
                # heatmaps = heatmaps1*bbox_masks + heatmaps2*(1-bbox_masks)
                
                # # Simple average
                # heatmaps = heatmaps1 + heatmaps2
            else:
                # heatmaps = heatmaps1
                heatmaps = heatmaps1 + heatmaps2
        else:
            raise ValueError(f"Unknown split_heatmaps_by: {self.split_heatmaps_by}")
        
        heatmaps = heatmaps.reshape((B, C, H, W))
        if return_stats:
            hin_in = hin_in.reshape((B, C))
            hout_in = hout_in.reshape((B, C))
            return heatmaps, hin_in, hout_in
        else:
            return heatmaps

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward the network. The input is multi scale feature maps and the
        output is (1) the heatmap, (2) probability, (3) visibility, (4) oks and (5) error.

        Args:
            feats (Tensor): Multi scale feature maps.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: outputs.
        """
        x = feats[-1]

        if self.detach_all:
            x = x.detach()

        heatmaps1 = self.forward_first_heatmap(x)
        heatmaps2 = self.forward_second_heatmap(x)
        probabilities = self.forward_probability(x)
        visibilities = self.forward_visibility(x)
        oks = self.forward_oks(x)
        errors = self.forward_error(x)

        return heatmaps1, heatmaps2, probabilities, visibilities, oks, errors
    
    def forward_first_heatmap(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            x (Tensor): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)
        x = self.normalize_layer(x)
        return x
    
    def forward_second_heatmap(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            x (Tensor): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        if self.detach_second_heatmaps:
            x = x.detach()
        x = self.second_head(x)
        return x
    
    def forward_probability(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the probability.

        Args:
            x (Tensor): Multi scale feature maps.
            detach (bool): Whether to detach the probability from gradient

        Returns:
            Tensor: output probability.
        """
        if self.detach_probability:
            x = x.detach()
        x = self.probability_layers(x)
        return x

    def forward_visibility(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the visibility.

        Args:
            x (Tensor): Multi scale feature maps.
            detach (bool): Whether to detach the visibility from gradient

        Returns:
            Tensor: output visibility.
        """
        if self.detach_visibility:
            x = x.detach()
        x = self.visibility_layers(x)
        return x
    
    def forward_oks(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the oks.

        Args:
            x (Tensor): Multi scale feature maps.

        Returns:
            Tensor: output oks.
        """
        x = x.detach()
        x = self.oks_layers(x)
        return x
    
    def forward_error(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the euclidean error.

        Args:
            x (Tensor): Multi scale feature maps.

        Returns:
            Tensor: output error.
        """
        x = x.detach()
        x = self.error_layers(x)
        return x

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """

        if "bbox_mask" in batch_data_samples[0].gt_instances:
            bbox_masks = torch.stack(
                [torch.from_numpy(d.gt_instances.bbox_mask) for d in batch_data_samples])
        else:
            bbox_masks = None

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats
            _htm1, _htm2, _prob, _vis, _oks, _err = self.forward(_feats)
            _htm1_flip, _htm2_flip, _prob_flip, _vis_flip, _oks_flip, _err_flip = self.forward(_feats_flip)

            B, C, H, W = _htm1.shape

            # Flip back the keypoints
            _htm1_flip = flip_heatmaps(
                _htm1_flip,
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            _htm2_flip = flip_heatmaps(
                _htm2_flip,
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))

            # Merge heatmaps
            heatmaps1 = (_htm1 + _htm1_flip) * 0.5
            heatmaps2 = (_htm2 + _htm2_flip) * 0.5
            # heatmaps, hin_in, hout_in = self.merge_heatmaps(heatmaps1, heatmaps2, bbox_masks,return_stats=True)

            # Flip back scalars
            _prob_flip = _prob_flip[:, flip_indices]
            _vis_flip = _vis_flip[:, flip_indices]
            _oks_flip = _oks_flip[:, flip_indices]
            _err_flip = _err_flip[:, flip_indices]
            
            probabilities = (_prob + _prob_flip) * 0.5
            visibilities = (_vis + _vis_flip) * 0.5
            oks = (_oks + _oks_flip) * 0.5
            errors = (_err + _err_flip) * 0.5
        else:
            heatmaps1, heatmaps2, probabilities, visibilities, oks, errors = self.forward(feats)
            # heatmaps, hin_in, hout_in = self.merge_heatmaps(heatmaps1, heatmaps2, bbox_masks, return_stats=True)
            B, C, H, W = heatmaps1.shape

        preds, hin_in, hout_in = self._merge_predictions(heatmaps1, heatmaps2, bbox_masks, return_masks=True)
        hin_in = torch.tensor(hin_in.reshape((B, C)), device='cpu').bool()
        hout_in = torch.tensor(hout_in.reshape((B, C)), device='cpu').bool()
        # preds = self.decode(heatmaps)

        probabilities = to_numpy(probabilities).reshape((B, 1, C))
        visibilities = to_numpy(visibilities).reshape((B, 1, C))
        oks = to_numpy(oks).reshape((B, 1, C))
        errors = to_numpy(errors).reshape((B, 1, C))

        if (
            "bboxes" in batch_data_samples[0].gt_instances and
            "keypoints" in batch_data_samples[0].gt_instances and
            bbox_masks is not None):
            gt_bboxes = torch.stack(
                [torch.from_numpy(d.gt_instances.bboxes) for d in batch_data_samples]).squeeze()
            gt_keypoints = torch.stack(
                [torch.from_numpy(d.gt_instances.keypoints) for d in batch_data_samples]).squeeze()
            gt_annotated = torch.stack(
                [torch.from_numpy(d.gt_instances.keypoints_visible) for d in batch_data_samples]).squeeze()
            gt_annotated = gt_annotated > 0.5
            
            # Move all tensors to the same device
            gt_bboxes = gt_bboxes.to(hin_in.device)
            gt_keypoints = gt_keypoints.to(hin_in.device)
            gt_annotated = gt_annotated.to(hin_in.device)
            
            kpts_in_bbox = (
                (gt_keypoints[:, :, 0] >= gt_bboxes[:, None, 0]) &
                (gt_keypoints[:, :, 0] <= gt_bboxes[:, None, 2]) &
                (gt_keypoints[:, :, 1] >= gt_bboxes[:, None, 1]) &
                (gt_keypoints[:, :, 1] <= gt_bboxes[:, None, 3])
            )
            kpts_dist_to_bbox = torch.stack([
                torch.min(torch.abs(gt_keypoints[:, :, 0] - gt_bboxes[:, None, 0]),
                          torch.abs(gt_keypoints[:, :, 0] - gt_bboxes[:, None, 2])),
                torch.min(torch.abs(gt_keypoints[:, :, 1] - gt_bboxes[:, None, 1]),
                          torch.abs(gt_keypoints[:, :, 1] - gt_bboxes[:, None, 3]))
            ], dim=-1)
            kpts_dist_to_bbox = kpts_dist_to_bbox.min(dim=-1).values
            bbox_diag = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0])**2 + (gt_bboxes[:, 3] - gt_bboxes[:, 1])**2)
            kpts_dist_to_bbox = kpts_dist_to_bbox / bbox_diag[:, None]

            img_paths = [d.img_path for d in batch_data_samples]
            self.update_results_log(img_paths, kpts_in_bbox, hin_in, hout_in, kpts_dist_to_bbox, mask=gt_annotated)

            first_dataset_name = batch_data_samples[0].dataset_name
            last_dataset_name = batch_data_samples[-1].dataset_name
            if first_dataset_name != last_dataset_name:
                # Split the batch by dataset
                idxs = [b.dataset_name == self.in_out_stats['dataset_name'] for b in batch_data_samples]
                idxs = torch.tensor(idxs, dtype=torch.bool)
                self.update_in_out_stats(kpts_in_bbox[idxs], hin_in[idxs], hout_in[idxs], kpts_dist_to_bbox[idxs], mask=gt_annotated[idxs])
                self.print_in_out_stats()
                
                # Reset and update the stats
                for key, _ in self.in_out_stats.items():
                    self.in_out_stats[key] = 0
                self.in_out_stats['dataset_name'] = last_dataset_name
                self.update_in_out_stats(kpts_in_bbox[~idxs], hin_in[~idxs], hout_in[~idxs], kpts_dist_to_bbox[~idxs], mask=gt_annotated[~idxs])
                
            elif last_dataset_name != self.in_out_stats['dataset_name']:
                # Print the stats for the previous dataset
                self.print_in_out_stats()
                # Reset the stats
                for key, _ in self.in_out_stats.items():
                    self.in_out_stats[key] = 0
                self.in_out_stats['dataset_name'] = last_dataset_name
            else:
                self.update_in_out_stats(kpts_in_bbox, hin_in, hout_in, kpts_dist_to_bbox, mask=gt_annotated)

            if len(batch_data_samples) != 64:
                self.print_in_out_stats()
                self.save_results_log()

            
        # Normalize errors by dividing with the diagonal of the heatmap
        htm_diagonal = np.sqrt(H**2 + W**2)
        errors = errors / htm_diagonal

        for pi, p in enumerate(preds):
            p.set_field(probabilities[pi], "keypoints_probs")
            p.set_field(visibilities[pi], "keypoints_visible")
            p.set_field(oks[pi], "keypoints_oks")
            p.set_field(errors[pi], "keypoints_error")

        # print("Agreement rate : {:.2f} %".format(self.tmp_count/self.tmp_sum*100))

        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(
                    # heatmaps=hm,
                    heatmaps1=hm1,
                    heatmaps2=hm2
                ) for hm, hm1, hm2 in zip(
                    # heatmaps.detach(),
                    heatmaps1.detach(),
                    heatmaps2.detach()
                )
            ]
            return preds, pred_fields
        else:
            return preds

    def update_in_out_stats(self, in_bbox, hin_in, hout_in, dists, mask=None):
        if mask is None:
            mask = torch.ones_like(in_bbox)
        else:
            mask = mask.to(in_bbox.device)

        current_p_in = self.in_out_stats['p(in)']
        current_p_out = self.in_out_stats['p(out)'] + 1e-10
        current_in_out_ratio = current_p_in / current_p_out
        p_in = in_bbox[mask].sum()
        p_out = (~in_bbox[mask]).sum() + 1e-10
        in_out_ratio = p_in / p_out
        
        self.in_out_stats['p(in)'] += p_in
        self.in_out_stats['p(out)'] += p_out
        
        self.in_out_stats['p_in(in|in)'] += (in_bbox[mask] & hin_in[mask]).sum()
        self.in_out_stats['p_in(out|out)'] += ((~in_bbox[mask]) & (~hin_in[mask])).sum()
        self.in_out_stats['p_in(in|out)'] += ((~in_bbox[mask]) & hin_in[mask]).sum()
        self.in_out_stats['p_in(out|in)'] += (in_bbox[mask] & (~hin_in[mask])).sum()

        self.in_out_stats['p_out(in|in)'] += (in_bbox[mask] & hout_in[mask]).sum()
        self.in_out_stats['p_out(out|out)'] += ((~in_bbox[mask]) & (~hout_in[mask])).sum()
        self.in_out_stats['p_out(in|out)'] += ((~in_bbox[mask]) & hout_in[mask]).sum()
        self.in_out_stats['p_out(out|in)'] += (in_bbox[mask] & (~hout_in[mask])).sum()

        self.in_out_stats["dist_in(in|in)"] += dists[mask & in_bbox & hin_in].sum()
        self.in_out_stats["dist_in(in|out)"] += dists[mask & ~in_bbox & hin_in].sum()
        self.in_out_stats["dist_in(out|out)"] += dists[mask & ~in_bbox & ~hin_in].sum()
        self.in_out_stats["dist_in(out|in)"] += dists[mask & in_bbox & ~hin_in].sum()
        
        self.in_out_stats["dist_out(in|in)"] += dists[mask & in_bbox & hout_in].sum()
        self.in_out_stats["dist_out(in|out)"] += dists[mask & ~in_bbox & hout_in].sum()
        self.in_out_stats["dist_out(out|out)"] += dists[mask & ~in_bbox & ~hout_in].sum()
        self.in_out_stats["dist_out(out|in)"] += dists[mask & in_bbox & ~hout_in].sum()

    def update_results_log(self, img_paths, in_bbox, hin_in, hout_in, dists, mask=None):
        if mask is None:
            mask = torch.ones_like(in_bbox)
        else:
            mask = mask.to(in_bbox.device)
        
        for i in range(len(img_paths)):
            img_p = img_paths[i]
            img_p = "/".join(img_p.split("/")[-3:])
            self.results_log[img_p] = {}
            for j in range(in_bbox.shape[1]):
                joint_name = KEYPOINT_NAMES[j]
                in_b = in_bbox[i, j].item()
                hin_i = hin_in[i, j].item()
                hout_i = hout_in[i, j].item()
                dist = dists[i, j].item()
                mask_i = mask[i, j].item()

                if not mask_i:
                    cat = "not_annotated"
                elif in_b and not hout_i:
                    cat = "gtin_predout"
                elif not in_b and hout_i:
                    cat = "gtout_predin"
                elif not in_b:
                    cat = "gtout_else"
                else:
                    cat = "else"

                self.results_log[img_p][joint_name] = {
                    "in_bbox": in_b,
                    "hin_in": hin_i,
                    "hout_in": hout_i,
                    "dist": dist,
                    "mask": mask_i,
                    "category": cat
                }

    def save_results_log(self):
        with open("results_log.json", "w") as f:
            json.dump(self.results_log, f, indent=2)

    def print_in_out_stats(self):
        for key, value in self.in_out_stats.items():
            if isinstance(value, float) or isinstance(value, int):
                self.in_out_stats[key] += 1e-10

        # Normalize stats
        num_in = self.in_out_stats['p(in)']
        num_out = self.in_out_stats['p(out)']
        num_samples = num_in + num_out

        print("\nDataset '{}':".format(self.in_out_stats['dataset_name']))
        print("{:<8}|{:<20}|{:<20}".format("", "IN", "OUT"))
        print("{:<8}+{:<20}+{:<20}".format("-"*8, "-"*20, "-"*20))
        
        print("{:<8}|{:8.2f} % ({:6d}) |{:8.2f} % ({:6d})".format(
            "GT",
            self.in_out_stats['p(in)']/num_samples * 100,
            int(self.in_out_stats['p(in)']),
            self.in_out_stats['p(out)']/num_samples * 100,
            int(self.in_out_stats['p(out)']),
        ))
        
        print("{:<8}+{:<20}+{:<20}".format("- "*4, "- "*10, "- "*10))
        
        print("{:<8}|{:6.2f} % | {:6.2f} % |{:6.2f} % | {:6.2f} %".format(
            "In-HTM",
            self.in_out_stats['p_in(in|in)']/num_in * 100,
            self.in_out_stats['p_in(out|in)']/num_in * 100,
            self.in_out_stats['p_in(out|out)']/num_out * 100,
            self.in_out_stats['p_in(in|out)']/num_out * 100,
        ))
        print("{:<8}|{:8.2f} | {:8.2f} |{:8.2f} | {:8.2f}".format(
            "",
            self.in_out_stats['dist_in(in|in)'] / self.in_out_stats['p_in(in|in)'] * 100,
            self.in_out_stats['dist_in(out|in)'] / self.in_out_stats['p_in(out|in)'] * 100,
            self.in_out_stats['dist_in(in|out)'] / self.in_out_stats['p_in(out|out)'] * 100,
            self.in_out_stats['dist_in(out|out)'] / self.in_out_stats['p_in(in|out)'] * 100,
        ))

        print("{:<8}+{:<20}+{:<20}".format("- "*4, "- "*10, "- "*10))
        
        print("{:<8}|{:6.2f} % | {:6.2f} % |{:6.2f} % | {:6.2f} %".format(
            "Out-HTM",
            self.in_out_stats['p_out(in|in)']/num_in * 100,
            self.in_out_stats['p_out(out|in)']/num_in * 100,
            self.in_out_stats['p_out(out|out)']/num_out * 100,
            self.in_out_stats['p_out(in|out)']/num_out * 100,
        ))
        print("{:<8}|{:8.2f} | {:8.2f} |{:8.2f} | {:8.2f}".format(
            "",
            self.in_out_stats['dist_out(in|in)'] / self.in_out_stats['p_out(in|in)'] * 100,
            self.in_out_stats['dist_out(out|in)'] / self.in_out_stats['p_out(out|in)'] * 100,
            self.in_out_stats['dist_out(in|out)'] / self.in_out_stats['p_out(out|out)'] * 100,
            self.in_out_stats['dist_out(out|out)'] / self.in_out_stats['p_out(in|out)'] * 100,
        ))
        print("{:<8}+{:<20}+{:<20}".format("-"*8, "-"*20, "-"*20))
        
    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """
        dt_heatmaps1, dt_heatmaps2, dt_probs, dt_vis, dt_oks, dt_errs = self.forward(feats)
        device=dt_heatmaps1.device
        B, C, H, W = dt_heatmaps1.shape
        
        # Extract GT data
        gt_in_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        gt_out_heatmaps = np.stack(
            [d.gt_instances.out_heatmaps for d in batch_data_samples])
        gt_probs = np.stack(
            [d.gt_instances.in_image.astype(int) for d in batch_data_samples])
        gt_annotated = np.stack(
            [d.gt_instances.keypoints_visible.astype(int) for d in batch_data_samples])
        gt_vis = np.stack(
            [d.gt_instances.keypoints_visibility.astype(int) for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])
        out_keypoint_weights = np.stack([
            d.gt_instances.out_kpt_weights for d in batch_data_samples
        ])
        gt_in_image = np.stack(
            [d.gt_instances.keypoints_in_image.astype(int) for d in batch_data_samples])

        if "bbox_mask" in batch_data_samples[0].gt_instances:
            bbox_masks = torch.stack(
                [torch.from_numpy(d.gt_instances.bbox_mask) for d in batch_data_samples])
        else:
            bbox_masks = None
        merged_dt_heatmaps = self.merge_heatmaps(dt_heatmaps1, dt_heatmaps2, bbox_masks)
        
            
        # Compute GT errors and OKS -> ToDo: Edit and check for heatmap merging
        if self.freeze_error:
            gt_errs = np.zeros((B, C, 1))
        else:
            gt_errs = self._error_from_heatmaps(gt_out_heatmaps, merged_dt_heatmaps)
        if self.freeze_oks:
            gt_oks = np.zeros((B, C, 1))
            oks_weight = np.zeros((B, C, 1))
        else:
            gt_oks, oks_weight = self._oks_from_heatmaps(
                gt_out_heatmaps,
                merged_dt_heatmaps,
                gt_probs & gt_annotated,
            )

        # Convert everything to tensors
        gt_probs = torch.tensor(gt_probs, device=device, dtype=dt_probs.dtype)
        gt_vis = torch.tensor(gt_vis, device=device, dtype=dt_vis.dtype)
        gt_in_image = torch.tensor(gt_in_image, device=device)
        gt_annotated = torch.tensor(gt_annotated, device=device)
        gt_oks = torch.tensor(gt_oks, device=device, dtype=dt_oks.dtype)
        oks_weight = torch.tensor(oks_weight, device=device, dtype=dt_oks.dtype)
        gt_errs = torch.tensor(gt_errs, device=device, dtype=dt_errs.dtype)
        out_keypoint_weights = torch.tensor(out_keypoint_weights, device=device, dtype=dt_probs.dtype)
        gt_out_heatmaps = torch.tensor(gt_out_heatmaps, device=device, dtype=dt_heatmaps1.dtype)

        # Reshape everything to comparable shapes
        gt_in_heatmaps = gt_in_heatmaps.view((B, C, H, W))
        gt_out_heatmaps = gt_out_heatmaps.view((B, C, H, W))
        dt_heatmaps1 = dt_heatmaps1.view((B, C, H, W))
        dt_heatmaps2 = dt_heatmaps2.view((B, C, H, W))
        gt_probs = gt_probs.view((B, C))
        dt_probs = dt_probs.view((B, C))
        gt_vis = gt_vis.view((B, C))
        dt_vis = dt_vis.view((B, C))
        gt_oks = gt_oks.view((B, C))
        dt_oks = dt_oks.view((B, C))
        gt_errs = gt_errs.view((B, C))
        dt_errs = dt_errs.view((B, C))
        gt_in_image = gt_in_image.view((B, C))
        keypoint_weights = keypoint_weights.view((B, C))
        out_keypoint_weights = out_keypoint_weights.view((B, C))
        gt_annotated = gt_annotated.view((B, C))
        # oks_weight = oks_weight.view((B, C))

        annotated_in = (gt_annotated & (gt_probs > 0.5)).bool()

        # calculate losses
        losses = dict()
        if self.split_heatmaps_by == "visibility":
            heatmap1_weights = gt_vis.to(torch.uint8) & annotated_in
            heatmap2_weights = ~ gt_vis.to(torch.uint8) & annotated_in
        elif self.split_heatmaps_by == "in/out":
            heatmap1_weights = gt_in_image & annotated_in
            heatmap2_weights = ~gt_in_image & annotated_in
        elif self.split_heatmaps_by == "in/all":
            heatmap1_weights = gt_in_image & annotated_in
            heatmap2_weights = annotated_in
        else:
            raise ValueError(f"Unknown split_heatmaps_by: {self.split_heatmaps_by}")

        heatmap1_loss    = self.keypoint_loss_module(dt_heatmaps1, gt_in_heatmaps, heatmap1_weights)
        heatmap2_loss    = self.keypoint_loss_module(dt_heatmaps2, gt_out_heatmaps, heatmap2_weights)
        probability_loss = self.probability_loss_module(dt_probs, gt_probs, gt_annotated)
        visibility_loss  = self.visibility_loss_module(dt_vis, gt_vis, annotated_in.float())
        oks_loss         = self.oks_loss_module(dt_oks, gt_oks, annotated_in.float())
        error_loss       = self.error_loss_module(dt_errs, gt_errs, annotated_in.float())
        
        losses.update(
            loss_kpt=heatmap1_loss,
            loss_kpt2=heatmap2_loss,
            loss_probability=probability_loss,
            loss_visibility=visibility_loss,
            loss_oks=oks_loss,
            loss_error=error_loss)
        
        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            dt_preds = self._merge_predictions(dt_heatmaps1, dt_heatmaps2, bbox_masks)
            gt_preds = self._merge_predictions(gt_in_heatmaps, gt_out_heatmaps, bbox_masks)
            acc_pose = self.get_pose_accuracy_from_preds(
                dt_preds, gt_preds,
                annotated_in > 0.5,
                H=bbox_masks.shape[2], W=bbox_masks.shape[3]
            )
            losses.update(acc_pose=acc_pose)
            acc_pose1 = self.get_pose_accuracy(
                dt_heatmaps1, gt_in_heatmaps, heatmap1_weights > 0.5
            )
            losses.update(acc_pose1=acc_pose1)
            acc_pose2 = self.get_pose_accuracy(
                dt_heatmaps2, gt_out_heatmaps, heatmap2_weights > 0.5
            )
            losses.update(acc_pose2=acc_pose2)

            # Calculate the best binary accuracy for probability
            acc_prob, _ = self.get_binary_accuracy(
                dt_probs,
                gt_probs,
                gt_annotated > 0.5,
                force_balanced=True,
            )
            losses.update(acc_prob=acc_prob)

            # Calculate the best binary accuracy for visibility
            acc_vis, _ = self.get_binary_accuracy(
                dt_vis,
                gt_vis,
                annotated_in,
                force_balanced=True,
            )
            losses.update(acc_vis=acc_vis)

            # Calculate the MAE for OKS
            acc_oks = self.get_mae(
                dt_oks,
                gt_oks,
                annotated_in,
            )
            losses.update(mae_oks=acc_oks)

            # Calculate the MAE for euclidean error
            acc_err = self.get_mae(
                dt_errs,
                gt_errs,
                annotated_in,
            )
            losses.update(mae_err=acc_err)

            # Calculate the MAE between Euclidean error and OKS
            err_to_oks_mae = self.get_mae(
                self.error_to_OKS(dt_errs, area=H*W),
                gt_oks,
                annotated_in,
            )
            losses.update(mae_err_to_oks=err_to_oks_mae)

        return losses
    
    def _merge_predictions(self, dt_in, dt_out, bbox_masks, return_masks=False):
        B, C, H, W = bbox_masks.shape
        if not return_masks:
            device = dt_in.device
        else:
            bbox_masks = bbox_masks.cpu().numpy()
        dt_in = self.decode(dt_in, htm_type="in")
        dt_out = self.decode(dt_out, htm_type="out")
        kpts_in = np.stack([d.keypoints for d in dt_in])
        kpts_in_int = kpts_in.astype(int)
        scores_in = np.stack([d.keypoint_scores for d in dt_in])
        kpts_out = np.stack([d.keypoints for d in dt_out])
        kpts_out_int = kpts_out.astype(int)
        scores_out = np.stack([d.keypoint_scores for d in dt_out])
        _, _, K, D = kpts_out.shape        
        batch_idx = np.arange(B)[:, None, None]
        channel_idx = np.arange(C)[None, :, None]
        
        # Find out hout_in
        out_mask = (kpts_out < 0) | (kpts_out >= np.array([W, H]))
        kpts_out_int[out_mask] = 0  # Use any value in teh bbox as it will be rewritten later
        hout_in = bbox_masks[batch_idx, channel_idx, kpts_out_int[:, :, :, 1], kpts_out_int[:, :, :, 0]]
        out_mask = out_mask.any(axis=-1)
        try:
            hout_in[out_mask] = 0
        except IndexError:
            breakpoint()

        # Find out hin_in
        out_mask = (kpts_in < 0) | (kpts_in >= np.array([W, H]))
        kpts_in_int[out_mask] = 0   # Use any value in teh bbox as it will be rewritten later
        hin_in = bbox_masks[batch_idx, channel_idx, kpts_in_int[:, :, :, 1], kpts_in_int[:, :, :, 0]]
        out_mask = out_mask.any(axis=-1)
        hin_in[out_mask] = 0

        # If the Out-htm has points inside the bbox, take the In-htm. Otherwise, take the Out-htm.   
        merge_kpts = kpts_out
        merge_scores = scores_out
        merge_kpts[hout_in, :] = kpts_in[hout_in, :]
        merge_scores[hout_in] = scores_in[hout_in]
        
        if return_masks:
            # Used for testing in 'predict' function
            merged_preds = [InstanceData(keypoints=kpts, keypoint_scores=scores) for kpts, scores in zip(merge_kpts, merge_scores)]
            return merged_preds, hin_in, hout_in
        else:
            # Used for training in 'loss' function
            merged_preds = merge_kpts.reshape((B, K, 2))
            merged_preds = torch.tensor(merged_preds, device=device)
            return merged_preds

    def decode(self, batch_outputs: Union[Tensor,
                                          Tuple[Tensor]],
                                          htm_type='out') -> InstanceList:
        """Decode keypoints from outputs.

        Args:
            batch_outputs (Tensor | Tuple[Tensor]): The network outputs of
                a data batch

        Returns:
            List[InstanceData]: A list of InstanceData, each contains the
            decoded pose information of the instances of one data sample.
        """

        def _pack_and_call(args, func):
            if not isinstance(args, tuple):
                args = (args, )
            return func(*args, htm_type=htm_type)

        if self.decoder is None:
            raise RuntimeError(
                f'The decoder has not been set in {self.__class__.__name__}. '
                'Please set the decoder configs in the init parameters to '
                'enable head methods `head.predict()` and `head.decode()`')

        if self.decoder.support_batch_decoding:
            batch_keypoints, batch_scores = _pack_and_call(
                batch_outputs, self.decoder.batch_decode)
            if isinstance(batch_scores, tuple) and len(batch_scores) == 2:
                batch_scores, batch_visibility = batch_scores
            else:
                batch_visibility = [None] * len(batch_keypoints)

        else:
            batch_output_np = to_numpy(batch_outputs, unzip=True)
            batch_keypoints = []
            batch_scores = []
            batch_visibility = []
            for outputs in batch_output_np:
                keypoints, scores = _pack_and_call(outputs,
                                                   self.decoder.decode)
                batch_keypoints.append(keypoints)
                if isinstance(scores, tuple) and len(scores) == 2:
                    batch_scores.append(scores[0])
                    batch_visibility.append(scores[1])
                else:
                    batch_scores.append(scores)
                    batch_visibility.append(None)

        preds = []
        for keypoints, scores, visibility in zip(batch_keypoints, batch_scores,
                                                 batch_visibility):
            pred = InstanceData(keypoints=keypoints, keypoint_scores=scores)
            if visibility is not None:
                pred.keypoints_visible = visibility
            preds.append(pred)

        return preds

    def get_pose_accuracy_from_preds(self, dt, gt, mask, H=256, W=192):
        device = gt.device
        mask = to_numpy(mask)
        B, _, = mask.shape
        norm = np.tile(np.array([[H, W]]), (B, 1))
        if isinstance(dt, torch.Tensor) and dt.device != 'cpu':
            dt = dt.cpu()
        if isinstance(gt, torch.Tensor) and gt.device != 'cpu':
            gt = gt.cpu()
        _, acc_pose, _ = keypoint_pck_accuracy(dt, gt, mask, thr=0.05, norm_factor=norm)
        acc_pose = torch.tensor(acc_pose, device=device)
        return acc_pose
    
    def get_pose_accuracy(self, dt, gt, mask):
        """Calculate the accuracy of predicted pose."""
        _, avg_acc, _ = pose_pck_accuracy(
            output=to_numpy(dt),
            target=to_numpy(gt),
            mask=to_numpy(mask),
        )
        acc_pose = torch.tensor(avg_acc, device=gt.device)
        return acc_pose
    
    def get_binary_accuracy(self, dt, gt, mask, force_balanced=False):
        """Calculate the binary accuracy."""
        assert dt.shape == gt.shape
        device = gt.device
        dt = to_numpy(dt)
        gt = to_numpy(gt)
        mask = to_numpy(mask)

        dt = dt[mask]
        gt = gt[mask]
        gt = gt.astype(bool)

        if force_balanced:
            # Force the number of positive and negative samples to be balanced
            pos_num = np.sum(gt)
            neg_num = len(gt) - pos_num
            num = min(pos_num, neg_num)
            if num == 0:
                return torch.tensor([0.0], device=device), torch.tensor([0.0], device=device)
            pos_idx = np.where(gt)[0]
            neg_idx = np.where(~gt)[0]

            # Randomly sample the same number of positive and negative samples
            np.random.shuffle(pos_idx)
            np.random.shuffle(neg_idx)
            idx = np.concatenate([pos_idx[:num], neg_idx[:num]])
            dt = dt[idx]
            gt = gt[idx]

        n_samples = len(gt)
        thresholds = np.arange(0.1, 1.0, 0.05)
        preds = (dt[:, None] > thresholds)
        correct = preds == gt[:, None]
        counts = correct.sum(axis=0)

        # Find the threshold that maximizes the accuracy
        best_idx = np.argmax(counts)
        best_threshold = thresholds[best_idx]
        best_acc = counts[best_idx] / n_samples

        best_acc = torch.tensor(best_acc, device=device).float()
        best_threshold = torch.tensor(best_threshold, device=device).float()
        return best_acc, best_threshold

    def get_mae(self, dt, gt, mask):
        """Calculate the mean absolute error."""
        assert dt.shape == gt.shape
        device = gt.device
        dt = to_numpy(dt)
        gt = to_numpy(gt)
        mask = to_numpy(mask).astype(bool)
        
        dt = dt[mask]
        gt = gt[mask]
        mae = np.abs(dt - gt).mean()

        mae = torch.tensor(mae, device=device)
        return mae

    def error_to_OKS(self, error, area=1.0):
        """Convert the error to OKS."""
        sigmas = np.array(
                [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        if isinstance(error, torch.Tensor):
            sigmas = torch.tensor(sigmas, device=error.device)
        norm_error = error**2 / sigmas**2 / area / 2.0
        return torch.exp(-norm_error)

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`TopdownHeatmapSimpleHead` (before MMPose v1.0.0) to a
        compatible format of :class:`HeatmapHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            k = _k[len(prefix):]
            # In old version, "final_layer" includes both intermediate
            # conv layers (new "conv_layers") and final conv layers (new
            # "final_layer").
            #
            # If there is no intermediate conv layer, old "final_layer" will
            # have keys like "final_layer.xxx", which should be still
            # named "final_layer.xxx";
            #
            # If there are intermediate conv layers, old "final_layer"  will
            # have keys like "final_layer.n.xxx", where the weights of the last
            # one should be renamed "final_layer.xxx", and others should be
            # renamed "conv_layers.n.xxx"
            k_parts = k.split('.')
            if k_parts[0] == 'final_layer':
                if len(k_parts) == 3:
                    assert isinstance(self.conv_layers, nn.Sequential)
                    idx = int(k_parts[1])
                    if idx < len(self.conv_layers):
                        # final_layer.n.xxx -> conv_layers.n.xxx
                        k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                    else:
                        # final_layer.n.xxx -> final_layer.xxx
                        k_new = 'final_layer.' + k_parts[2]
                else:
                    # final_layer.xxx remains final_layer.xxx
                    k_new = k
            else:
                k_new = k

            state_dict[prefix + k_new] = v


def compute_oks(gt, dt, use_area=True, per_kpt=False):
    sigmas = np.array(
                [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    vars = (sigmas * 2)**2
    k = len(sigmas)
    visibility_condition = lambda x: x > 0
    g = np.array(gt['keypoints']).reshape(k, 3)
    xg = g[:, 0]; yg = g[:, 1]; vg = g[:, 2]
    k1 = np.count_nonzero(visibility_condition(vg))
    bb = gt['bbox']
    x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
    y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
    
    d = np.array(dt['keypoints']).reshape((k, 3))
    xd = d[:, 0]; yd = d[:, 1]
            
    if k1>0:
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg

    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        z = np.zeros((k))
        dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
        dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)

    if use_area:
        e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
    else:
        tmparea = gt['bbox'][3] * gt['bbox'][2] * 0.53
        e = (dx**2 + dy**2) / vars / (tmparea+np.spacing(1)) / 2
        
    if per_kpt:
        oks = np.exp(-e)
        if k1 > 0:
            oks[~visibility_condition(vg)] = 0

    else:
        if k1 > 0:
            e=e[visibility_condition(vg)]
        oks = np.sum(np.exp(-e)) / e.shape[0]

    return oks