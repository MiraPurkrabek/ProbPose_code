# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmpose.registry import MODELS


@MODELS.register_module()
class KeypointMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self, use_target_weight: bool = False, skip_empty_channel: bool = False, loss_weight: float = 1.0):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight

    def forward(
        self,
        output: Tensor,
        target: Tensor,
        target_weights: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        per_keypoint: bool = False,
        per_pixel: bool = False,
    ) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """

        _mask = self._get_mask(target, target_weights, mask)

        _loss = F.mse_loss(output, target, reduction="none")

        if _mask is not None:
            loss = _loss * _mask

        if per_pixel:
            pass
        elif per_keypoint:
            loss = loss.mean(dim=(2, 3))
        else:
            loss = loss.mean()

        return loss * self.loss_weight

    def _get_mask(self, target: Tensor, target_weights: Optional[Tensor], mask: Optional[Tensor]) -> Optional[Tensor]:
        """Generate the heatmap mask w.r.t. the given mask, target weight and
        `skip_empty_channel` setting.

        Returns:
            Tensor: The mask in shape (B, K, *) or ``None`` if no mask is
            needed.
        """
        # Given spatial mask
        if mask is not None:
            # check mask has matching type with target
            assert mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1 for d_m, d_t in zip(mask.shape, target.shape)
            ), (f"mask and target have mismatched shapes {mask.shape} v.s." f"{target.shape}")

        # Mask by target weights (keypoint-wise mask)
        if target_weights is not None:
            # check target weight has matching shape with target
            assert target_weights.ndim in (2, 4) and target_weights.shape == target.shape[: target_weights.ndim], (
                "target_weights and target have mismatched shapes " f"{target_weights.shape} v.s. {target.shape}"
            )

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape + (1,) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # Mask by ``skip_empty_channel``
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1,) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask


@MODELS.register_module()
class CombinedTargetMSELoss(nn.Module):
    """MSE loss for combined target.

    CombinedTarget: The combination of classification target
    (response map) and regression target (offset map).
    Paper ref: Huang et al. The Devil is in the Details: Delving into
    Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self, use_target_weight: bool = False, loss_weight: float = 1.0):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output: Tensor, target: Tensor, target_weights: Tensor) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_channels: C
            - heatmaps height: H
            - heatmaps weight: W
            - num_keypoints: K
            Here, C = 3 * K

        Args:
            output (Tensor): The output feature maps with shape [B, C, H, W].
            target (Tensor): The target feature maps with shape [B, C, H, W].
            target_weights (Tensor): The target weights of differet keypoints,
                with shape [B, K].

        Returns:
            Tensor: The calculated loss.
        """
        batch_size = output.size(0)
        num_channels = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_channels, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_channels, -1)).split(1, 1)
        loss = 0.0
        num_joints = num_channels // 3
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx * 3].squeeze()
            heatmap_gt = heatmaps_gt[idx * 3].squeeze()
            offset_x_pred = heatmaps_pred[idx * 3 + 1].squeeze()
            offset_x_gt = heatmaps_gt[idx * 3 + 1].squeeze()
            offset_y_pred = heatmaps_pred[idx * 3 + 2].squeeze()
            offset_y_gt = heatmaps_gt[idx * 3 + 2].squeeze()
            if self.use_target_weight:
                target_weight = target_weights[:, idx, None]
                heatmap_pred = heatmap_pred * target_weight
                heatmap_gt = heatmap_gt * target_weight
            # classification loss
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
            # regression loss
            loss += 0.5 * self.criterion(heatmap_gt * offset_x_pred, heatmap_gt * offset_x_gt)
            loss += 0.5 * self.criterion(heatmap_gt * offset_y_pred, heatmap_gt * offset_y_gt)
        return loss / num_joints * self.loss_weight


@MODELS.register_module()
class KeypointOHKMMSELoss(nn.Module):
    """MSE loss with online hard keypoint mining.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        topk (int): Only top k joint losses are kept. Defaults to 8
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self, use_target_weight: bool = False, topk: int = 8, loss_weight: float = 1.0):
        super().__init__()
        assert topk > 0
        self.criterion = nn.MSELoss(reduction="none")
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.loss_weight = loss_weight

    def _ohkm(self, losses: Tensor) -> Tensor:
        """Online hard keypoint mining.

        Note:
            - batch_size: B
            - num_keypoints: K

        Args:
            loss (Tensor): The losses with shape [B, K]

        Returns:
            Tensor: The calculated loss.
        """
        ohkm_loss = 0.0
        B = losses.shape[0]
        for i in range(B):
            sub_loss = losses[i]
            _, topk_idx = torch.topk(sub_loss, k=self.topk, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= B
        return ohkm_loss

    def forward(self, output: Tensor, target: Tensor, target_weights: Tensor) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W].
            target (Tensor): The target heatmaps with shape [B, K, H, W].
            target_weights (Tensor): The target weights of differet keypoints,
                with shape [B, K].

        Returns:
            Tensor: The calculated loss.
        """
        num_keypoints = output.size(1)
        if num_keypoints < self.topk:
            raise ValueError(f"topk ({self.topk}) should not be " f"larger than num_keypoints ({num_keypoints}).")

        losses = []
        for idx in range(num_keypoints):
            if self.use_target_weight:
                target_weight = target_weights[:, idx, None, None]
                losses.append(self.criterion(output[:, idx] * target_weight, target[:, idx] * target_weight))
            else:
                losses.append(self.criterion(output[:, idx], target[:, idx]))

        losses = [loss.mean(dim=(1, 2)).unsqueeze(dim=1) for loss in losses]
        losses = torch.cat(losses, dim=1)

        return self._ohkm(losses) * self.loss_weight


@MODELS.register_module()
class AdaptiveWingLoss(nn.Module):
    """Adaptive wing loss. paper ref: 'Adaptive Wing Loss for Robust Face
    Alignment via Heatmap Regression' Wang et al. ICCV'2019.

    Args:
        alpha (float), omega (float), epsilon (float), theta (float)
            are hyper-parameters.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5, use_target_weight=False, loss_weight=1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def criterion(self, pred, target):
        """Criterion of wingloss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            pred (torch.Tensor[NxKxHxW]): Predicted heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
        """
        H, W = pred.shape[2:4]
        delta = (target - pred).abs()

        A = (
            self.omega
            * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target)))
            * (self.alpha - target)
            * (torch.pow(self.theta / self.epsilon, self.alpha - target - 1))
            * (1 / self.epsilon)
        )
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - target))

        losses = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C,
        )

        return torch.mean(losses)

    def forward(self, output: Tensor, target: Tensor, target_weights: Optional[Tensor] = None):
        """Forward function.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, H, W]): Output heatmaps.
            target (torch.Tensor[N, K, H, W]): Target heatmaps.
            target_weight (torch.Tensor[N, K]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert target_weights.ndim in (2, 4) and target_weights.shape == target.shape[: target_weights.ndim], (
                "target_weights and target have mismatched shapes " f"{target_weights.shape} v.s. {target.shape}"
            )

            ndim_pad = target.ndim - target_weights.ndim
            target_weights = target_weights.view(target_weights.shape + (1,) * ndim_pad)
            loss = self.criterion(output * target_weights, target * target_weights)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@MODELS.register_module()
class FocalHeatmapLoss(KeypointMSELoss):
    """A class for calculating the modified focal loss for heatmap prediction.

    This loss function is exactly the same as the one used in CornerNet. It
    runs faster and costs a little bit more memory.

    `CornerNet: Detecting Objects as Paired Keypoints
    arXiv: <https://arxiv.org/abs/1808.01244>`_.

    Arguments:
        alpha (int): The alpha parameter in the focal loss equation.
        beta (int): The beta parameter in the focal loss equation.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(
        self,
        alpha: int = 2,
        beta: int = 4,
        use_target_weight: bool = False,
        skip_empty_channel: bool = False,
        loss_weight: float = 1.0,
    ):
        super(FocalHeatmapLoss, self).__init__(use_target_weight, skip_empty_channel, loss_weight)
        self.alpha = alpha
        self.beta = beta

    def forward(
        self, output: Tensor, target: Tensor, target_weights: Optional[Tensor] = None, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate the modified focal loss for heatmap prediction.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """
        _mask = self._get_mask(target, target_weights, mask)

        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        if _mask is not None:
            pos_inds = pos_inds * _mask
            neg_inds = neg_inds * _mask

        neg_weights = torch.pow(1 - target, self.beta)

        pos_loss = torch.log(output) * torch.pow(1 - output, self.alpha) * pos_inds
        neg_loss = torch.log(1 - output) * torch.pow(output, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = -neg_loss.sum()
        else:
            loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss * self.loss_weight


@MODELS.register_module()
class MLECCLoss(nn.Module):
    """Maximum Likelihood Estimation loss for Coordinate Classification.

    This loss function is designed to work with coordinate classification
    problems where the likelihood of each target coordinate is maximized.

    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.
        mode (str): Specifies the mode of calculating loss:
            'linear' | 'square' | 'log'. Default: 'log'.
        use_target_weight (bool): If True, uses weighted loss. Different
            joint types may have different target weights. Defaults to False.
        loss_weight (float): Weight of the loss. Defaults to 1.0.

    Raises:
        AssertionError: If the `reduction` or `mode` arguments are not in the
                        expected choices.
        NotImplementedError: If the selected mode is not implemented.
    """

    def __init__(
        self, reduction: str = "mean", mode: str = "log", use_target_weight: bool = False, loss_weight: float = 1.0
    ):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), (
            f"`reduction` should be either 'mean', 'sum', or 'none', " f"but got {reduction}"
        )
        assert mode in ("linear", "square", "log"), (
            f"`mode` should be either 'linear', 'square', or 'log', " f"but got {mode}"
        )

        self.reduction = reduction
        self.mode = mode
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, outputs, targets, target_weight=None):
        """Forward pass for the MLECCLoss.

        Args:
            outputs (torch.Tensor): The predicted outputs.
            targets (torch.Tensor): The ground truth targets.
            target_weight (torch.Tensor, optional): Optional tensor of weights
                for each target.

        Returns:
            torch.Tensor: Calculated loss based on the specified mode and
                reduction.
        """

        assert len(outputs) == len(targets), "Outputs and targets must have the same length"

        prob = 1.0
        for o, t in zip(outputs, targets):
            prob *= (o * t).sum(dim=-1)

        if self.mode == "linear":
            loss = 1.0 - prob
        elif self.mode == "square":
            loss = 1.0 - prob.pow(2)
        elif self.mode == "log":
            loss = -torch.log(prob + 1e-4)

        loss[torch.isnan(loss)] = 0.0

        if self.use_target_weight:
            assert target_weight is not None
            for i in range(loss.ndim - target_weight.ndim):
                target_weight = target_weight.unsqueeze(-1)
            loss = loss * target_weight

        if self.reduction == "sum":
            loss = loss.flatten(1).sum(dim=1)
        elif self.reduction == "mean":
            loss = loss.flatten(1).mean(dim=1)

        return loss * self.loss_weight


@MODELS.register_module()
class OKSHeatmapLoss(nn.Module):
    """Loss that maximizes expected Object Keypoint Similarity (OKS) score.

    This loss computes the expected OKS by multiplying probability maps with OKS
    maps and maximizes the OKS score by minimizing (1-OKS). The approach was
    introduced in "ProbPose: A Probabilistic Approach to 2D Human Pose Estimation"
    by Purkrabek et al. in 2025.
    See https://arxiv.org/abs/2412.02254 for more details.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(
        self,
        use_target_weight: bool = False,
        skip_empty_channel: bool = False,
        smoothing_weight: float = 0.2,
        gaussian_weight: float = 0.0,
        loss_weight: float = 1.0,
        oks_type: str = "minus",
    ):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight
        self.smoothing_weight = smoothing_weight
        self.gaussian_weight = gaussian_weight
        self.oks_type = oks_type.lower()

        assert self.oks_type in ["minus", "plus", "both"]

    def forward(
        self,
        output: Tensor,
        target: Tensor,
        target_weights: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        per_pixel: bool = False,
        per_keypoint: bool = False,
    ) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """

        assert target.max() <= 1, "target should be normalized"
        assert target.min() >= 0, "target should be normalized"

        B, K, H, W = output.shape

        _mask = self._get_mask(target, target_weights, mask)

        oks_minus = output * (1 - target)
        oks_plus = (1 - output) * (target)
        if self.oks_type == "both":
            oks = (oks_minus + oks_plus) / 2
        elif self.oks_type == "minus":
            oks = oks_minus
        elif self.oks_type == "plus":
            oks = oks_plus
        else:
            raise ValueError(f"oks_type {self.oks_type} not recognized")

        mse = F.mse_loss(output, target, reduction="none")

        # Smoothness loss
        sobel_x = (
            torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(output.device)
        )
        sobel_y = (
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(output.device)
        )
        gradient_x = F.conv2d(output.reshape(B * K, 1, H, W), sobel_x, padding="same")
        gradient_y = F.conv2d(output.reshape(B * K, 1, H, W), sobel_y, padding="same")
        gradient = (gradient_x**2 + gradient_y**2).reshape(B, K, H, W)

        if _mask is not None:
            oks = oks * _mask
            mse = mse * _mask
            gradient = gradient * _mask

        oks_minus_weight = 1 - self.smoothing_weight - self.gaussian_weight

        if per_pixel:
            loss = self.smoothing_weight * gradient + oks_minus_weight * oks + self.gaussian_weight * mse
        elif per_keypoint:
            max_gradient, _ = gradient.reshape((B, K, H * W)).max(dim=-1)
            loss = (
                oks_minus_weight * oks.sum(dim=(2, 3))
                + self.smoothing_weight * max_gradient
                + self.gaussian_weight * mse.mean(dim=(2, 3))
            )
        else:
            max_gradient, _ = gradient.reshape((B, K, H * W)).max(dim=-1)
            loss = (
                oks_minus_weight * oks.sum(dim=(2, 3))
                + self.smoothing_weight * max_gradient
                + self.gaussian_weight * mse.mean(dim=(2, 3))
            ).mean()

        return loss * self.loss_weight

    def _get_mask(self, target: Tensor, target_weights: Optional[Tensor], mask: Optional[Tensor]) -> Optional[Tensor]:
        """Generate the heatmap mask w.r.t. the given mask, target weight and
        `skip_empty_channel` setting.

        Returns:
            Tensor: The mask in shape (B, K, *) or ``None`` if no mask is
            needed.
        """
        # Given spatial mask
        if mask is not None:
            # check mask has matching type with target
            assert mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1 for d_m, d_t in zip(mask.shape, target.shape)
            ), (f"mask and target have mismatched shapes {mask.shape} v.s." f"{target.shape}")

        # Mask by target weights (keypoint-wise mask)
        if target_weights is not None:
            # check target weight has matching shape with target
            assert target_weights.ndim in (2, 4) and target_weights.shape == target.shape[: target_weights.ndim], (
                "target_weights and target have mismatched shapes " f"{target_weights.shape} v.s. {target.shape}"
            )

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape + (1,) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # Mask by ``skip_empty_channel``
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1,) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask


@MODELS.register_module()
class CalibrationLoss(nn.Module):
    """Calibration loss for heatmap-based pose estimation.

    This loss function evaluates the calibration of predicted probmaps by comparing
    the predicted confidence scores with the actual localization accuracy. It helps
    ensure that the model's confidence predictions correlate well with its actual
    performance.

    Args:
        use_target_weight (bool): Option to use weighted loss calculation.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
        ignore_bottom_percentile (float): The percentile threshold below which
            predictions are ignored in the calibration calculation. This helps
            focus the calibration on more confident predictions. Value should be
            between 0 and 1. Defaults to 0.7
    """

    def __init__(
        self,
        use_target_weight: bool = False,
        skip_empty_channel: bool = False,
        loss_weight: float = 1.0,
        ignore_bottom_percentile: float = 0.7,
    ):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight
        self.ignore_bottom_percentile = ignore_bottom_percentile

    def forward(
        self,
        output: Tensor,
        target: Tensor,
        target_weights: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        per_pixel: bool = False,
        per_keypoint: bool = False,
    ) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """

        assert target.max() <= 1, "target should be normalized"
        assert target.min() >= 0, "target should be normalized"

        B, K, H, W = output.shape

        _mask = self._get_mask(target, target_weights, mask)

        pred_probs = output * target
        pred_probs_sum = pred_probs.sum(dim=(2, 3))

        if per_pixel:
            cross_entropy = -torch.log(pred_probs + 1e-10)
            loss = cross_entropy * _mask
        elif per_keypoint:
            cross_entropy = -torch.log(pred_probs_sum + 1e-10)
            loss = cross_entropy * _mask
        else:
            cross_entropy = -torch.log(pred_probs_sum + 1e-10)
            loss = cross_entropy * _mask
            loss = loss.mean()

        return loss * self.loss_weight

    def _get_mask(self, target: Tensor, target_weights: Optional[Tensor], mask: Optional[Tensor]) -> Optional[Tensor]:
        """Generate the heatmap mask w.r.t. the given mask, target weight and
        `skip_empty_channel` setting.

        Returns:
            Tensor: The mask in shape (B, K, *) or ``None`` if no mask is
            needed.
        """
        # Given spatial mask
        if mask is not None:
            # check mask has matching type with target
            assert mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1 for d_m, d_t in zip(mask.shape, target.shape)
            ), (f"mask and target have mismatched shapes {mask.shape} v.s." f"{target.shape}")

        # Mask by target weights (keypoint-wise mask)
        if target_weights is not None:
            # check target weight has matching shape with target
            assert target_weights.ndim in (2, 4) and target_weights.shape == target.shape[: target_weights.ndim], (
                "target_weights and target have mismatched shapes " f"{target_weights.shape} v.s. {target.shape}"
            )

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape + (1,) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # Mask by ``skip_empty_channel``
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1,) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask
