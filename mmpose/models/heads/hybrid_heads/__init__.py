# Copyright (c) OpenMMLab. All rights reserved.
from .dekr_head import DEKRHead
from .rtmo_head import RTMOHead
from .vis_head import VisPredictHead
from .yoloxpose_head import YOLOXPoseHead
from .probmap_head import ProbMapHead
from .DP_head import DoubleProbMapHead

__all__ = ['DEKRHead', 'VisPredictHead', 'YOLOXPoseHead', 'RTMOHead', 'ProbMapHead', 'DoubleHead', 'PoseIDHead',
           'DoubleProbMapHead', 'IterativeHead']
