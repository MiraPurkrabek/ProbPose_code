COCO_ROOT = "PATH/TO/COCO/DATASET/"
CropCOCO_ROOT = "PATH/TO/CropCOCO/DATASET/"

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
INPUT_PADDING = 1.25

COCO_NAME = "COCO"
CropCOCO_NAME = "CropCOCO"

_base_ = ['../../../_base_/default_runtime.py']

load_from = 'models/trained/ViTs-m-norm-epoch_21.pth'

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
custom_imports = dict(
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=TRAIN_BATCH_SIZE/64*5e-5, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.8,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1., norm_type=2),
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    checkpoint=dict(save_best='{}/AP'.format(COCO_NAME), rule='greater', max_keep_ckpts=1))

# codec settings
codec = dict(
    type='ProbMap', input_size=(192, 256), heatmap_size=(48, 64), sigma=-1)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch={
            'embed_dims': 384,
            'num_layers': 12,
            'num_heads': 12,
            'feedforward_channels': 384 * 4
        },
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.1,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=None
    ),
    head=dict(
        type='ProbMapHead',
        in_channels=384,
        out_channels=17,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        keypoint_loss=dict(type='OKSHeatmapLoss', use_target_weight=True, smoothing_weight=0.05),
        probability_loss=dict(type='BCELoss', use_target_weight=True, use_sigmoid=True),
        visibility_loss=dict(type='BCELoss', use_target_weight=True, use_sigmoid=True),
        oks_loss=dict(type='MSELoss', use_target_weight=True),
        error_loss=dict(type='L1LogLoss', use_target_weight=True),
        detach_probability=True,
        detach_visibility=True,
        normalize=1.0,
        freeze_error=True,
        freeze_oks=False,
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=False,
        # output_heatmaps=True,
    ),
    # freeze_backbone=False,
    )

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True, input_padding=INPUT_PADDING),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', pad_to_aspect_ratio=False),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True, input_padding=INPUT_PADDING),
    dict(type='PackPoseInputs')
]

# base dataset settings
data_root = COCO_ROOT
dataset_type = 'CocoDataset'
data_mode = 'topdown'

coco_val = dict(
    type="CocoDataset",
    data_root=COCO_ROOT,
    data_mode="topdown",
    ann_file="annotations/person_keypoints_val2017.json",
    test_mode=True,
    pipeline=[],
    data_prefix=dict(img="val2017/"),
)
coco_train = dict(
    type="CocoDataset",
    data_root=COCO_ROOT,
    data_mode="topdown",
    ann_file="annotations/person_keypoints_train2017.json",
    test_mode=False,
    pipeline=train_pipeline,
    data_prefix=dict(img="train2017/"),
)
CropCOCO_val = dict(
    type="CocoCropDataset",
    data_root=CropCOCO_ROOT,
    data_mode="topdown",
    ann_file="annotations/person_keypoints_val2017.json",
    test_mode=True,
    pipeline=[],
    data_prefix=dict(img="val2017/"),
)
combined_val_dataset = dict(
    type='CombinedDataset',
    metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
    datasets=[CropCOCO_val, coco_val],
    pipeline=val_pipeline,
    test_mode=True,
)

# data loaders
train_dataloader = dict(
    batch_size=TRAIN_BATCH_SIZE,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=coco_train,
)
val_dataloader = dict(
    batch_size=TEST_BATCH_SIZE,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=combined_val_dataset,
)
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='MultiDatasetEvaluator',
    metrics=[
        dict(type='CocoMetric',
            ann_file=CropCOCO_ROOT + 'annotations/person_keypoints_val2017.json',
            prefix=CropCOCO_NAME,
            extended=[False, True],
            match_by_bbox=[False, False],
            ignore_border_points=[False, False],
            padding=INPUT_PADDING,
            score_thresh_type='prob',
            keypoint_score_thr=0.45,
        ),
        dict(type='CocoMetric',
            ann_file=COCO_ROOT + 'annotations/person_keypoints_val2017.json',
            prefix=COCO_NAME,
            extended=[False, True],
            match_by_bbox=[False, False],
            ignore_border_points=[False, False],
            padding=INPUT_PADDING,
            score_thresh_type='prob',
            keypoint_score_thr=0.45,
        )
    ],
    datasets=combined_val_dataset['datasets'],
    )
test_evaluator = val_evaluator

