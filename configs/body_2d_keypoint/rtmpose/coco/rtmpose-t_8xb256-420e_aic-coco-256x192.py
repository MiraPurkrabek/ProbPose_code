_base_ = ["../../../_base_/default_runtime.py"]

# runtime
max_epochs = 420
stage2_num_epochs = 30
base_lr = 4e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.0),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

# learning rate
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        # use cosine lr from 210 to 420 epoch
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(
    type="SimCCLabel", input_size=(192, 256), sigma=(4.9, 5.66), simcc_split_ratio=2.0, normalize=False, use_dark=False
)

# model settings
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True
    ),
    backbone=dict(
        _scope_="mmdet",
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="SiLU"),
        init_cfg=dict(
            type="Pretrained",
            prefix="backbone.",
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/"
            "rtmposev1/cspnext-tiny_udp-aic-coco_210e-256x192-cbed682d_20230130.pth",  # noqa
        ),
    ),
    head=dict(
        type="RTMCCHead",
        in_channels=384,
        out_channels=17,
        input_size=codec["input_size"],
        in_featuremap_size=tuple([s // 32 for s in codec["input_size"]]),
        simcc_split_ratio=codec["simcc_split_ratio"],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn="SiLU",
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss=dict(type="KLDiscretLoss", use_target_weight=True, beta=10.0, label_softmax=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
    ),
)

# base dataset settings
dataset_type = "CocoDataset"
data_mode = "topdown"
data_root = "data/"

backend_args = dict(backend="local")
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         f'{data_root}': 's3://openmmlab/datasets/',
#         f'{data_root}': 's3://openmmlab/datasets/'
#     }))

# pipelines
train_pipeline = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(type="RandomBBoxTransform", scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="mmdet.YOLOXHSVRandomAug"),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="Blur", p=0.1),
            dict(type="MedianBlur", p=0.1),
            dict(
                type="CoarseDropout",
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0,
            ),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]

train_pipeline_stage2 = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(type="RandomBBoxTransform", shift_factor=0.0, scale_factor=[0.75, 1.25], rotate_factor=60),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="mmdet.YOLOXHSVRandomAug"),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="Blur", p=0.1),
            dict(type="MedianBlur", p=0.1),
            dict(
                type="CoarseDropout",
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5,
            ),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

# train datasets
dataset_coco = dict(
    type="RepeatDataset",
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="coco/annotations/person_keypoints_train2017.json",
        data_prefix=dict(img="detection/coco/train2017/"),
        pipeline=[],
    ),
    times=3,
)

dataset_aic = dict(
    type="AicDataset",
    data_root=data_root,
    data_mode=data_mode,
    ann_file="aic/annotations/aic_train.json",
    data_prefix=dict(img="pose/ai_challenge/ai_challenger_keypoint" "_train_20170902/keypoint_train_images_20170902/"),
    pipeline=[
        dict(
            type="KeypointConverter",
            num_keypoints=17,
            mapping=[
                (0, 6),
                (1, 8),
                (2, 10),
                (3, 5),
                (4, 7),
                (5, 9),
                (6, 12),
                (7, 14),
                (8, 16),
                (9, 11),
                (10, 13),
                (11, 15),
            ],
        )
    ],
)

# data loaders
train_dataloader = dict(
    batch_size=256,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="CombinedDataset",
        metainfo=dict(from_file="configs/_base_/datasets/coco.py"),
        datasets=[dataset_coco, dataset_aic],
        pipeline=train_pipeline,
        test_mode=False,
    ),
)
val_dataloader = dict(
    batch_size=64,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="coco/annotations/person_keypoints_val2017.json",
        # bbox_file='data/coco/person_detection_results/'
        # 'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img="detection/coco/val2017/"),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

# hooks
default_hooks = dict(checkpoint=dict(save_best="coco/AP", rule="greater", max_keep_ckpts=1))

custom_hooks = [
    # Turn off EMA while training the tiny model
    # dict(
    #     type='EMAHook',
    #     ema_type='ExpMomentumEMA',
    #     momentum=0.0002,
    #     update_buffers=True,
    #     priority=49),
    dict(
        type="mmdet.PipelineSwitchHook",
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2,
    )
]

# evaluators
val_evaluator = dict(type="CocoMetric", ann_file=data_root + "coco/annotations/person_keypoints_val2017.json")
test_evaluator = val_evaluator
