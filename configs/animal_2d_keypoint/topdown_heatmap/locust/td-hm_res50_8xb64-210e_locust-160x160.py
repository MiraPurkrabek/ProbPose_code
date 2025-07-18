_base_ = ["../../../_base_/default_runtime.py"]

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type="Adam",
        lr=5e-4,
    )
)

# learning policy
param_scheduler = [
    dict(type="LinearLR", begin=0, end=500, start_factor=0.001, by_epoch=False),  # warm-up
    dict(type="MultiStepLR", begin=0, end=210, milestones=[170, 200], gamma=0.1, by_epoch=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best="AUC", rule="greater"))

# codec settings
codec = dict(type="MSRAHeatmap", input_size=(160, 160), heatmap_size=(40, 40), sigma=2)

# model settings
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    head=dict(
        type="HeatmapHead",
        in_channels=2048,
        out_channels=35,
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode="heatmap",
        shift_heatmap=True,
    ),
)

# base dataset settings
dataset_type = "LocustDataset"
data_mode = "topdown"
data_root = "data/locust/"

# pipelines
train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale", padding=0.8),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomBBoxTransform", shift_factor=0.25, rotate_factor=180, scale_factor=(0.7, 1.3)),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale", padding=0.8),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annotations/locust_train.json",
        data_prefix=dict(img="images/"),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annotations/locust_test.json",
        data_prefix=dict(img="images/"),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(type="PCKAccuracy", thr=0.2),
    dict(type="AUC"),
    dict(type="EPE"),
]
test_evaluator = val_evaluator
