_base_ = ["../../../rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py"]  # noqa: E501

# model settings
find_unused_parameters = True

# dis settings
second_dis = True

# config settings
logit = True

train_cfg = dict(max_epochs=60, val_interval=10)

# method details
model = dict(
    _delete_=True,
    type="DWPoseDistiller",
    two_dis=second_dis,
    teacher_pretrained="work_dirs/" "dwpose_x_dis_l_coco-ubody-384x288/dw-x-l_ucoco_384.pth",  # noqa: E501
    teacher_cfg="configs/wholebody_2d_keypoint/rtmpose/ubody/"
    "rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py",  # noqa: E501
    student_cfg="configs/wholebody_2d_keypoint/rtmpose/ubody/"
    "rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py",  # noqa: E501
    distill_cfg=[
        dict(
            methods=[
                dict(
                    type="KDLoss",
                    name="loss_logit",
                    use_this=logit,
                    weight=1,
                )
            ]
        ),
    ],
    data_preprocessor=dict(
        type="PoseDataPreprocessor", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True
    ),
    train_cfg=train_cfg,
)

optim_wrapper = dict(clip_grad=dict(max_norm=1.0, norm_type=2))
