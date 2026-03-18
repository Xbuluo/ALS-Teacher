_base_ = "base_oriented_rcnn_default.py"

data_root = 'data/CODrone/'
classes = ('bicycle','boat','bridge','bus','car','motor','people',
'ship','traffic-light','traffic-sign','tricycle','truck')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        sup=dict(
            ann_file='path/to/your/10per/annotation/',
            img_prefix='path/to/your/10per/images/',
            classes=classes,
        ),
        unsup=dict(
            ann_file='path/to/your/unsup/annotation/',
            img_prefix='path/to/your/unsup/images/',
            classes=classes,
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[2, 2],
        )
    ),
)

model = dict(
    semi_loss=dict(type='RotatedSoftEMDLoss', distance_type="sinkhorn"),
    train_cfg=dict(
        iter_count=0,
        burn_in_steps=16000,
        logit_specific_weights=dict(loss_bbox_unsup=0, loss_score_unsup=0),
    )
)

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(
        #     type="WandbLoggerHook",
        #     init_kwargs=dict(
        #         project="rotated_DenseTeacher_10percent",
        #         name="default_bce4cls",
        #     ),
        #     by_epoch=False,
        # ),
    ],
)
