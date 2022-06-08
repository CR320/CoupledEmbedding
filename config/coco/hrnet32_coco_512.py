# model configuration
model = dict(
    backbone=dict(
        type='HRNet',
        in_channels=3,
        multiscale_output=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        pre_weights='pretrained/hrnet_w32-36af842e.pth',
    ),
    head=dict(
        type='AssociativeEmbeddingHead',
        in_channels=32,
        num_keys=17
    )
)

# trainer configuration
trainer = dict(
    model=model,
    loss_weights=dict(
        hms_loss=1.0,
        pull_loss=1e-3,
        push_loss=1e-3
    )
)

# evaluation configuration
eval_cfg = dict(
    val_th=0.1,
    tag_th=1.0,
    max_num_people=30,
    nms_kernel_size=5,
    with_flip=True,
    with_refine=True
)

# data-set configuration
data_root = '/home/wanghaixin/datasets/coco'
data_cfg = dict(
    image_size=512,
    heatmap_size=[128],
    use_nms=False,
    soft_nms=False,
    oks_thr=0.9,
    num_scales=1
)
train_pipeline = [
    dict(type='LoadImageFromFile', channel_order='bgr'),
    dict(type='RandomAffine',
         rot_factor=30,
         scale_factor=[0.75, 1.5],
         scale_type='short',
         trans_factor=40),
    dict(type='RandomFlip', flip_prob=0.5),
    dict(type='FormatGroundTruth', sigma=2, max_num_people=30),
    dict(type='NormalizeImage',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='Collect',
         keys=['image', 'joints', 'masks', 'target_hms'],
         meta_keys=[])
]
val_pipeline = [
    dict(type='LoadImageFromFile', channel_order='bgr'),
    dict(type='ResizeAlign', size_divisor=64),
    dict(type='NormalizeImage',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='Collect',
         keys=['image'],
         meta_keys=['image_file', 'flip_index', 'inference_channel', 'center', 'scale'])
]
set_cfg = dict(
    samples_per_gpu=10,
    workers_per_gpu=4,
    train=dict(
        type='COCOPose',
        ann_file='{}/annotations/person_keypoints_train2017.json'.format(data_root),
        img_prefix='{}/train2017/'.format(data_root),
        pipeline=train_pipeline),
    val=dict(
        type='COCOPose',
        ann_file='{}/annotations/person_keypoints_val2017.json'.format(data_root),
        img_prefix='{}/val2017/'.format(data_root),
        pipeline=val_pipeline),
    test=dict(
        type='COCOPose',
        ann_file='{}/annotations/image_info_test-dev2017.json'.format(data_root),
        img_prefix='{}/test2017/'.format(data_root),
        pipeline=val_pipeline),
)

# solver
solver = dict(
    resume_from=None,
    optimizer=dict(
        type='Adam',
        lr=1.5e-3
    ),
    lr_scheduler=dict(
        warmup_iters=500,
        warmup_ratio=1e-3,
        milestones=[200, 260],
        gamma=0.1
    ),
    total_epochs=300,
    eval_interval=10,   # epoch
    log_interval=50,   # iter
    log_loss=['hms_loss', 'pull_loss', 'push_loss']
)
