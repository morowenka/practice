# model settings
classes = ('bus','car','cyclist','jeep',
           'misc','pedestrian','truck','van')
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SEResNet',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/se-resnet/se-resnet101_batch256_imagenet_20200804-ba5b51d4.pth',
            prefix='backbone'),
        frozen_stages=3),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=len(classes),
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ))

# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(mean=[124.508, 116.050, 106.438], std=[58.577, 57.310, 57.437], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_prefix='/content/drive/MyDrive/ColabNotebooks/Practice/datasets/ClsVehicleDataset/train',
        ann_file='/content/drive/MyDrive/ColabNotebooks/Practice/datasets/ClsVehicleDataset/train/_annotations.txt',
        classes='/content/drive/MyDrive/ColabNotebooks/Practice/datasets/ClsVehicleDataset/train/_classes.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/content/drive/MyDrive/ColabNotebooks/Practice/datasets/ClsVehicleDataset/valid',
        ann_file='/content/drive/MyDrive/ColabNotebooks/Practice/datasets/ClsVehicleDataset/valid/_annotations.txt',
        classes='/content/drive/MyDrive/ColabNotebooks/Practice/datasets/ClsVehicleDataset/valid/_classes.txt',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='/content/drive/MyDrive/ColabNotebooks/Practice/datasets/ClsVehicleDataset/test',
        ann_file='/content/drive/MyDrive/ColabNotebooks/Practice/datasets/ClsVehicleDataset/test/_annotations.txt',
        classes='/content/drive/MyDrive/ColabNotebooks/Practice/datasets/ClsVehicleDataset/test/_classes.txt',
        pipeline=test_pipeline))
evaluation = dict(
    interval=1,
    save_best='auto', # 'accuracy_top-1'
    metric=('accuracy', 'precision', 'recall'))

# optimizer
optimizer = dict(
    type='Adam',
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0005,
    amsgrad=True)
optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
    type="GradientCumulativeOptimizerHook",
    cumulative_iters=4)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='exp',
    warmup_iters=2,
    warmup_ratio=0.15,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=20, meta=dict())
work_dir = '/content/drive/MyDrive/ColabNotebooks/Practice/logs/classifiers/seresnet'
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

seed = 0
gpu_ids = range(1)