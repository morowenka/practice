
classes = ('bus','car','cyclist','jeep',
           'misc','pedestrian','truck','van')

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        groups=32,
        width_per_group=8,
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=len(classes),
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='exp',
    warmup_iters=1,
    warmup_ratio=0.4,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=100)

work_dir = '/content/drive/MyDrive/ColabNotebooks/Practice/logs/classifiers/resnext'
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x8d_b32x8_imagenet_20210506-23a247d5.pth'
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

seed = 0
gpu_ids = range(1)