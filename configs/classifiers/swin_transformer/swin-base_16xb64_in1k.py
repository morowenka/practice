# model settings
classes = ('bus','car','cyclist','jeep',
           'misc','pedestrian','truck','van')

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='base',
        img_size=224,
        drop_path_rate=0.5,
        frozen_stages=2),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=len(classes),
        in_channels=1024,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth',
            prefix='backbone'),  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=True),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict())

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

runner = dict(type='EpochBasedRunner', max_epochs=60, meta=dict())
work_dir = '/content/drive/MyDrive/ColabNotebooks/Practice/logs/classifiers/swin_transformer'
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    })

optimizer = dict(
    type='Adam',
    lr=0.0001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.00005,
    amsgrad=True)
#optimizer_config = dict(
#    grad_clip=dict(max_norm=35, norm_type=2),
#    type="GradientCumulativeOptimizerHook",
#    cumulative_iters=4)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='exp',
    warmup_iters=5,
    warmup_ratio=0.1,
    warmup_by_epoch=True)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth'
resume_from = None
workflow = [('train', 1)]

seed = 0
gpu_ids = range(1)