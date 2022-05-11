classes = ('bus','car','cyclist','jeep',
           'misc','pedestrian','truck','van')

# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ConvNeXt',
        arch='xlarge',
        out_indices=(3, ),
        drop_path_rate=0.5,
        gap_before_final_norm=True,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
        ]),
    head=dict(
        type='LinearClsHead',
        num_classes=len(classes),
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    })

# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optimizer = dict(
    type='Adam',
    lr=0.001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='exp',
    warmup_iters=2,
    warmup_ratio=0.15,
    warmup_by_epoch=True)

runner = dict(type='EpochBasedRunner', max_epochs=30)

# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(256, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=2,
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


custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

# checkpoint saving
work_dir = '/content/drive/MyDrive/ColabNotebooks/Practice/logs/classifiers/seresnet'
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-xlarge_in21k-pre-3rdparty_64xb64_in1k_20220124-76b6863d.pth'
resume_from = None
workflow = [('train', 1)]

seed = 0
gpu_ids = range(1)