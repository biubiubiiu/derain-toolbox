exp_name = 'pmg_c64d5s3'
work_dir = f'./work_dirs/{exp_name}'

# model settings
model = dict(
    type='PhysicalModelGuided',
    model_cfg=dict(
        type='PhysicalModelGuided',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        depth=5,
        scales=[1, 2, 4]
    ),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    loss_weight=(0.5, 0.5, 1.0, 0.01)
)

# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'])

# dataset settings
train_dataset_type = 'DerainPairedDataset'
val_dataset_type = 'DerainPairedDataset'
train_pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        io_backend='disk',
        key='gt,lq',
        flag='color'
    ),
    dict(type='RandomCrop', keys=['lq', 'gt'], crop_size=(128, 128)),  # reduce from 160 to 128
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path']
    )
]
test_pipeline = [
    dict(
        type='LoadPairedImageFromFile',
        io_backend='disk',
        key='gt,lq',
        flag='color'
    ),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path']
    )
]
data_root = '../data/Rain200L'
data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),  # reduce from 32 to 4
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type=train_dataset_type,
        dataroot=data_root,
        pipeline=train_pipeline,
        test_mode=False
    ),
    val=dict(
        type=val_dataset_type,
        dataroot=data_root,
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=val_dataset_type,
        dataroot=data_root,
        pipeline=test_pipeline,
        test_mode=True
    )
)

# optimizer
optimizers = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)

# learning policy
runner = dict(type='EpochBasedRunner', max_epochs=200)  # reduce from 2000 to 200
lr_config = dict(
    policy='Step',
    by_epoch=True,
    step=[120, 160],
    gamma=0.1
)

checkpoint_config = dict(interval=20, save_optimizer=True, by_epoch=True)
evaluation = dict(interval=40, save_image=True, by_epoch=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False)
    ]
)
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
