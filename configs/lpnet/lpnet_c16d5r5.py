exp_name = 'lpnet_c16d5r5'
work_dir = f'./work_dirs/{exp_name}'

# model settings
model = dict(
    type='LPNet',
    generator=dict(
        type='LPNet',
        in_channels=3,
        out_channels=3,
        mid_channels=16,
        max_level=5,
        n_recursion=5
    ),
    losses=[
        dict(type='L1Loss', loss_weight=1.0, reduction='mean', levels=[0, 1, 2, 3, 4]),
        dict(type='SSIMLoss', loss_weight=1.0, reduction='mean', levels=[3, 4])
    ]
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
    dict(type='RandomCrop', keys=['lq', 'gt'], crop_size=(80, 80)),
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
    train_dataloader=dict(samples_per_gpu=10, drop_last=True),
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
optimizers = dict(type='Adam', lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
optimizer_config = dict(grad_clip=None)

# learning policy
total_iters = 300000
lr_config = None

checkpoint_config = dict(interval=50000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=50000, save_image=True, by_epoch=False)
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
