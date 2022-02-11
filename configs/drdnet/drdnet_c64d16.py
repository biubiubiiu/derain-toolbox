exp_name = 'drdnet_c64d16'
work_dir = f'./work_dirs/{exp_name}'

# model settings
model = dict(
    type='MultiOutputRestorer',
    generator=dict(
        type='DRDNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=16,
    ),
    losses=[
        dict(type='MSELoss', loss_weight=0.1, reduction='mean', idx=0),
        dict(type='MSELoss', loss_weight=1.0, reduction='mean', idx=1),
    ],
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
    dict(type='RandomCrop', keys=['lq', 'gt'], crop_size=(128, 128)),
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
    samples_per_gpu=4,
    workers_per_gpu=8,
    drop_last=True,
    val_samples_per_gpu=1,
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
optimizers = dict(type='Adam', lr=0.1, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)

# learning policy
iters_per_round = 1000
total_iters = 120 * iters_per_round
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=list(range(0, total_iters+1, 15*iters_per_round))[1:],
    gamma=0.5
)

checkpoint_config = dict(interval=30*iters_per_round, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=30*iters_per_round, save_image=True, by_epoch=False)
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
