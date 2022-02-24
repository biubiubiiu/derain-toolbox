exp_name = 'rehen_c24s4n5'
work_dir = f'./work_dirs/{exp_name}'

# model settings
model = dict(
    type='MultiStageRestorer',
    generator=dict(
        type='ReHEN',
        in_out_channels=3,
        mid_channels=24,
        n_stages=4,
        n_blocks=5
    ),
    losses=[
        dict(type='MSELoss', loss_weight=1.0, reduction='mean', recurrent=True),
        dict(type='PSNRLoss', loss_weight=0.1, reduction='mean', reciprocal=True, recurrent=False),
        dict(type='SSIMLoss', loss_weight=1e-3, reduction='mean', recurrent=False)
    ],
    init_cfg=[
        dict(type='Xavier', layer='Conv2d', distribution='uniform', gain=1.0),
        dict(type='Constant', layer='BatchNorm2d', val=1.0, bias=0.0),
    ]
)

# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'])

# dataset settings
train_dataset_type = 'DerainPairedDataset'
val_dataset_type = 'DerainPairedDataset'
train_pipeline = [
    dict(type='LoadPairedImageFromFile', key='gt,lq'),
    dict(type='RandomCrop', keys=['lq', 'gt'], crop_size=(64, 64)),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path']
    )
]
test_pipeline = [
    dict(type='LoadPairedImageFromFile', key='gt,lq'),
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
    train_dataloader=dict(samples_per_gpu=32, drop_last=True),
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
optimizers = dict(type='Adam', lr=5e-3, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)

# learning policy
total_iters = 60000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[50000],
    gamma=0.1
)

checkpoint_config = dict(interval=20000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=20000, save_image=True, by_epoch=False)
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
