exp_name = 'rescan_c24s4d5_gru_add'
work_dir = f'./work_dirs/{exp_name}'

# model settings
model = dict(
    type='MultiStageRestorer',
    generator=dict(
        type='RESCAN',
        in_channels=3,
        out_channels=3,
        mid_channels=24,
        num_stages=4,
        depth=5,
        recurrent_unit='GRU',
        prediction_type='Additive'
    ),
    losses=[dict(type='MSELoss', loss_weight=1.0, reduction='mean')],
    recurrent_loss=True
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
    dict(type='FixedCrop', keys=['lq', 'gt'], crop_size=(64, 64)),
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
    samples_per_gpu=64,
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
optimizers = dict(generator=dict(type='Adam', lr=5e-3, betas=(0.9, 0.999)))

# learning policy
total_iters = 20000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[15000, 17500],
    gamma=0.1
)

checkpoint_config = dict(interval=800, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=1000, save_image=True, by_epoch=False)
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
