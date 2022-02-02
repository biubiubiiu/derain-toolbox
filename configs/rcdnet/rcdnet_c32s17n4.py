exp_name = 'rcdnet_c32s17n4'
work_dir = f'./work_dirs/{exp_name}'

# model settings
model = dict(
    type='RCDNet',
    model_cfg=dict(
        type='RCDNet',
        mid_channels=32,
        num_stages=17,
        num_blocks=4,
        init_etaM=1.0,
        init_etaB=5.0
    ),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
    loss_weight=(0.1, 1.0)
)

# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'])

# dataset settings
train_dataset_type = 'DerainPairedDataset'
val_dataset_type = 'DerainPairedDataset'
train_pipeline = [
    dict(type='LoadPairedImageFromFile', key='gt,lq', channel_order='rgb'),
    dict(type='RandomCrop', keys=['lq', 'gt'], crop_size=(64, 64)),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]
test_pipeline = [
    dict(type='LoadPairedImageFromFile', key='gt,lq', channel_order='rgb'),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]
data_root = '../data/Rain200L'
data = dict(
    samples_per_gpu=16,
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
optimizers = dict(type='Adam', lr=1e-3, betas=(0.9, 0.999))

# learning policy
iters_per_round = 1500
total_iters = 100 * iters_per_round
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[x * iters_per_round for x in (25, 50, 75)],
    gamma=0.2
)

checkpoint_config = dict(interval=20*iters_per_round, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=20*iters_per_round, save_image=True, by_epoch=False)
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
