exp_name = 'rlnet_stage2'
work_dir = f'./work_dirs/{exp_name}'

# model settings
model = dict(
    type='RLNet',
    model_cfg=dict(
        type='RLNet',
        in_channels=3,
        out_channels=3,
        mid_channels=[24, 32, 18],
        theta=[0.15, 0.05],
        n_scale=4,
        n_residual=7
    ),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    structural_loss=dict(type='SSIMLoss', loss_weight=1.0, reduction='mean'),
    lambdas=[0.01, 0.6, 0.6, 0.6]
)

# model training and testing settings
train_cfg = dict(joint_training=True)
test_cfg = dict(metrics=['PSNR', 'SSIM'])

# dataset settings
train_dataset_type = 'DerainPairedDataset'
val_dataset_type = 'DerainPairedDataset'
train_pipeline = [
    dict(type='LoadPairedImageFromFile', key='gt,lq', channel_order='rgb'),
    dict(type='Resize', keys=['lq', 'gt'], scale=(512, 512), interpolation='nearest'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]
test_pipeline = [
    dict(type='LoadPairedImageFromFile', key='gt,lq', channel_order='rgb'),
    dict(type='Pad', keys=['lq'], ds_factor=32, mode='reflect'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path', 'pad', 'lq_ori_shape'])
]
data_root = '../data/Rain200L'
data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),
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
optimizers = dict(type='Adam', lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
optimizer_config = dict(grad_clip=None)

# learning policy
runner = dict(type='EpochBasedRunner', max_epochs=240)
lr_config = dict(
    policy='Step',
    by_epoch=True,
    step=list(range(0, 240+1, 30))[1:],
    gamma=0.5
)

# hooks
custom_hooks = [
    dict(type='RLNetHyperParamAdjustmentHook', mode='paper'),
]

checkpoint_config = dict(interval=30, save_optimizer=False, by_epoch=True)
evaluation = dict(interval=30, save_image=True, by_epoch=True)
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
load_from = './work_dirs/rlnet_stage1/latest.pth'
resume_from = None
workflow = [('train', 1)]
