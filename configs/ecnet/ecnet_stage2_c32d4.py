exp_name = 'ecnet_stage2'
work_dir = f'./work_dirs/{exp_name}'

# model settings
model = dict(
    type='ECNet',
    rain_encoder=dict(
        type='RainEncoder',
        in_channels=3,
        out_channels=3,
        mid_channels=32,
        depth=4,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./work_dirs/ecnet_stage1/latest.pth',
            prefix='rain_encoder'
        )
    ),
    derain_net=dict(
        type='ECNet',
        in_out_channels=3,
        mid_channels=32,
        depth=4,
        lcn_window_size=9,
        use_rnn=False,
        init_cfg=[
            dict(type='Normal', layer='Conv2d', mean=0.0, std=0.02),
            dict(type='Normal', layer='BatchNorm2d', mean=1.0, std=0.02),
            dict(type='ECNetTransfer', checkpoint='./work_dirs/ecnet_stage1/latest.pth')
        ]
    ),
    loss_embed=dict(type='L1Loss', loss_weight=0.02, reduction='mean'),
    loss_att=dict(type='MSELoss', loss_weight=0.1, reduction='mean'),
    loss_image=dict(type='SSIMLoss', loss_weight=1.0, reduction='mean'),
)

# model training and testing settings
train_cfg = dict(
    train_ecnet=True,
    mask_threshold=2e-3,
    stage_loss_weights=[1.0]
)
test_cfg = dict(
    test_ecnet=True,
    metrics=['PSNR', 'SSIM']
)

# dataset settings
train_dataset_type = 'DerainPairedDataset'
val_dataset_type = 'DerainPairedDataset'
train_pipeline = [
    dict(type='LoadPairedImageFromFile', key='gt,lq', channel_order='rgb'),
    dict(type='ArgsCrop', keys=['lq', 'gt']),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq', 'gt'], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path']
    )
]
test_pipeline = [
    dict(type='LoadPairedImageFromFile', key='gt,lq', channel_order='rgb'),
    dict(type='Pad', keys=['lq', 'gt'], ds_factor=8, mode='reflect'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        save_original=True
    ),
    dict(type='ImageToTensor', keys=['lq', 'gt', 'lq_unnormalised', 'gt_unnormalised']),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'lq_unnormalised', 'gt_unnormalised'],
        meta_keys=['lq_path', 'gt_path', 'pad', 'lq_ori_shape']
    )
]
data_root = '../data/Rain200L'
data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=16, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='ExhaustivePatchDataset',
        patch_size=96,
        stride=96,
        dataset=dict(
            type=train_dataset_type,
            dataroot=data_root,
            pipeline=train_pipeline,
            test_mode=False
        )
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
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

# learning policy
runner = dict(type='EpochBasedRunner', max_epochs=100)
lr_config = dict(
    policy='Step',
    by_epoch=True,
    step=[25, 50, 75],
    gamma=0.2
)

checkpoint_config = dict(interval=10, save_optimizer=True, by_epoch=True)
evaluation = dict(interval=10, save_image=True, by_epoch=True)
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
cudnn_benchmark=True
