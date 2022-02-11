exp_name = 'oucdnet_o3n5_msff'
work_dir = f'./work_dirs/{exp_name}'

# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='OUCDNet',
        out_channels=3,
        enc_undercomplete_channels=[3, 32, 64, 128, 512, 1024],
        dec_undercomplete_channels=[1024, 512, 128, 64, 32, 16],
        enc_overcomplete_channels=[3, 32, 64, 128],
        dec_overcomplete_channels=[128, 64, 32, 16],
        use_msff=True
    ),
    losses=[
        dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        dict(
            type='PerceptualLoss',
            vgg_type='vgg16',
            pretrained='torchvision://vgg16',
            layer_weights={'3': 1./3, '8': 1./3, '15': 1./3},  # average loss
            perceptual_weight=0.04,
            use_input_norm=False,
            norm_img=False,
            criterion='mse'
        ),
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
    dict(type='Normalize', keys=['lq'], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
    dict(type='Pad', keys=['lq'], ds_factor=32),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq'], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'pad', 'lq_ori_shape']
    )
]
data_root = '../data/Rain200L'
data = dict(
    samples_per_gpu=2,
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
optimizers = dict(type='Adam', lr=2e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)

# learning policy
runner = dict(type='EpochBasedRunner', max_epochs=200)
lr_config = dict(
    policy='Step',
    by_epoch=True,
    step=[100],
    gamma=0.5
)

checkpoint_config = dict(interval=20, save_optimizer=True, by_epoch=True)
evaluation = None
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
