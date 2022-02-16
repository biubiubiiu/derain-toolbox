exp_name = 'cgan_gc64dc48'
work_dir = f'./work_dirs/{exp_name}'

# model settings
model = dict(
    type='IDCGAN',
    generator=dict(
        type='IDGenerator',
        in_channels=3,
        out_channels=3,
        mid_channels=64
    ),
    discriminator=dict(
        type='IDDiscriminator',
        in_channels=6,
        out_channels=1,
        mid_channels=64
    ),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=6.6e-3),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        vgg_type='vgg16',
        pretrained='torchvision://vgg16',
        layer_weights={'8': 1.0},
        perceptual_weight=1.0,
        use_input_norm=False,
        norm_img=False,
        criterion='mse'
    ),
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
    dict(type='Resize', keys=['lq', 'gt'], scale=(286, 286)),
    dict(type='RandomCrop', keys=['lq', 'gt'], crop_size=(256, 256)),
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
    samples_per_gpu=7,
    workers_per_gpu=7,
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
optimizers = dict(
    generator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)),
    discriminator=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999))
)

# learning policy
runner = dict(type='EpochBasedRunner', max_epochs=2000)
lr_config = dict(policy='Fixed', by_epoch=False)

# checkpoint saving
checkpoint_config = dict(interval=400, save_optimizer=True, by_epoch=True)
evaluation = dict(interval=400, save_image=True, by_epoch=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
