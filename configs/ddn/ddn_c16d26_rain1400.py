# An example usage of `DerainMultipleLQDataset` for Rain1400 training

exp_name = 'ddn_c16d26_rain1400'
work_dir = f'./work_dirs/{exp_name}'

# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='DDN',
        in_channels=3,
        out_channels=3,
        mid_channels=16,
        num_blocks=24,
    ),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean')
)

# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'])

# dataset settings
train_dataset_type = 'DerainMultipleLQDataset'
val_dataset_type = 'DerainMultipleLQDataset'
train_pipeline = [
    dict(type='LoadImageFromFile', key='lq'),
    dict(type='LoadImageFromFile', key='gt'),
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
    dict(type='LoadImageFromFile', key='lq'),
    dict(type='LoadImageFromFile', key='gt'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path']
    )
]
common_data_setting = dict(
    dataroot='../data/Rain1400',
    lq_folder='rainy_image',
    gt_folder='ground_truth',
    mapping_rule='prefix',
    separator='_'
)
data = dict(
    samples_per_gpu=20,
    workers_per_gpu=8,
    drop_last=True,
    val_samples_per_gpu=1,
    train=dict(
        type=train_dataset_type,
        pipeline=train_pipeline,
        test_mode=False,
        **common_data_setting
    ),
    val=dict(
        type=val_dataset_type,
        pipeline=test_pipeline,
        test_mode=True,
        **common_data_setting
    ),
    test=dict(
        type=val_dataset_type,
        pipeline=test_pipeline,
        test_mode=True,
        **common_data_setting
    )
)

# optimizer
optimizers = dict(generator=dict(type='SGD', lr=1e-1,
                  weight_decay=1e-10, momentum=0.9))

# learning policy
total_iters = 210000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[100000, 200000],
    gamma=0.1
)

checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=5000, save_image=True)
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
