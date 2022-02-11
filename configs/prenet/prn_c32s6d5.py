exp_name = 'prn_c32s6d5'
work_dir = f'./work_dirs/{exp_name}'

# model settings
model = dict(
    type='MultiStageRestorer',
    generator=dict(
        type='PRN',
        in_channels=3,
        out_channels=3,
        mid_channels=32,
        num_stages=6,
        num_resblocks=5,
        recursive_resblock=False
    ),
    losses=[dict(type='SSIMLoss', loss_weight=1.0, reduction='mean')],
    recurrent_loss=False
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
    dict(type='ArgsCrop', keys=['lq', 'gt']),
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
    samples_per_gpu=18,
    workers_per_gpu=8,
    drop_last=True,
    val_samples_per_gpu=1,
    train=dict(
        type='ExhaustivePatchDataset',
        patch_size=100,
        stride=80,  # set to 100 for Rain1200 and Rain1400 dataset
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
optimizer_config = dict(grad_clip=None)

# learning policy
runner = dict(type='EpochBasedRunner', max_epochs=100)
lr_config = dict(
    policy='Step',
    by_epoch=True,
    step=[30, 50, 80],
    gamma=0.2
)

checkpoint_config = dict(interval=10, save_optimizer=True, by_epoch=True)
evaluation = dict(interval=25, save_image=True, by_epoch=True)
log_config = dict(
    interval=400,
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
