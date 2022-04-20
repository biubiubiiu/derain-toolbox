exp_name = 'spdnet_c32s3_l3ns3nr3'
work_dir = f'./work_dirs/{exp_name}'

# model settings
model = dict(
    type='SPDNet',
    model_cfg=dict(
        type='SPDNet',
        in_channels=3,
        out_channels=3,
        mid_channels=32,
        n_stage=3,
        n_level=3,
        n_srir=3,
        n_resblock=3
    ),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
)

# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'])

# dataset settings
train_dataset_type = 'DerainPairedDataset'
val_dataset_type = 'DerainPairedDataset'
train_pipeline = [
    dict(type='LoadPairedImageFromFile', key='gt,lq', channel_order='rgb'),
    dict(type='RandomCrop', keys=['lq', 'gt'], crop_size=(128, 128)),
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
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=16, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        # NOTE: The offical implementation repeated the training set for several times,
        # where `times` is calculated by:
        #
        #       times = max(1, 1000//num_of_batches)
        #
        # For example, since we're using a batch size of 16, we can get that:
        # - For Rain200L and Rain200H (both have 1800 data pairs), repeat times is 8
        # - For Rain800 (700 data pairs), repeat times in 23
        # - For Rain1200 and Rain1400, repeat times is 1
        times=8,
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
optimizers = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)

# learning policy
runner = dict(type='EpochBasedRunner', max_epochs=300)
lr_config = dict(
    policy='Step',
    by_epoch=True,
    step=[100, 150, 200, 230, 260, 280, 300],
    gamma=0.5
)

checkpoint_config = dict(interval=30, save_optimizer=True, by_epoch=True)
evaluation = dict(interval=15, save_image=True, by_epoch=True)
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
