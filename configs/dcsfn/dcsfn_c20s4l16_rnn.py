exp_name = 'dcsfn_c20s4l16_rnn'
work_dir = f'./work_dirs/{exp_name}'

# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='DCSFN',
        in_channels=3,
        out_channels=3,
        mid_channels=20,
        recurrent_unit='RNN',
        num_encoder_decoder_layers=16,
        num_inner_scales=4,
        num_inner_convs=4,
    ),
    losses=[dict(type='SSIMLoss', loss_weight=1.0, reduction='mean')],
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
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=24, drop_last=True),
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
optimizers = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)

# learning policy
runner = dict(type='EpochBasedRunner', max_epochs=800)
lr_config = dict(
    policy='Step',
    by_epoch=True,
    step=[480, 640],
    gamma=0.1
)

checkpoint_config = dict(interval=100, save_optimizer=True, by_epoch=True)
evaluation = dict(interval=100, save_image=True, by_epoch=True)
log_config = dict(
    interval=200,
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
