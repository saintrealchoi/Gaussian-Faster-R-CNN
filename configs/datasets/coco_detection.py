# dataset settings
dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
classes = ('pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor')
data_root = 'data/VisDrone/'
fold = 1
percent = 10
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="CocoDataset",
        classes = classes,
        ann_file="data/VisDrone/annotations/train.json",
        img_prefix="data/VisDrone/VisDrone2019-DET-train/images/",
        pipeline=train_pipeline
        ),
    val=dict(
        type="CocoDataset",
        classes=classes,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'VisDrone2019-DET-val/images/',
        pipeline=test_pipeline
        ),
    test=dict(
        type="CocoDataset",
        classes=classes,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'VisDrone2019-DET-val/images/',
        pipeline=test_pipeline
        )
)
evaluation = dict(interval=1, metric='bbox')


work_dir = "work_dirs/Gaussian-FRCNN"
# log_config = dict(
#     interval=50,
#     hooks=[
#         dict(type="TextLoggerHook"),
#         dict(
#             type="WandbLoggerHook",
#             init_kwargs=dict(
#                 project="pre_release",
#                 name="${cfg_name}",
#                 config=dict(
#                     fold="${fold}",
#                     percent="${percent}",
#                     work_dirs="${work_dir}",
#                     total_step="${runner.max_iters}",
#                 ),
#             ),
#             by_epoch=False,
#         ),
#     ],
# )
