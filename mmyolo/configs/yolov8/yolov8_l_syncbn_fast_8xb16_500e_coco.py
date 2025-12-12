from mmengine.config import read_base


with read_base():
    from .yolov8_m_syncbn_fast_8xb16_500e_coco import *  # noqa

# ========================modified parameters======================
from mmyolo.datasets.transforms import YOLOv5MixUp


model = model  # noqa: F405
train_dataloader = train_dataloader  # noqa: F405

deepen_factor = 1.00
widen_factor = 1.00
last_stage_out_channels = 512

mixup_prob = 0.15

# =======================Unmodified in most cases==================
pre_transform = pre_transform  # noqa: F405
mosaic_affine_transform = mosaic_affine_transform  # noqa: F405
last_transform = last_transform  # noqa: F405

model.update(
    dict(
        backbone=dict(
            last_stage_out_channels=last_stage_out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
        ),
        neck=dict(
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            in_channels=[256, 512, last_stage_out_channels],
            out_channels=[256, 512, last_stage_out_channels],
        ),
        bbox_head=dict(
            head_module=dict(
                widen_factor=widen_factor,
                in_channels=[256, 512, last_stage_out_channels],
            ),
        ),
    )
)

train_pipeline = [
    *pre_transform,
    *mosaic_affine_transform,
    dict(
        type=YOLOv5MixUp,
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_transform],
    ),
    *last_transform,
]

train_dataloader.update(dict(dataset=dict(pipeline=train_pipeline)))
