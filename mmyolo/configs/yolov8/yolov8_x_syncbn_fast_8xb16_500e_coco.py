from mmengine.config import read_base


with read_base():
    from .yolov8_l_syncbn_fast_8xb16_500e_coco import *  # noqa

model = model  # noqa: F405

deepen_factor = 1.00
widen_factor = 1.25

model.update(
    dict(
        backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
        neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
        bbox_head=dict(head_module=dict(widen_factor=widen_factor)),
    )
)
