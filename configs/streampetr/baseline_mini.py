import os

# The child config is executed before MMCV resolves its base file, so this
# environment switch is visible while mini_debug.py builds its pipeline.
os.environ['EVIDENCE3D_DISABLE_RANDOM_CORRUPTION'] = '1'

_base_ = ['../evidence_conserving/mini_debug.py']

# Clean-data official StreamPETR baseline with the same R50/query/memory budget.
model = dict(
    pts_bbox_head=dict(
        type='StreamPETRHead',
    )
)
custom_hooks = []
