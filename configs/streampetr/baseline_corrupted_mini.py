_base_ = ['../evidence_conserving/mini_debug.py']

# Official binary-objectness StreamPETR under the same random partial-
# observation training distribution used by the full model.
model = dict(pts_bbox_head=dict(type='StreamPETRHead'))
custom_hooks = []
