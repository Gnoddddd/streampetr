"""B0: explicitly clean-only StreamPETR."""
_base_ = "./feq_common.py"
work_dir = "outputs/stage4/feq_core/b0"
model = dict(pts_bbox_head=dict(type="FEQStreamPETRHead", enable_feq_core=False))
