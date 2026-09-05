"""R0: frozen persistent fault episodes with original detection loss."""
_base_ = "./feq_common.py"
work_dir = "outputs/stage4/feq_core/r0"
data = dict(train=dict(pipeline={{_base_.fault_train_pipeline}}))
model = dict(pts_bbox_head=dict(type="FEQStreamPETRHead", enable_feq_core=False))
