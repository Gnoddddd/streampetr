"""Same graph/data as the stock reference, changing only the head class."""
_base_ = "./feq_legacy_stock_smoke2.py"
work_dir = "outputs/stage4/feq_core/invariance_train_disabled"
model = dict(pts_bbox_head=dict(type="FEQStreamPETRHead", enable_feq_core=False))
