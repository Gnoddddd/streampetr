_base_ = ['./mini_debug.py']

runner = dict(type='IterBasedRunner', max_iters=2)
checkpoint_config = dict(interval=2, max_keep_ckpts=1)
evaluation = dict(interval=999999)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
