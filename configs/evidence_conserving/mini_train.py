_base_ = ['./mini_debug.py']

# Longer local run. Full nuScenes and publication-scale experiments should be
# moved to a Linux server as specified by the development guide.
runner = dict(type='IterBasedRunner', max_iters=4000)
checkpoint_config = dict(interval=500, max_keep_ckpts=3)
evaluation = dict(interval=1000)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
