"""Common frozen settings for S3-R1 mini screening (6 equivalent epochs)."""

_base_ = "./mini_convergence_b0.py"

mini_train_samples = 323
effective_batch_size = 1
iters_per_epoch = 323
mini_equivalent_epochs = 6
max_iters = 1938
checkpoint_milestones = (323, 969, 1938)
seed = 2026

runner = dict(type="IterBasedRunner", max_iters=max_iters)
checkpoint_config = dict(interval=max_iters, by_epoch=False, max_keep_ckpts=1)
evaluation = dict(interval=100000)
custom_hooks = [
    dict(type="MilestoneCheckpointHook", milestones=checkpoint_milestones[:-1]),
]

