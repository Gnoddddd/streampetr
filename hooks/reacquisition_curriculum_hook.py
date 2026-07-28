"""Configure the training corruption pipeline before workers are spawned."""

from __future__ import annotations

from mmcv.runner import HOOKS, Hook

from datasets.corruption import ApplyPartialObservation


def _walk_dataset(node):
    if node is None:
        return
    yield node
    child = getattr(node, "dataset", None)
    if child is not None:
        yield from _walk_dataset(child)
    for child in getattr(node, "datasets", ()):
        yield from _walk_dataset(child)


@HOOKS.register_module()
class ReacquisitionCurriculumHook(Hook):
    def __init__(self, curriculum_cfg):
        self.curriculum_cfg = dict(curriculum_cfg)

    def before_run(self, runner):
        configured = 0
        for dataset in _walk_dataset(runner.data_loader.dataset):
            pipeline = getattr(dataset, "pipeline", None)
            for transform in getattr(pipeline, "transforms", ()):
                if isinstance(transform, ApplyPartialObservation):
                    transform.configure_curriculum(self.curriculum_cfg)
                    configured += 1
        if configured == 0:
            raise RuntimeError(
                "ReacquisitionCurriculumHook found no "
                "ApplyPartialObservation transform"
            )
        runner.logger.info(
            "Configured reacquisition curriculum on %d pipeline(s): %s",
            configured,
            self.curriculum_cfg,
        )
