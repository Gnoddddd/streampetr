from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class FreezeExceptHook(Hook):
    """Freeze all parameters except those matching selected patterns."""

    def __init__(self, trainable_patterns):
        self.trainable_patterns = tuple(trainable_patterns)

    def before_run(self, runner):
        model = runner.model

        if hasattr(model, "module"):
            model = model.module

        trainable_names = []
        frozen_names = []
        trainable_numel = 0
        frozen_numel = 0

        for name, parameter in model.named_parameters():
            trainable = any(
                pattern in name
                for pattern in self.trainable_patterns
            )

            parameter.requires_grad = trainable

            if trainable:
                trainable_names.append(name)
                trainable_numel += parameter.numel()
            else:
                frozen_names.append(name)
                frozen_numel += parameter.numel()

        if not trainable_names:
            raise RuntimeError(
                "FreezeExceptHook没有找到任何可训练参数"
            )

        runner.logger.info(
            "FreezeExceptHook trainable patterns: %s",
            self.trainable_patterns,
        )
        runner.logger.info(
            "Trainable parameter tensors: %d, numel: %d",
            len(trainable_names),
            trainable_numel,
        )
        runner.logger.info(
            "Frozen parameter tensors: %d, numel: %d",
            len(frozen_names),
            frozen_numel,
        )

        runner.logger.info(
            "First trainable parameters: %s",
            trainable_names[:20],
        )
