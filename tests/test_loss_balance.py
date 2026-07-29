from types import SimpleNamespace

import torch
from mmcv import Config

from hooks.loss_balance_hook import AuxiliaryLossRampHook


class _LogBuffer:
    def __init__(self):
        self.output = {}

    def update(self, values, count):
        self.output.update(values)


class _ToyLoss(torch.nn.Module):
    def forward(self, value):
        return value.square().mean()


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pts_bbox_head = torch.nn.Module()
        self.pts_bbox_head.ternary_loss = _ToyLoss()


def _runner(model, iteration, max_iters=100):
    return SimpleNamespace(
        model=model,
        iter=iteration,
        max_iters=max_iters,
        log_buffer=_LogBuffer(),
        outputs=None,
    )


def test_auxiliary_ramp_only_scales_loss_and_adds_no_state():
    model = _ToyModel()
    before = tuple(model.state_dict())
    value = torch.tensor([2.0, 4.0])
    reference = model.pts_bbox_head.ternary_loss(value)
    hook = AuxiliaryLossRampHook(zero_until=0.2, full_at=0.5)
    runner = _runner(model, iteration=0)
    hook.before_run(runner)

    hook.before_train_iter(runner)
    assert model.pts_bbox_head.ternary_loss(value).item() == 0.0
    runner.iter = 35
    hook.before_train_iter(runner)
    assert torch.equal(
        model.pts_bbox_head.ternary_loss(value),
        reference * 0.5,
    )
    runner.iter = 50
    hook.before_train_iter(runner)
    assert torch.equal(model.pts_bbox_head.ternary_loss(value), reference)
    assert tuple(model.state_dict()) == before
    hook.after_run(runner)
    assert torch.equal(model.pts_bbox_head.ternary_loss(value), reference)


def test_convergence_configs_freeze_fair_budget_and_disable_dn():
    paths = {
        name: Config.fromfile(f"configs/stage3/mini_convergence_{name}.py")
        for name in ("b0", "m1", "m1_ramp")
    }
    for cfg in paths.values():
        assert cfg.runner.max_iters == 3876
        assert cfg.iters_per_epoch == 323
        assert cfg.data.samples_per_gpu == 1
        assert cfg.seed == 2026
        assert cfg.model.pts_bbox_head.with_dn is False
        assert cfg.optimizer.type == "AdamW"
        assert cfg.optimizer_config.type == "GroupedFp16OptimizerHook"
        assert cfg.optimizer_config.grad_clip.max_norm == 35
        assert cfg.load_from.endswith(
            "stream_petr_r50_flash_704_bs2_seq_90e.pth"
        )
    assert paths["b0"].model.pts_bbox_head.type == "StreamPETRHead"
    assert (
        paths["m1"].model.pts_bbox_head.type
        == "EvidenceConservingStreamPETRHead"
    )
    assert paths["m1"].model == paths["m1_ramp"].model
    ramp_hooks = [hook.type for hook in paths["m1_ramp"].custom_hooks]
    assert "AuxiliaryLossRampHook" in ramp_hooks
    assert "AuxiliaryLossRampHook" not in [
        hook.type for hook in paths["m1"].custom_hooks
    ]
