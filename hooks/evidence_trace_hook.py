"""MMCV hook writing compact evidence diagnostics as JSON Lines."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from mmcv.runner import HOOKS, Hook
except Exception:  # pragma: no cover
    HOOKS = None

    class Hook:  # type: ignore
        pass


def _register(cls):
    if HOOKS is not None:
        return HOOKS.register_module()(cls)
    return cls


@_register
class EvidenceTraceHook(Hook):
    def __init__(self, interval: int = 20, out_file: str = "evidence_trace.jsonl") -> None:
        self.interval = max(int(interval), 1)
        self.out_file = out_file

    def after_train_iter(self, runner) -> None:
        if (runner.iter + 1) % self.interval != 0:
            return
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        head = getattr(model, "pts_bbox_head", None)
        if head is None or not hasattr(head, "get_last_evidence_summary"):
            return
        summary = head.get_last_evidence_summary()
        if not summary:
            return
        summary = {"iter": int(runner.iter + 1), **summary}
        runner.log_buffer.output.update(
            {
                key: value
                for key, value in summary.items()
                if key != "iter"
            }
        )
        path = Path(runner.work_dir) / self.out_file
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
