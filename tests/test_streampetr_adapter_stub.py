"""Shape-level integration test for the StreamPETR adapter without OpenMMLab.

The real legacy stack is installed only in the WSL ``streampetr`` environment.
This test stubs the upstream registry/base class and exercises the custom
memory-write path, which is where most integration shape errors occur.
"""

from __future__ import annotations

import importlib
import sys
import types

import torch
from torch import nn


def _install_module(monkeypatch, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # mark parent modules as packages
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_adapter_memory_update_contract(monkeypatch):
    mmcv = _install_module(monkeypatch, "mmcv")
    mmcv_runner = _install_module(monkeypatch, "mmcv.runner")
    mmcv.runner = mmcv_runner
    mmcv_runner.force_fp32 = lambda apply_to=None: (lambda function: function)

    mmdet = _install_module(monkeypatch, "mmdet")
    mmdet_core = _install_module(monkeypatch, "mmdet.core")
    mmdet_models = _install_module(monkeypatch, "mmdet.models")
    mmdet_utils = _install_module(monkeypatch, "mmdet.models.utils")
    mmdet_transformer = _install_module(monkeypatch, "mmdet.models.utils.transformer")
    mmdet.core = mmdet_core
    mmdet.models = mmdet_models
    mmdet_models.utils = mmdet_utils
    mmdet_utils.transformer = mmdet_transformer

    def multi_apply(function, *args):
        mapped = list(map(function, *args))
        return tuple(map(list, zip(*mapped))) if mapped else tuple()

    mmdet_core.multi_apply = multi_apply
    mmdet_core.reduce_mean = lambda tensor: tensor
    mmdet_transformer.inverse_sigmoid = lambda tensor, eps=1e-5: torch.logit(
        tensor.clamp(eps, 1.0 - eps)
    )

    class Registry:
        def register_module(self):
            return lambda cls: cls

    mmdet_models.HEADS = Registry()

    for name in (
        "projects",
        "projects.mmdet3d_plugin",
        "projects.mmdet3d_plugin.core",
        "projects.mmdet3d_plugin.core.bbox",
        "projects.mmdet3d_plugin.models",
        "projects.mmdet3d_plugin.models.dense_heads",
        "projects.mmdet3d_plugin.models.utils",
    ):
        _install_module(monkeypatch, name)

    util = _install_module(monkeypatch, "projects.mmdet3d_plugin.core.bbox.util")
    util.normalize_bbox = lambda boxes, pc_range: boxes

    misc = _install_module(monkeypatch, "projects.mmdet3d_plugin.models.utils.misc")

    def topk_gather(values, indexes):
        if indexes is None:
            return values
        if indexes.ndim == 3 and indexes.shape[-1] == 1:
            indexes = indexes.squeeze(-1)
        expanded = indexes
        while expanded.ndim < values.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand(*indexes.shape, *values.shape[2:])
        return torch.gather(values, 1, expanded)

    misc.topk_gather = topk_gather
    misc.transform_reference_points = (
        lambda points, transform, reverse=False: points
    )

    positional = _install_module(
        monkeypatch, "projects.mmdet3d_plugin.models.utils.positional_encoding"
    )
    positional.pos2posemb3d = lambda points: points

    head_module = _install_module(
        monkeypatch,
        "projects.mmdet3d_plugin.models.dense_heads.streampetr_head",
    )

    class FakeStreamPETRHead(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.embed_dims = int(kwargs.get("embed_dims", 8))
            self.num_pred = int(kwargs.get("num_pred", 2))
            self.num_query = int(kwargs.get("num_query", 4))
            self.memory_len = int(kwargs.get("memory_len", 6))
            self.topk_proposals = int(kwargs.get("topk_proposals", 3))
            self.num_propagated = int(kwargs.get("num_propagated", 2))
            self.reset_memory()

        def reset_memory(self):
            self.memory_embedding = None
            self.memory_reference_point = None
            self.memory_timestamp = None
            self.memory_egopose = None
            self.memory_velo = None

        def pre_update_memory(self, data):
            previous = data["prev_exists"]
            batch = previous.shape[0]
            if self.memory_embedding is None:
                self.memory_embedding = previous.new_zeros(
                    batch, self.memory_len, self.embed_dims
                )
                self.memory_reference_point = previous.new_zeros(
                    batch, self.memory_len, 3
                )
                self.memory_timestamp = previous.new_zeros(
                    batch, self.memory_len, 1, dtype=torch.float64
                )
                eye = torch.eye(4, device=previous.device, dtype=previous.dtype)
                self.memory_egopose = eye.view(1, 1, 4, 4).repeat(
                    batch, self.memory_len, 1, 1
                )
                self.memory_velo = previous.new_zeros(batch, self.memory_len, 2)
            else:
                self.memory_embedding = self.memory_embedding[:, : self.memory_len]
                self.memory_reference_point = self.memory_reference_point[:, : self.memory_len]
                self.memory_timestamp = self.memory_timestamp[:, : self.memory_len]
                self.memory_egopose = self.memory_egopose[:, : self.memory_len]
                self.memory_velo = self.memory_velo[:, : self.memory_len]

        def post_update_memory(
            self, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict
        ):
            self.parent_post_update_called = True

    head_module.StreamPETRHead = FakeStreamPETRHead

    sys.modules.pop("models.streampetr_adapter", None)
    adapter = importlib.import_module("models.streampetr_adapter")
    head = adapter.EvidenceConservingStreamPETRHead(
        num_classes=3,
        in_channels=8,
        embed_dims=8,
        num_pred=2,
        num_query=4,
        memory_len=6,
        topk_proposals=3,
        num_propagated=2,
        num_cameras=2,
        evidence_warmup_steps=0,
        enable_source_ledger=True,
        enable_reacquisition_diagnostics=True,
        source_camera_names=("CAM_LEFT", "CAM_RIGHT"),
        temporal_update_cfg={"gamma": 0.9, "evidence_scale": 2.0},
    )

    identity = torch.eye(4).view(1, 4, 4)
    data = {
        "prev_exists": torch.zeros(1),
        "timestamp": torch.zeros(1, dtype=torch.float64),
        "ego_pose": identity,
        "ego_pose_inv": identity,
    }
    head.pre_update_memory(data)

    layers, batch, queries, classes = 2, 1, 6, 3
    all_cls_scores = torch.full((layers, batch, queries, classes), 2.0)
    all_bbox_preds = torch.zeros(layers, batch, queries, 10)
    all_bbox_preds[..., :3] = torch.tensor([5.0, 0.0, 1.0])
    outs_dec = torch.randn(layers, batch, queries, 8)
    rec_pose = identity.view(1, 1, 4, 4).repeat(batch, queries, 1, 1)
    ternary = torch.tensor([0.90, 0.05, 0.05]).view(1, 1, 1, 3).repeat(
        layers, batch, queries, 1
    )
    source = torch.tensor([0.5, 0.5]).view(1, 1, 1, 2).repeat(
        layers, batch, queries, 1
    )
    observation = {
        "observability": torch.ones(layers, batch, queries),
        "source_vector": source,
        "fresh_ratio": torch.ones(layers, batch, queries),
        "effective_count": torch.ones(layers, batch, queries),
    }

    state = head._update_memory_with_evidence(
        data,
        rec_pose,
        all_cls_scores,
        all_bbox_preds,
        outs_dec,
        None,
        ternary,
        observation,
    )
    assert state["alpha"].shape == (batch, queries)
    assert head.memory_embedding.shape == (batch, 9, 8)
    assert head.memory_reference_point.shape == (batch, 9, 3)
    assert head.evidence_ledger.alpha.shape == (batch, 9)
    assert head.evidence_ledger.source_evidence.shape == (batch, 9, 2)
    diagnostics = head.get_last_evidence_diagnostics()
    assert diagnostics["source_camera_names"] == ("CAM_LEFT", "CAM_RIGHT")
    assert diagnostics["source_evidence"].shape == (batch, queries, 2)
    assert diagnostics["query_index"].shape == (batch, queries)
    assert diagnostics["query_source"].tolist() == [[0, 0, 0, 0, 1, 1]]
    assert diagnostics["decoder_layer"].unique().item() == layers - 1
    assert diagnostics["previous_action"].shape == (batch, queries)
    assert diagnostics["previous_source_vector"].shape == (
        batch,
        queries,
        2,
    )
    assert diagnostics["previous_center"].shape == (batch, queries, 3)
    assert diagnostics["current_center_global"].shape == (
        batch,
        queries,
        3,
    )
    assert diagnostics["velocity_extrapolated_center"].shape == (
        batch,
        queries,
        3,
    )
    assert diagnostics["velocity_global"].shape == (batch, queries, 2)
    assert diagnostics["predicted_class"].shape == (batch, queries)
    assert diagnostics["predicted_score"].shape == (batch, queries)
    assert diagnostics["actual_memory_write"].shape == (batch, queries)
    assert diagnostics["actual_memory_write"].sum().item() == 3
    assert diagnostics["topk_selected"].sum().item() == 3
    assert (diagnostics["memory_slot"] >= 0).sum().item() == 3
    assert head.get_last_reacquisition_trigger_diagnostics() == []
    head._last_evidence_diagnostics["is_reacquired"][0, 0] = True
    trigger_rows = head.get_last_reacquisition_trigger_diagnostics()
    assert len(trigger_rows) == 1
    assert trigger_rows[0]["query_index"] == 0
    assert trigger_rows[0]["query_source"] == 0
    assert trigger_rows[0]["current_center_global"].shape == (3,)
    for key, value in tuple(head._last_evidence_diagnostics.items()):
        if torch.is_tensor(value) and value.is_floating_point():
            head._last_evidence_diagnostics[key] = value.half()
    half_rows = head.get_last_reacquisition_trigger_diagnostics()
    assert len(half_rows) == 1
    assert torch.isfinite(half_rows[0]["current_center_global"]).all()
    assert not any(
        "diagnostic" in key or "_last_evidence" in key
        for key in head.state_dict()
    )
    checkpoint = {"state_dict": head.state_dict()}
    assert not any(
        "diagnostic" in key or "_last_evidence" in key
        for key in checkpoint["state_dict"]
    )
    assert torch.all(state["write_mask"])

    # The observer is downstream of all decision and memory tensors. Running
    # the same update with it disabled must therefore be bit-for-bit equal.
    head_without_diagnostics = adapter.EvidenceConservingStreamPETRHead(
        num_classes=3,
        in_channels=8,
        embed_dims=8,
        num_pred=2,
        num_query=4,
        memory_len=6,
        topk_proposals=3,
        num_propagated=2,
        num_cameras=2,
        evidence_warmup_steps=0,
        enable_source_ledger=True,
        source_camera_names=("CAM_LEFT", "CAM_RIGHT"),
        temporal_update_cfg={"gamma": 0.9, "evidence_scale": 2.0},
    )
    head_without_diagnostics.pre_update_memory(data)
    state_without_diagnostics = (
        head_without_diagnostics._update_memory_with_evidence(
            data,
            rec_pose,
            all_cls_scores,
            all_bbox_preds,
            outs_dec,
            None,
            ternary,
            observation,
        )
    )
    for key in (
        "alpha",
        "beta",
        "conservation_residual",
        "source_evidence",
        "source_mass_residual",
        "action",
        "write_mask",
    ):
        assert torch.equal(state[key], state_without_diagnostics[key])
    for key in (
        "memory_embedding",
        "memory_reference_point",
        "memory_timestamp",
        "memory_egopose",
        "memory_velo",
    ):
        assert torch.equal(
            getattr(head, key),
            getattr(head_without_diagnostics, key),
        )
    assert "query_index" not in (
        head_without_diagnostics.get_last_evidence_diagnostics()
    )

    head.reset_memory()
    assert head.get_last_evidence_diagnostics() == {}
    for name in head.evidence_ledger._STATE_NAMES:
        assert getattr(head.evidence_ledger, name) is None


def test_adapter_can_disable_evidence_memory_for_ternary_only_ablation(monkeypatch):
    # Reuse the full stub setup in a separate subprocess-like pytest invocation
    # by calling the primary test setup, then instantiate from its imported module.
    test_adapter_memory_update_contract(monkeypatch)
    adapter = sys.modules["models.streampetr_adapter"]
    head = adapter.EvidenceConservingStreamPETRHead(
        num_classes=3,
        in_channels=8,
        embed_dims=8,
        num_pred=2,
        num_query=4,
        memory_len=6,
        topk_proposals=3,
        num_propagated=2,
        num_cameras=2,
        enable_evidence_memory=False,
        calibrate_detection_scores=False,
    )
    identity = torch.eye(4).view(1, 4, 4)
    data = {
        "prev_exists": torch.zeros(1),
        "timestamp": torch.zeros(1, dtype=torch.float64),
        "ego_pose": identity,
        "ego_pose_inv": identity,
    }
    head.pre_update_memory(data)
    layers, batch, queries, classes = 2, 1, 6, 3
    cls = torch.zeros(layers, batch, queries, classes)
    boxes = torch.zeros(layers, batch, queries, 10)
    decoder = torch.zeros(layers, batch, queries, 8)
    pose = identity.view(1, 1, 4, 4).repeat(batch, queries, 1, 1)
    ternary = torch.tensor([0.8, 0.1, 0.1]).view(1, 1, 1, 3).repeat(
        layers, batch, queries, 1
    )
    observation = {
        "observability": torch.ones(layers, batch, queries),
        "source_vector": torch.full((layers, batch, queries, 2), 0.5),
        "fresh_ratio": torch.ones(layers, batch, queries),
        "effective_count": torch.ones(layers, batch, queries),
    }
    head._update_memory_with_evidence(
        data, pose, cls, boxes, decoder, None, ternary, observation
    )
    assert head.parent_post_update_called
    assert head.evidence_ledger.alpha.shape[1] == head.memory_len


def test_warmup_keeps_official_memory_write_before_hard_defer(monkeypatch):
    test_adapter_memory_update_contract(monkeypatch)
    adapter = sys.modules["models.streampetr_adapter"]
    head = adapter.EvidenceConservingStreamPETRHead(
        num_classes=3,
        in_channels=8,
        embed_dims=8,
        num_pred=2,
        num_query=4,
        memory_len=6,
        topk_proposals=3,
        num_propagated=2,
        num_cameras=2,
        evidence_warmup_steps=10,
    )
    identity = torch.eye(4).view(1, 4, 4)
    data = {
        "prev_exists": torch.zeros(1),
        "timestamp": torch.zeros(1, dtype=torch.float64),
        "ego_pose": identity,
        "ego_pose_inv": identity,
    }
    head.pre_update_memory(data)
    layers, batch, queries, classes = 2, 1, 6, 3
    cls = torch.ones(layers, batch, queries, classes)
    boxes = torch.zeros(layers, batch, queries, 10)
    decoder = torch.ones(layers, batch, queries, 8)
    pose = identity.view(1, 1, 4, 4).repeat(batch, queries, 1, 1)
    ternary = torch.tensor([0.01, 0.01, 0.98]).view(1, 1, 1, 3).repeat(
        layers, batch, queries, 1
    )
    observation = {
        "observability": torch.zeros(layers, batch, queries),
        "source_vector": torch.zeros(layers, batch, queries, 2),
        "fresh_ratio": torch.zeros(layers, batch, queries),
        "effective_count": torch.zeros(layers, batch, queries),
    }
    state = head._update_memory_with_evidence(
        data, pose, cls, boxes, decoder, None, ternary, observation
    )
    assert not torch.any(state["write_mask"])
    assert torch.all(head.memory_embedding[:, :3] != 0)
    assert head.get_last_evidence_summary()["warmup_active"] == 1.0
