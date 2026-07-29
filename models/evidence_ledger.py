"""Compact provenance-aware evidence ledger aligned with StreamPETR memory."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .keep_recover_defer import Action, KeepRecoverDeferPolicy
from .innovation import ReliabilityCalibratedInnovation
from .temporal_update import EvidenceConservingTemporalUpdate


def _gather(values: Tensor, indexes: Tensor) -> Tensor:
    if indexes.ndim == 3 and indexes.shape[-1] == 1:
        indexes = indexes.squeeze(-1)
    expand = indexes
    while expand.ndim < values.ndim:
        expand = expand.unsqueeze(-1)
    expand = expand.expand(*indexes.shape, *values.shape[2:])
    return torch.gather(values, dim=1, index=expand)


class EvidenceLedger(nn.Module):
    """Stateful ledger for source, age, and Beta evidence of memory queries."""

    _STATE_NAMES = (
        "alpha",
        "beta",
        "source_evidence",
        "provenance",
        "legacy_provenance",
        "age",
        "effective_count",
        "observability",
        "novelty",
        "action",
        "reference_feature",
        "reference_geometry",
        "reference_class_distribution",
        "reference_ternary_distribution",
        "reference_valid",
        "pre_gap_strength",
        "pre_gap_presence",
        "pre_gap_uncertainty",
        "pre_gap_source_evidence",
        "gap_active",
        "gap_age",
        "reacquisition_consumed",
    )

    def __init__(
        self,
        memory_len: int,
        num_cameras: int = 6,
        temporal_update: Optional[EvidenceConservingTemporalUpdate] = None,
        policy: Optional[KeepRecoverDeferPolicy] = None,
        novelty_eps: float = 1e-6,
        observation_gate_threshold: float = 0.05,
        enable_source_ledger: bool = False,
        source_decay: Optional[float] = None,
        source_mass_tolerance: float = 1e-5,
        use_source_ledger_for_evidence: bool = False,
        use_source_ledger_for_policy: bool = False,
        feature_dim: int = 0,
        class_dim: int = 0,
        innovation_cfg: Optional[Dict[str, Any]] = None,
        innovation_warmup_iters: int = 0,
        innovation_transition_iters: int = 0,
    ) -> None:
        super().__init__()
        self.memory_len = int(memory_len)
        self.num_cameras = int(num_cameras)
        self.temporal_update = temporal_update or EvidenceConservingTemporalUpdate()
        self.policy = policy or KeepRecoverDeferPolicy()
        self.novelty_eps = float(novelty_eps)
        self.observation_gate_threshold = float(observation_gate_threshold)
        self.enable_source_ledger = bool(enable_source_ledger)
        self.source_decay = float(
            self.temporal_update.gamma if source_decay is None else source_decay
        )
        if not 0.0 <= self.source_decay <= 1.0:
            raise ValueError("source_decay must be in [0, 1]")
        self.source_mass_tolerance = float(source_mass_tolerance)
        if self.source_mass_tolerance < 0.0:
            raise ValueError("source_mass_tolerance must be non-negative")
        self.use_source_ledger_for_evidence = bool(
            use_source_ledger_for_evidence
        )
        self.use_source_ledger_for_policy = bool(
            use_source_ledger_for_policy
        )
        self.feature_dim = int(feature_dim)
        self.class_dim = int(class_dim)
        self.geometry_dim = 6
        self.innovation = ReliabilityCalibratedInnovation(
            **dict(innovation_cfg or {})
        )
        self.innovation_warmup_iters = max(int(innovation_warmup_iters), 0)
        self.innovation_transition_iters = max(
            int(innovation_transition_iters), 0
        )
        if (
            self.use_source_ledger_for_evidence
            or self.use_source_ledger_for_policy
        ):
            raise ValueError(
                "S2.2 supports source tracking only; source-ledger evidence "
                "and policy coupling remain disabled"
            )
        for name in self._STATE_NAMES:
            self.register_buffer(name, None, persistent=False)
        # MMCV 1.6's custom get_state_dict intentionally serializes even
        # non-persistent buffers. Strip scene-local state at the owning module
        # boundary so ordinary MMCV training checkpoints remain clean.
        self._register_state_dict_hook(self._strip_runtime_checkpoint_state)
        self.reset()

    @staticmethod
    def _strip_runtime_checkpoint_state(
        module: nn.Module,
        destination: Dict[str, Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
    ) -> None:
        del local_metadata
        for name in EvidenceLedger._STATE_NAMES:
            destination.pop(prefix + name, None)

    def reset(self) -> None:
        for name in self._STATE_NAMES:
            setattr(self, name, None)
        self._scene_tokens: Optional[Tuple[str, ...]] = None
        self.last_scene_reset = False
        self.scene_reset_count = 0

    def export_runtime_state(self, to_cpu: bool = True) -> Dict[str, Any]:
        """Explicitly export scene-local state without affecting checkpoints."""
        buffers: Dict[str, Optional[Tensor]] = {}
        for name in self._STATE_NAMES:
            value = getattr(self, name)
            if value is None:
                buffers[name] = None
            else:
                value = value.detach().clone()
                buffers[name] = value.cpu() if to_cpu else value
        return {
            "version": 4,
            "memory_len": self.memory_len,
            "num_cameras": self.num_cameras,
            "feature_dim": self.feature_dim,
            "class_dim": self.class_dim,
            "scene_tokens": self._scene_tokens,
            "buffers": buffers,
        }

    def load_runtime_state(
        self,
        state: Dict[str, Any],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Restore state only when explicitly requested and shape-compatible."""
        version = int(state.get("version", -1))
        if version not in (1, 2, 3, 4):
            raise ValueError("Unsupported evidence runtime-state version")
        if int(state.get("memory_len", -1)) != self.memory_len:
            raise ValueError("runtime memory_len does not match this ledger")
        if int(state.get("num_cameras", -1)) != self.num_cameras:
            raise ValueError("runtime num_cameras does not match this ledger")

        buffers = state.get("buffers")
        if not isinstance(buffers, dict):
            raise ValueError("runtime state must contain a buffers dictionary")
        if version == 1:
            buffers = dict(buffers)
            provenance = buffers.get("provenance")
            buffers["source_evidence"] = (
                None
                if provenance is None
                else torch.zeros_like(provenance)
            )
            buffers["legacy_provenance"] = (
                None
                if provenance is None
                else provenance.detach().clone()
            )
        if version in (1, 2):
            buffers = dict(buffers)
            alpha = buffers.get("alpha")
            if alpha is None:
                for name in (
                    "reference_feature",
                    "reference_geometry",
                    "reference_class_distribution",
                    "reference_ternary_distribution",
                    "reference_valid",
                ):
                    buffers[name] = None
            else:
                batch_size, state_len = alpha.shape
                buffers["reference_feature"] = alpha.new_zeros(
                    batch_size, state_len, self.feature_dim
                )
                buffers["reference_geometry"] = alpha.new_zeros(
                    batch_size, state_len, self.geometry_dim
                )
                buffers["reference_class_distribution"] = alpha.new_zeros(
                    batch_size, state_len, self.class_dim
                )
                buffers["reference_ternary_distribution"] = alpha.new_zeros(
                    batch_size, state_len, 3
                )
                buffers["reference_valid"] = torch.zeros(
                    batch_size,
                    state_len,
                    device=alpha.device,
                    dtype=torch.bool,
                )
        if version in (3, 4):
            if int(state.get("feature_dim", -1)) != self.feature_dim:
                raise ValueError("runtime feature_dim does not match this ledger")
            if int(state.get("class_dim", -1)) != self.class_dim:
                raise ValueError("runtime class_dim does not match this ledger")
        if version in (1, 2, 3):
            buffers = dict(buffers)
            alpha = buffers.get("alpha")
            for name in (
                "pre_gap_strength",
                "pre_gap_presence",
                "pre_gap_uncertainty",
                "gap_active",
                "gap_age",
                "reacquisition_consumed",
            ):
                buffers[name] = (
                    None if alpha is None else torch.zeros_like(alpha)
                )
            source = buffers.get("source_evidence")
            buffers["pre_gap_source_evidence"] = (
                None if source is None else torch.zeros_like(source)
            )
        missing = set(self._STATE_NAMES) - set(buffers)
        if missing:
            raise ValueError(f"runtime state is missing buffers: {sorted(missing)}")
        if all(buffers[name] is None for name in self._STATE_NAMES):
            self.reset()
            return
        if any(buffers[name] is None for name in self._STATE_NAMES):
            raise ValueError("runtime buffers must be either all set or all empty")

        alpha = buffers["alpha"]
        if not torch.is_tensor(alpha) or alpha.ndim != 2:
            raise ValueError("runtime alpha must have shape [B,M]")
        batch_size, state_len = alpha.shape
        expected_shapes = {
            "alpha": (batch_size, state_len),
            "beta": (batch_size, state_len),
            "source_evidence": (
                batch_size,
                state_len,
                self.num_cameras,
            ),
            "provenance": (batch_size, state_len, self.num_cameras),
            "legacy_provenance": (
                batch_size,
                state_len,
                self.num_cameras,
            ),
            "age": (batch_size, state_len),
            "effective_count": (batch_size, state_len),
            "observability": (batch_size, state_len),
            "novelty": (batch_size, state_len),
            "action": (batch_size, state_len),
            "reference_feature": (
                batch_size,
                state_len,
                self.feature_dim,
            ),
            "reference_geometry": (
                batch_size,
                state_len,
                self.geometry_dim,
            ),
            "reference_class_distribution": (
                batch_size,
                state_len,
                self.class_dim,
            ),
            "reference_ternary_distribution": (
                batch_size,
                state_len,
                3,
            ),
            "reference_valid": (batch_size, state_len),
            "pre_gap_strength": (batch_size, state_len),
            "pre_gap_presence": (batch_size, state_len),
            "pre_gap_uncertainty": (batch_size, state_len),
            "pre_gap_source_evidence": (
                batch_size,
                state_len,
                self.num_cameras,
            ),
            "gap_active": (batch_size, state_len),
            "gap_age": (batch_size, state_len),
            "reacquisition_consumed": (batch_size, state_len),
        }
        target_device = torch.device(device) if device is not None else alpha.device
        for name, expected_shape in expected_shapes.items():
            value = buffers[name]
            if not torch.is_tensor(value) or tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"runtime {name} must have shape {expected_shape}"
                )
            if name == "action":
                restored = value.to(device=target_device, dtype=torch.long)
            elif name in (
                "reference_valid",
                "gap_active",
                "reacquisition_consumed",
            ):
                restored = value.to(device=target_device, dtype=torch.bool)
            else:
                target_dtype = dtype if dtype is not None else value.dtype
                restored = value.to(device=target_device, dtype=target_dtype)
            setattr(self, name, restored.detach().clone())

        scene_tokens = state.get("scene_tokens")
        if scene_tokens is not None:
            scene_tokens = tuple(str(token) for token in scene_tokens)
            if len(scene_tokens) != batch_size:
                raise ValueError("scene_tokens length must match runtime batch size")
        self._scene_tokens = scene_tokens
        self.last_scene_reset = False
        self.scene_reset_count = 0

    def _initialize(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self.alpha = torch.ones(batch_size, self.memory_len, device=device, dtype=dtype)
        self.beta = torch.ones_like(self.alpha)
        self.source_evidence = torch.zeros(
            batch_size, self.memory_len, self.num_cameras, device=device, dtype=dtype
        )
        self.provenance = torch.zeros(
            batch_size, self.memory_len, self.num_cameras, device=device, dtype=dtype
        )
        self.legacy_provenance = torch.zeros_like(self.provenance)
        self.age = torch.zeros_like(self.alpha)
        self.effective_count = torch.zeros_like(self.alpha)
        self.observability = torch.zeros_like(self.alpha)
        self.novelty = torch.zeros_like(self.alpha)
        self.action = torch.full(
            (batch_size, self.memory_len),
            int(Action.DEFER),
            device=device,
            dtype=torch.long,
        )
        self.reference_feature = torch.zeros(
            batch_size,
            self.memory_len,
            self.feature_dim,
            device=device,
            dtype=dtype,
        )
        self.reference_geometry = torch.zeros(
            batch_size,
            self.memory_len,
            self.geometry_dim,
            device=device,
            dtype=dtype,
        )
        self.reference_class_distribution = torch.zeros(
            batch_size,
            self.memory_len,
            self.class_dim,
            device=device,
            dtype=dtype,
        )
        self.reference_ternary_distribution = torch.zeros(
            batch_size,
            self.memory_len,
            3,
            device=device,
            dtype=dtype,
        )
        self.reference_valid = torch.zeros(
            batch_size,
            self.memory_len,
            device=device,
            dtype=torch.bool,
        )
        self.pre_gap_strength = torch.zeros_like(self.alpha)
        self.pre_gap_presence = torch.zeros_like(self.alpha)
        self.pre_gap_uncertainty = torch.zeros_like(self.alpha)
        self.pre_gap_source_evidence = torch.zeros_like(self.source_evidence)
        self.gap_active = torch.zeros_like(self.alpha, dtype=torch.bool)
        self.gap_age = torch.zeros_like(self.alpha)
        self.reacquisition_consumed = torch.zeros_like(
            self.alpha, dtype=torch.bool
        )

    def pre_update(
        self,
        prev_exists: Tensor,
        scene_tokens: Optional[Sequence[str]] = None,
        geometry_transform: Optional[Tensor] = None,
    ) -> None:
        if prev_exists.ndim != 1:
            prev_exists = prev_exists.reshape(-1)
        batch_size = prev_exists.shape[0]
        normalized_tokens = (
            None
            if scene_tokens is None
            else tuple(str(token) for token in scene_tokens)
        )
        if normalized_tokens is not None and len(normalized_tokens) != batch_size:
            raise ValueError("scene_tokens length must match prev_exists batch size")

        scene_changed = torch.zeros(
            batch_size, device=prev_exists.device, dtype=torch.bool
        )
        if (
            normalized_tokens is not None
            and self._scene_tokens is not None
            and len(self._scene_tokens) == batch_size
        ):
            scene_changed = torch.tensor(
                [
                    previous != current
                    for previous, current in zip(
                        self._scene_tokens, normalized_tokens
                    )
                ],
                device=prev_exists.device,
                dtype=torch.bool,
            )
        effective_prev_exists = prev_exists * (~scene_changed).to(prev_exists.dtype)
        reset_rows = (~effective_prev_exists.bool()).sum().item()
        self.last_scene_reset = bool(reset_rows)
        self.scene_reset_count += int(reset_rows)

        if self.alpha is None or self.alpha.shape[0] != batch_size:
            self._initialize(batch_size, prev_exists.device, prev_exists.dtype)
            self._scene_tokens = normalized_tokens
            return

        if geometry_transform is not None:
            self.transform_reference_geometry(geometry_transform)

        # Match StreamPETR: keep only the configured memory horizon first.
        self.alpha = self.alpha[:, : self.memory_len]
        self.beta = self.beta[:, : self.memory_len]
        self.source_evidence = self.source_evidence[:, : self.memory_len]
        self.provenance = self.provenance[:, : self.memory_len]
        self.legacy_provenance = self.legacy_provenance[:, : self.memory_len]
        self.age = self.age[:, : self.memory_len]
        self.effective_count = self.effective_count[:, : self.memory_len]
        self.observability = self.observability[:, : self.memory_len]
        self.novelty = self.novelty[:, : self.memory_len]
        self.action = self.action[:, : self.memory_len]
        self.reference_feature = self.reference_feature[:, : self.memory_len]
        self.reference_geometry = self.reference_geometry[:, : self.memory_len]
        self.reference_class_distribution = self.reference_class_distribution[
            :, : self.memory_len
        ]
        self.reference_ternary_distribution = self.reference_ternary_distribution[
            :, : self.memory_len
        ]
        self.reference_valid = self.reference_valid[:, : self.memory_len]
        self.pre_gap_strength = self.pre_gap_strength[:, : self.memory_len]
        self.pre_gap_presence = self.pre_gap_presence[:, : self.memory_len]
        self.pre_gap_uncertainty = self.pre_gap_uncertainty[:, : self.memory_len]
        self.pre_gap_source_evidence = self.pre_gap_source_evidence[
            :, : self.memory_len
        ]
        self.gap_active = self.gap_active[:, : self.memory_len]
        self.gap_age = self.gap_age[:, : self.memory_len]
        self.reacquisition_consumed = self.reacquisition_consumed[
            :, : self.memory_len
        ]

        keep = effective_prev_exists.to(self.alpha.dtype).view(batch_size, 1)
        self.alpha = 1.0 + (self.alpha - 1.0) * keep
        self.beta = 1.0 + (self.beta - 1.0) * keep
        self.source_evidence = self.source_evidence * keep.unsqueeze(-1)
        self.provenance = self.provenance * keep.unsqueeze(-1)
        self.legacy_provenance = self.legacy_provenance * keep.unsqueeze(-1)
        self.age = self.age * keep
        self.effective_count = self.effective_count * keep
        self.observability = self.observability * keep
        self.novelty = self.novelty * keep
        self.action = torch.where(
            keep.bool(),
            self.action,
            torch.full_like(self.action, int(Action.DEFER)),
        )
        self.reference_feature = self.reference_feature * keep.unsqueeze(-1)
        self.reference_geometry = self.reference_geometry * keep.unsqueeze(-1)
        self.reference_class_distribution = (
            self.reference_class_distribution * keep.unsqueeze(-1)
        )
        self.reference_ternary_distribution = (
            self.reference_ternary_distribution * keep.unsqueeze(-1)
        )
        self.reference_valid = self.reference_valid & keep.bool()
        self.pre_gap_strength = self.pre_gap_strength * keep
        self.pre_gap_presence = self.pre_gap_presence * keep
        self.pre_gap_uncertainty = self.pre_gap_uncertainty * keep
        self.pre_gap_source_evidence = (
            self.pre_gap_source_evidence * keep.unsqueeze(-1)
        )
        self.gap_active = self.gap_active & keep.bool()
        self.gap_age = self.gap_age * keep
        self.reacquisition_consumed = (
            self.reacquisition_consumed & keep.bool()
        )
        self._scene_tokens = normalized_tokens

    def transform_reference_geometry(self, transform: Tensor) -> None:
        """Transform stored geometry between ego/global coordinate frames."""
        if self.reference_geometry is None or not torch.any(self.reference_valid):
            return
        transform = transform.to(
            device=self.reference_geometry.device,
            dtype=self.reference_geometry.dtype,
        )
        if transform.ndim != 3 or transform.shape[-2:] != (4, 4):
            raise ValueError("geometry transform must have shape [B,4,4]")
        center = self.reference_geometry[..., :3]
        ones = torch.ones_like(center[..., :1])
        center_h = torch.cat((center, ones), dim=-1)
        transformed_center = torch.einsum(
            "bij,bmj->bmi", transform, center_h
        )[..., :3]
        rotation = transform[..., :2, :2]
        heading = torch.stack(
            (
                torch.cos(self.reference_geometry[..., 3]),
                torch.sin(self.reference_geometry[..., 3]),
            ),
            dim=-1,
        )
        transformed_heading = torch.einsum(
            "bij,bmj->bmi", rotation, heading
        )
        transformed_yaw = torch.atan2(
            transformed_heading[..., 1], transformed_heading[..., 0]
        )
        transformed_velocity = torch.einsum(
            "bij,bmj->bmi",
            rotation,
            self.reference_geometry[..., 4:6],
        )
        transformed = torch.cat(
            (
                transformed_center,
                transformed_yaw.unsqueeze(-1),
                transformed_velocity,
            ),
            dim=-1,
        )
        self.reference_geometry = torch.where(
            self.reference_valid.unsqueeze(-1),
            transformed,
            self.reference_geometry,
        )

    def _query_priors(
        self,
        batch_size: int,
        num_queries: int,
        num_base_queries: int,
        num_propagated: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Tensor]:
        alpha = torch.ones(batch_size, num_queries, device=device, dtype=dtype)
        beta = torch.ones_like(alpha)
        provenance = torch.zeros(
            batch_size, num_queries, self.num_cameras, device=device, dtype=dtype
        )
        legacy_provenance = torch.zeros_like(provenance)
        source_evidence = torch.zeros_like(provenance)
        age = torch.zeros_like(alpha)
        previous_action = torch.full(
            (batch_size, num_queries),
            int(Action.DEFER),
            device=device,
            dtype=torch.long,
        )
        reference_feature = torch.zeros(
            batch_size,
            num_queries,
            self.feature_dim,
            device=device,
            dtype=dtype,
        )
        reference_geometry = torch.zeros(
            batch_size,
            num_queries,
            self.geometry_dim,
            device=device,
            dtype=dtype,
        )
        reference_class_distribution = torch.zeros(
            batch_size,
            num_queries,
            self.class_dim,
            device=device,
            dtype=dtype,
        )
        reference_ternary_distribution = torch.zeros(
            batch_size,
            num_queries,
            3,
            device=device,
            dtype=dtype,
        )
        reference_valid = torch.zeros(
            batch_size, num_queries, device=device, dtype=torch.bool
        )
        pre_gap_strength = torch.zeros_like(alpha)
        pre_gap_presence = torch.zeros_like(alpha)
        pre_gap_uncertainty = torch.zeros_like(alpha)
        pre_gap_source_evidence = torch.zeros_like(provenance)
        gap_active = torch.zeros_like(alpha, dtype=torch.bool)
        gap_age = torch.zeros_like(alpha)
        reacquisition_consumed = torch.zeros_like(alpha, dtype=torch.bool)

        if self.alpha is None or num_propagated <= 0:
            return {
                "alpha": alpha,
                "beta": beta,
                "source_evidence": source_evidence,
                "provenance": provenance,
                "legacy_provenance": legacy_provenance,
                "age": age,
                "previous_action": previous_action,
                "reference_feature": reference_feature,
                "reference_geometry": reference_geometry,
                "reference_class_distribution": reference_class_distribution,
                "reference_ternary_distribution": reference_ternary_distribution,
                "reference_valid": reference_valid,
                "pre_gap_strength": pre_gap_strength,
                "pre_gap_presence": pre_gap_presence,
                "pre_gap_uncertainty": pre_gap_uncertainty,
                "pre_gap_source_evidence": pre_gap_source_evidence,
                "gap_active": gap_active,
                "gap_age": gap_age,
                "reacquisition_consumed": reacquisition_consumed,
            }

        start = min(num_base_queries, num_queries)
        count = min(
            num_propagated,
            num_queries - start,
            self.alpha.shape[1],
        )
        if count > 0:
            alpha[:, start : start + count] = self.alpha[:, :count].to(dtype)
            beta[:, start : start + count] = self.beta[:, :count].to(dtype)
            source_evidence[:, start : start + count] = self.source_evidence[
                :, :count
            ].to(dtype)
            provenance[:, start : start + count] = self.provenance[:, :count].to(dtype)
            legacy_provenance[:, start : start + count] = self.legacy_provenance[
                :, :count
            ].to(dtype)
            age[:, start : start + count] = self.age[:, :count].to(dtype)
            previous_action[:, start : start + count] = self.action[:, :count]
            reference_feature[:, start : start + count] = self.reference_feature[
                :, :count
            ].to(dtype)
            reference_geometry[:, start : start + count] = self.reference_geometry[
                :, :count
            ].to(dtype)
            reference_class_distribution[
                :, start : start + count
            ] = self.reference_class_distribution[:, :count].to(dtype)
            reference_ternary_distribution[
                :, start : start + count
            ] = self.reference_ternary_distribution[:, :count].to(dtype)
            reference_valid[:, start : start + count] = self.reference_valid[
                :, :count
            ]
            pre_gap_strength[:, start : start + count] = self.pre_gap_strength[
                :, :count
            ].to(dtype)
            pre_gap_presence[:, start : start + count] = self.pre_gap_presence[
                :, :count
            ].to(dtype)
            pre_gap_uncertainty[:, start : start + count] = self.pre_gap_uncertainty[
                :, :count
            ].to(dtype)
            pre_gap_source_evidence[
                :, start : start + count
            ] = self.pre_gap_source_evidence[:, :count].to(dtype)
            gap_active[:, start : start + count] = self.gap_active[:, :count]
            gap_age[:, start : start + count] = self.gap_age[:, :count].to(dtype)
            reacquisition_consumed[
                :, start : start + count
            ] = self.reacquisition_consumed[:, :count]
        return {
            "alpha": alpha,
            "beta": beta,
            "source_evidence": source_evidence,
            "provenance": provenance,
            "legacy_provenance": legacy_provenance,
            "age": age,
            "previous_action": previous_action,
            "reference_feature": reference_feature,
            "reference_geometry": reference_geometry,
            "reference_class_distribution": reference_class_distribution,
            "reference_ternary_distribution": reference_ternary_distribution,
            "reference_valid": reference_valid,
            "pre_gap_strength": pre_gap_strength,
            "pre_gap_presence": pre_gap_presence,
            "pre_gap_uncertainty": pre_gap_uncertainty,
            "pre_gap_source_evidence": pre_gap_source_evidence,
            "gap_active": gap_active,
            "gap_age": gap_age,
            "reacquisition_consumed": reacquisition_consumed,
        }

    def _update_source_ledger(
        self,
        prior_source_evidence: Tensor,
        raw_source: Tensor,
        actual_added_evidence: Tensor,
    ) -> Dict[str, Tensor]:
        """Track source mass without feeding it back into S2.1 decisions."""
        output_dtype = prior_source_evidence.dtype
        half_compute = output_dtype == torch.float16
        if half_compute:
            prior_source_evidence = prior_source_evidence.float()
            raw_source = raw_source.float()
            actual_added_evidence = actual_added_evidence.float()

        prior_source_evidence = prior_source_evidence.clamp_min(0.0)
        raw_source = raw_source.clamp_min(0.0)
        source_sum = raw_source.sum(dim=-1, keepdim=True)
        current_distribution = torch.where(
            source_sum > self.novelty_eps,
            raw_source / source_sum.clamp_min(self.novelty_eps),
            torch.zeros_like(raw_source),
        )
        current_increment = (
            current_distribution * actual_added_evidence.clamp_min(0.0).unsqueeze(-1)
        )
        previous_source_strength = prior_source_evidence.sum(dim=-1)
        source_evidence = (
            self.source_decay * prior_source_evidence + current_increment
        )
        source_strength = source_evidence.sum(dim=-1)
        provenance = torch.where(
            source_strength.unsqueeze(-1) > self.novelty_eps,
            source_evidence
            / source_strength.unsqueeze(-1).clamp_min(self.novelty_eps),
            torch.zeros_like(source_evidence),
        )
        expected_strength = (
            self.source_decay * previous_source_strength
            + current_increment.sum(dim=-1)
        )
        residual = source_strength - expected_strength
        violation = residual.abs() > self.source_mass_tolerance
        zero_increment = current_increment.sum(dim=-1) <= self.novelty_eps
        output = {
            "current_source_vector": raw_source,
            "current_source_distribution": current_distribution,
            "current_source_increment": current_increment,
            "source_evidence": source_evidence,
            "source_strength": source_strength,
            "provenance": provenance,
            "previous_source_strength": previous_source_strength,
            "source_mass_residual": residual,
            "source_mass_violation": violation,
            "zero_source_increment": zero_increment,
        }
        if half_compute:
            output = {
                key: (
                    value.to(output_dtype)
                    if value.is_floating_point()
                    else value
                )
                for key, value in output.items()
            }
        return output

    def compute_novelty(
        self,
        current_source: Tensor,
        prior_source: Tensor,
        fresh_ratio: Tensor,
    ) -> Tensor:
        output_dtype = current_source.dtype
        cpu_half = (
            current_source.device.type == "cpu"
            and output_dtype == torch.float16
        )
        if cpu_half:
            current_source = current_source.float()
            prior_source = prior_source.float()
            fresh_ratio = fresh_ratio.float()
        current_norm = torch.linalg.vector_norm(current_source, dim=-1)
        prior_norm = torch.linalg.vector_norm(prior_source, dim=-1)
        cosine = (current_source * prior_source).sum(dim=-1) / (
            current_norm * prior_norm
        ).clamp_min(self.novelty_eps)
        cosine = cosine.clamp(0.0, 1.0)
        no_prior = prior_norm <= self.novelty_eps
        provenance_novelty = torch.where(no_prior, torch.ones_like(cosine), 1.0 - cosine)
        # A genuinely fresh frame is new evidence even when it comes from the
        # same camera. Repeated/lost frames rely on source-overlap discounting.
        novelty = fresh_ratio + (1.0 - fresh_ratio) * provenance_novelty
        novelty = novelty.clamp(0.0, 1.0)
        return novelty.to(output_dtype) if cpu_half else novelty

    def update_queries(
        self,
        ternary_probabilities: Tensor,
        observability: Tensor,
        source_vector: Tensor,
        fresh_ratio: Tensor,
        effective_count: Tensor,
        num_base_queries: int,
        num_propagated: int,
        use_strong_negative: bool = True,
        raw_source_vector: Optional[Tensor] = None,
        current_feature: Optional[Tensor] = None,
        current_geometry: Optional[Tensor] = None,
        current_class_probability: Optional[Tensor] = None,
        source_quality: Optional[Tensor] = None,
        camera_coverage: Optional[Tensor] = None,
        innovation_step: Optional[int] = None,
        delta_time: Optional[Tensor] = None,
        restoration_scale: float = 1.0,
    ) -> Dict[str, Tensor]:
        batch_size, num_queries, _ = ternary_probabilities.shape
        priors = self._query_priors(
            batch_size,
            num_queries,
            num_base_queries,
            num_propagated,
            ternary_probabilities.device,
            ternary_probabilities.dtype,
        )
        novelty_prior = (
            priors["legacy_provenance"]
            if self.enable_source_ledger
            else priors["provenance"]
        )
        novelty = self.compute_novelty(
            source_vector, novelty_prior, fresh_ratio
        )
        reference_dtype = ternary_probabilities.dtype
        reference_device = ternary_probabilities.device
        if current_feature is None:
            current_feature = torch.zeros(
                batch_size,
                num_queries,
                self.feature_dim,
                device=reference_device,
                dtype=reference_dtype,
            )
        if current_geometry is None:
            current_geometry = torch.zeros(
                batch_size,
                num_queries,
                self.geometry_dim,
                device=reference_device,
                dtype=reference_dtype,
            )
        if current_class_probability is None:
            current_class_probability = torch.zeros(
                batch_size,
                num_queries,
                self.class_dim,
                device=reference_device,
                dtype=reference_dtype,
            )
        if source_quality is None:
            source_quality = (
                source_vector.sum(dim=-1) > self.novelty_eps
            ).to(reference_dtype)
        if camera_coverage is None:
            source_for_coverage = (
                source_vector
                if raw_source_vector is None
                else raw_source_vector
            )
            camera_coverage = (
                source_for_coverage > self.novelty_eps
            ).sum(dim=-1).to(reference_dtype)

        has_prior = priors["reference_valid"]
        propagated_mask = torch.zeros_like(has_prior)
        start = min(num_base_queries, num_queries)
        count = min(
            num_propagated,
            max(num_queries - start, 0),
            self.memory_len,
        )
        if count > 0:
            propagated_mask[:, start : start + count] = True
        current_reference_valid = torch.isfinite(current_geometry).all(-1)
        valid_feature_pair = (
            has_prior
            & propagated_mask
            & (self.feature_dim > 0)
            & torch.isfinite(current_feature).all(-1)
        )
        valid_geometry = (
            has_prior
            & propagated_mask
            & current_reference_valid
            & torch.isfinite(priors["reference_geometry"]).all(-1)
        )
        reliable_observation = (
            observability
            >= self.innovation.reliable_observation_threshold
        ) & (
            (
                ternary_probabilities[..., 0]
                + ternary_probabilities[..., 1]
            )
            > self.novelty_eps
        )
        previous_total = priors["alpha"] + priors["beta"]
        if previous_total.device.type == "cpu" and previous_total.dtype == torch.float16:
            previous_presence = (
                priors["alpha"].float()
                / previous_total.float().clamp_min(self.novelty_eps)
            ).to(previous_total.dtype)
        else:
            previous_presence = (
                priors["alpha"] / previous_total.clamp_min(self.novelty_eps)
            )
        base_update = self.temporal_update(
            priors["alpha"],
            priors["beta"],
            ternary_probabilities[..., 0],
            ternary_probabilities[..., 1],
            observability,
            novelty,
            effective_count,
        )
        innovation_state: Dict[str, Tensor] = {}
        positive_factor = None
        negative_factor = None
        if self.innovation.mode != "off":
            class_current_for_innovation = (
                current_class_probability
                if self.class_dim
                else ternary_probabilities
            )
            class_previous_for_innovation = (
                priors["reference_class_distribution"]
                if self.class_dim
                else priors["reference_ternary_distribution"]
            )
            innovation_state = self.innovation(
                current_source=source_vector,
                previous_source=priors["provenance"],
                current_feature=current_feature,
                previous_feature=priors["reference_feature"],
                current_geometry=current_geometry,
                previous_geometry=priors["reference_geometry"],
                current_class_probability=class_current_for_innovation,
                previous_class_probability=class_previous_for_innovation,
                current_ternary_probability=ternary_probabilities,
                previous_ternary_probability=priors[
                    "reference_ternary_distribution"
                ],
                previous_age=priors["age"],
                previous_strength=previous_total - 2.0,
                previous_presence=previous_presence,
                has_prior=has_prior,
                valid_feature_pair=valid_feature_pair,
                valid_geometry=valid_geometry,
                observability=observability,
                source_quality=source_quality,
                reliable_observation=reliable_observation,
                camera_coverage=camera_coverage,
            )
            if self.innovation.mode == "active":
                target_positive = (
                    innovation_state["positive_reliability"]
                    * innovation_state["positive_novelty_gain"]
                )
                target_negative = (
                    innovation_state["negative_reliability"]
                    * innovation_state["negative_novelty_gain"]
                )
                if innovation_step is None:
                    transition = 1.0
                elif innovation_step < self.innovation_warmup_iters:
                    transition = 0.0
                elif self.innovation_transition_iters == 0:
                    transition = 1.0
                else:
                    transition = min(
                        max(
                            (
                                innovation_step
                                - self.innovation_warmup_iters
                            )
                            / self.innovation_transition_iters,
                            0.0,
                        ),
                        1.0,
                    )
                legacy_factor = observability * novelty
                positive_factor = (
                    (1.0 - transition) * legacy_factor
                    + transition * target_positive
                )
                negative_factor = (
                    (1.0 - transition) * legacy_factor
                    + transition * target_negative
                )
                innovation_state["innovation_transition"] = (
                    observability.new_full(observability.shape, transition)
                )
        update = base_update
        strategy = self.innovation.active_strategy
        if self.innovation.mode == "active" and strategy == "legacy_multiplicative":
            update = self.temporal_update(
                priors["alpha"],
                priors["beta"],
                ternary_probabilities[..., 0],
                ternary_probabilities[..., 1],
                observability,
                novelty,
                effective_count,
                positive_evidence_factor=positive_factor,
                negative_evidence_factor=negative_factor,
            )
        elif (
            self.innovation.mode == "active"
            and strategy == "residual_preserving"
        ):
            candidate = self.temporal_update(
                priors["alpha"],
                priors["beta"],
                ternary_probabilities[..., 0],
                ternary_probabilities[..., 1],
                observability,
                novelty,
                effective_count,
                positive_evidence_factor=positive_factor,
                negative_evidence_factor=negative_factor,
            )
            mix = self.innovation.residual_preserving_mix
            positive = (
                (1.0 - mix)
                * base_update["actual_added_positive_evidence"]
                + mix * candidate["actual_added_positive_evidence"]
            )
            update = self.temporal_update(
                priors["alpha"],
                priors["beta"],
                ternary_probabilities[..., 0],
                ternary_probabilities[..., 1],
                observability,
                novelty,
                effective_count,
                positive_evidence_override=positive,
                negative_evidence_override=base_update[
                    "actual_added_negative_evidence"
                ],
            )
            innovation_state["candidate_positive_evidence"] = candidate[
                "actual_added_positive_evidence"
            ]

        prior_strength = previous_total - 2.0
        cpu_half = (
            previous_total.device.type == "cpu"
            and previous_total.dtype == torch.float16
        )
        if cpu_half:
            prior_uncertainty = (
                2.0 / previous_total.float().clamp_min(self.novelty_eps)
            ).to(previous_total.dtype)
        else:
            prior_uncertainty = (
                2.0 / previous_total.clamp_min(self.novelty_eps)
            )
        start_gap = has_prior & (~priors["gap_active"]) & (~reliable_observation)
        first_recovery = (
            has_prior
            & priors["gap_active"]
            & (~priors["reacquisition_consumed"])
            & reliable_observation
        )
        stable_recovery = (
            has_prior
            & priors["gap_active"]
            & priors["reacquisition_consumed"]
            & reliable_observation
        )
        continuing_gap = priors["gap_active"] & (~reliable_observation)

        pre_gap_strength = torch.where(
            start_gap, prior_strength, priors["pre_gap_strength"]
        )
        pre_gap_presence = torch.where(
            start_gap, previous_presence, priors["pre_gap_presence"]
        )
        pre_gap_uncertainty = torch.where(
            start_gap, prior_uncertainty, priors["pre_gap_uncertainty"]
        )
        pre_gap_source_evidence = torch.where(
            start_gap.unsqueeze(-1),
            priors["source_evidence"],
            priors["pre_gap_source_evidence"],
        )
        gap_age = torch.where(
            start_gap,
            torch.ones_like(priors["gap_age"]),
            torch.where(
                continuing_gap,
                priors["gap_age"] + 1.0,
                priors["gap_age"],
            ),
        )
        gap_active = priors["gap_active"] | start_gap
        reacquisition_consumed = (
            priors["reacquisition_consumed"] | first_recovery
        )
        gap_active = gap_active & (~stable_recovery)
        gap_age = torch.where(
            stable_recovery, torch.zeros_like(gap_age), gap_age
        )
        reacquisition_consumed = (
            reacquisition_consumed & (~stable_recovery)
        )
        pre_gap_strength = torch.where(
            stable_recovery, update["strength"], pre_gap_strength
        )
        pre_gap_presence = torch.where(
            stable_recovery,
            update["existence_probability"],
            pre_gap_presence,
        )
        pre_gap_uncertainty = torch.where(
            stable_recovery, update["uncertainty"], pre_gap_uncertainty
        )

        lost_strength = (
            pre_gap_strength.float() - prior_strength.float()
        ).clamp_min(0.0)
        age_numerator = (
            gap_age.float()
            - float(self.innovation.minimum_gap_age)
            + 1.0
        ).clamp_min(0.0)
        age_factor = 1.0 - torch.exp(
            -age_numerator / self.innovation.reacquisition_time_tau
        )
        p_unobserved = ternary_probabilities[..., 2].float().clamp(0.0, 1.0)
        current_reliability = (
            reliable_observation.float()
            * observability.float().clamp(0.0, 1.0)
            * source_quality.float().clamp(0.0, 1.0)
            * (1.0 - p_unobserved)
        )
        if delta_time is None:
            delta_time = torch.zeros_like(observability)
        else:
            delta_time = delta_time.to(
                device=observability.device, dtype=observability.dtype
            )
            if delta_time.shape != observability.shape:
                raise ValueError("delta_time must share [B,Q] query layout")
        previous_geometry_float = priors["reference_geometry"].float()
        predicted_center = (
            previous_geometry_float[..., :3]
            + torch.cat(
                (
                    previous_geometry_float[..., 4:6],
                    torch.zeros_like(
                        previous_geometry_float[..., 4:5]
                    ),
                ),
                dim=-1,
            )
            * delta_time.float().abs().unsqueeze(-1)
        )
        reacquisition_center_residual = torch.linalg.vector_norm(
            current_geometry.float()[..., :3] - predicted_center, dim=-1
        )
        motion_consistency = torch.exp(
            -reacquisition_center_residual.square()
            / (2.0 * self.innovation.motion_sigma ** 2)
        )
        motion_consistency = torch.where(
            valid_geometry,
            motion_consistency,
            torch.zeros_like(motion_consistency),
        )
        if not self.innovation.use_motion_gate:
            motion_consistency = torch.ones_like(motion_consistency)

        current_source_for_recovery = (
            source_vector
            if raw_source_vector is None
            else raw_source_vector
        ).float().clamp_min(0.0)
        anchor_source = pre_gap_source_evidence.float().clamp_min(0.0)
        current_source_norm = torch.linalg.vector_norm(
            current_source_for_recovery.float(), dim=-1
        )
        anchor_source_norm = torch.linalg.vector_norm(
            anchor_source.float(), dim=-1
        )
        source_recovery = (
            (
                current_source_for_recovery.float()
                * anchor_source.float()
            ).sum(-1)
            / (
                current_source_norm * anchor_source_norm
            ).clamp_min(self.novelty_eps)
        ).clamp(0.0, 1.0)
        source_recovery = torch.where(
            (current_source_norm > self.novelty_eps)
            & (anchor_source_norm > self.novelty_eps),
            source_recovery,
            torch.zeros_like(source_recovery),
        )
        if not self.innovation.use_source_recovery_gate:
            source_recovery = torch.ones_like(source_recovery)

        reacquisition_gate = (
            first_recovery.float()
            * current_reliability
            * motion_consistency
            * source_recovery
            * pre_gap_presence.float().clamp(0.0, 1.0)
            * age_factor
        )
        base_positive = base_update["actual_added_positive_evidence"]
        base_negative = base_update["actual_added_negative_evidence"]
        restoration_budget = torch.minimum(
            self.innovation.restore_ratio * lost_strength,
            self.innovation.max_relative_bonus * base_positive.float(),
        )
        restoration_budget = torch.minimum(
            restoration_budget,
            torch.full_like(
                restoration_budget, self.innovation.max_absolute_bonus
            ),
        )
        restoration_bonus = (
            restoration_budget
            * reacquisition_gate
            * float(max(0.0, min(restoration_scale, 1.0)))
        )
        if (
            self.innovation.mode == "active"
            and strategy == "budgeted_reacquisition"
        ):
            update = self.temporal_update(
                priors["alpha"],
                priors["beta"],
                ternary_probabilities[..., 0],
                ternary_probabilities[..., 1],
                observability,
                novelty,
                effective_count,
                positive_evidence_override=base_positive + restoration_bonus,
                negative_evidence_override=base_negative,
            )
        else:
            restoration_bonus = torch.zeros_like(restoration_bonus)

        numerical_state = {
            "lost_strength": lost_strength,
            "age_factor": age_factor,
            "current_reliability": current_reliability,
            "predicted_center": predicted_center,
            "reacquisition_center_residual": reacquisition_center_residual,
            "motion_consistency": motion_consistency,
            "source_recovery": source_recovery,
            "reacquisition_gate": reacquisition_gate,
            "restoration_budget": restoration_budget,
            "restoration_bonus": restoration_bonus,
        }
        if reference_dtype == torch.float16:
            numerical_state = {
                key: value.to(reference_dtype)
                for key, value in numerical_state.items()
            }
        innovation_state.update(
            {
                "base_positive_evidence": base_positive,
                "base_negative_evidence": base_negative,
                **numerical_state,
                "is_reacquired": first_recovery,
            }
        )
        source_state: Dict[str, Tensor] = {}
        if self.enable_source_ledger:
            actual_added = (
                update["actual_added_positive_evidence"]
                + update["actual_added_negative_evidence"]
            )
            source_state = self._update_source_ledger(
                priors["source_evidence"],
                source_vector if raw_source_vector is None else raw_source_vector,
                actual_added,
            )
        observed_now = (
            update["actual_added_positive_evidence"]
            + update["actual_added_negative_evidence"]
        ) > self.observation_gate_threshold
        age = torch.where(
            observed_now, torch.zeros_like(priors["age"]), priors["age"] + 1.0
        )
        legacy_provenance = torch.where(
            observed_now.unsqueeze(-1),
            source_vector,
            novelty_prior,
        )
        if self.enable_source_ledger:
            provenance = source_state["provenance"]
        else:
            provenance = legacy_provenance
        decision = self.policy(
            observability,
            update["existence_probability"],
            update["uncertainty"],
            age,
            ternary_probabilities[..., 1],
            update["prior_strength"],
            use_strong_negative=use_strong_negative,
        )
        return {
            **update,
            **decision,
            **source_state,
            **innovation_state,
            "provenance": provenance,
            "legacy_provenance": legacy_provenance,
            "age": age,
            "observability": observability,
            "novelty": novelty,
            "effective_count": effective_count,
            "reference_feature": current_feature.detach(),
            "reference_geometry": current_geometry.detach(),
            "previous_reference_geometry": priors[
                "reference_geometry"
            ].detach(),
            "previous_source_vector": priors["provenance"].detach(),
            "previous_source_evidence": priors[
                "source_evidence"
            ].detach(),
            "prior_alpha": priors["alpha"],
            "prior_beta": priors["beta"],
            "previous_action": priors["previous_action"],
            "ternary_probabilities": ternary_probabilities.detach(),
            "reference_class_distribution": current_class_probability.detach(),
            "reference_ternary_distribution": ternary_probabilities.detach(),
            "reference_valid": current_reference_valid,
            "pre_gap_strength": pre_gap_strength,
            "pre_gap_presence": pre_gap_presence,
            "pre_gap_uncertainty": pre_gap_uncertainty,
            "pre_gap_source_evidence": pre_gap_source_evidence,
            "gap_active": gap_active,
            "gap_age": gap_age,
            "reacquisition_consumed": reacquisition_consumed,
        }

    def apply_reacquisition_control(
        self,
        query_state: Dict[str, Tensor],
        isolation_mask: Tensor,
        confirmed_mask: Optional[Tensor] = None,
        confirmation_bonus: Optional[Tensor] = None,
        confirmation_prior_alpha: Optional[Tensor] = None,
        confirmation_prior_beta: Optional[Tensor] = None,
        confirmation_prior_source_evidence: Optional[Tensor] = None,
        raw_source_vector: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Apply formal-ledger isolation and one-shot confirmed evidence.

        This is a second, deterministic ledger pass.  Candidate discovery is
        computed by the unchanged S2.3 logic; unconfirmed candidates are then
        replaced by a zero-addition decay update.  Confirmed candidates use
        their pending identity's stored prior and receive exactly one bonus.
        """
        isolation_mask = isolation_mask.bool()
        if isolation_mask.shape != query_state["alpha"].shape:
            raise ValueError("isolation_mask must share [B,Q] layout")
        confirmed_mask = (
            torch.zeros_like(isolation_mask)
            if confirmed_mask is None
            else confirmed_mask.bool()
        )
        if confirmed_mask.shape != isolation_mask.shape:
            raise ValueError("confirmed_mask must share [B,Q] layout")
        confirmed_mask = confirmed_mask & isolation_mask
        isolated_only = isolation_mask & (~confirmed_mask)
        if not isolation_mask.any():
            query_state["memory_isolation_mask"] = isolation_mask
            query_state["confirmed_reacquisition_mask"] = confirmed_mask
            return query_state

        reference = query_state["alpha"]
        zeros = torch.zeros_like(reference)
        confirmation_bonus = (
            zeros
            if confirmation_bonus is None
            else confirmation_bonus.to(reference)
        )
        prior_alpha = query_state["prior_alpha"]
        prior_beta = query_state["prior_beta"]
        if confirmation_prior_alpha is not None:
            prior_alpha = torch.where(
                confirmed_mask,
                confirmation_prior_alpha.to(reference),
                prior_alpha,
            )
        if confirmation_prior_beta is not None:
            prior_beta = torch.where(
                confirmed_mask,
                confirmation_prior_beta.to(reference),
                prior_beta,
            )
        positive = torch.where(
            confirmed_mask,
            query_state["base_positive_evidence"] + confirmation_bonus,
            zeros,
        )
        negative = torch.where(
            confirmed_mask,
            query_state["base_negative_evidence"],
            zeros,
        )
        ternary = query_state["ternary_probabilities"]
        controlled_update = self.temporal_update(
            prior_alpha,
            prior_beta,
            ternary[..., 0],
            ternary[..., 1],
            query_state["observability"],
            query_state["novelty"],
            query_state["effective_count"],
            positive_evidence_override=positive,
            negative_evidence_override=negative,
        )
        for key, value in controlled_update.items():
            if key in query_state:
                query_state[key] = torch.where(
                    isolation_mask, value, query_state[key]
                )

        if self.enable_source_ledger:
            prior_source = query_state["previous_source_evidence"]
            if confirmation_prior_source_evidence is not None:
                prior_source = torch.where(
                    confirmed_mask.unsqueeze(-1),
                    confirmation_prior_source_evidence.to(prior_source),
                    prior_source,
                )
            source = (
                query_state["current_source_vector"]
                if raw_source_vector is None
                else raw_source_vector
            )
            controlled_source = self._update_source_ledger(
                prior_source,
                source,
                controlled_update["actual_added_positive_evidence"]
                + controlled_update["actual_added_negative_evidence"],
            )
            for key, value in controlled_source.items():
                if key in query_state:
                    expand_mask = isolation_mask
                    while expand_mask.ndim < value.ndim:
                        expand_mask = expand_mask.unsqueeze(-1)
                    query_state[key] = torch.where(
                        expand_mask, value, query_state[key]
                    )
            query_state["provenance"] = query_state["source_evidence"] / (
                query_state["source_strength"].unsqueeze(-1).clamp_min(
                    self.novelty_eps
                )
            )
            query_state["provenance"] = torch.where(
                query_state["source_strength"].unsqueeze(-1)
                > self.novelty_eps,
                query_state["provenance"],
                torch.zeros_like(query_state["provenance"]),
            )

        controlled_age = torch.where(
            confirmed_mask,
            torch.zeros_like(query_state["age"]),
            query_state["age"] + 1,
        )
        query_state["age"] = torch.where(
            isolation_mask, controlled_age, query_state["age"]
        )
        decision = self.policy(
            query_state["observability"],
            query_state["existence_probability"],
            query_state["uncertainty"],
            query_state["age"],
            ternary[..., 1],
            query_state["prior_strength"],
        )
        for key in ("action", "score_scale", "write_mask"):
            query_state[key] = torch.where(
                isolation_mask, decision[key], query_state[key]
            )
        query_state["action"] = torch.where(
            isolated_only,
            torch.full_like(
                query_state["action"], int(Action.DEFER)
            ),
            query_state["action"],
        )
        query_state["score_scale"] = torch.where(
            isolated_only,
            torch.full_like(
                query_state["score_scale"],
                self.policy.defer_score_scale,
            ),
            query_state["score_scale"],
        )
        query_state["write_mask"] = torch.where(
            isolated_only,
            torch.zeros_like(query_state["write_mask"]),
            query_state["write_mask"],
        )
        # A confirmed identity has already met the cross-frame admission gate.
        query_state["action"] = torch.where(
            confirmed_mask,
            torch.full_like(
                query_state["action"], int(Action.RECOVER)
            ),
            query_state["action"],
        )
        query_state["score_scale"] = torch.where(
            confirmed_mask,
            torch.full_like(
                query_state["score_scale"],
                self.policy.recover_score_scale,
            ),
            query_state["score_scale"],
        )
        query_state["write_mask"] = (
            query_state["write_mask"] | confirmed_mask
        )
        query_state["restoration_bonus"] = torch.where(
            isolation_mask,
            torch.where(confirmed_mask, confirmation_bonus, zeros),
            query_state["restoration_bonus"],
        )
        query_state["reacquisition_consumed"] = (
            query_state["reacquisition_consumed"] | confirmed_mask
        )
        query_state["gap_active"] = (
            query_state["gap_active"] & (~confirmed_mask)
        )
        query_state["memory_isolation_mask"] = isolation_mask
        query_state["confirmed_reacquisition_mask"] = confirmed_mask
        return query_state

    def commit_topk(
        self,
        query_state: Dict[str, Tensor],
        topk_indexes: Tensor,
        valid_write_mask: Optional[Tensor] = None,
    ) -> None:
        if self.alpha is None:
            raise RuntimeError("pre_update must be called before commit_topk")
        alpha = _gather(query_state["alpha"], topk_indexes)
        beta = _gather(query_state["beta"], topk_indexes)
        source_evidence = _gather(
            query_state.get(
                "source_evidence",
                torch.zeros_like(query_state["provenance"]),
            ),
            topk_indexes,
        )
        provenance = _gather(query_state["provenance"], topk_indexes)
        legacy_provenance = _gather(
            query_state["legacy_provenance"], topk_indexes
        )
        age = _gather(query_state["age"], topk_indexes)
        effective_count = _gather(query_state["effective_count"], topk_indexes)
        observability = _gather(query_state["observability"], topk_indexes)
        novelty = _gather(query_state["novelty"], topk_indexes)
        action = _gather(query_state["action"], topk_indexes)
        reference_feature = _gather(
            query_state["reference_feature"], topk_indexes
        )
        reference_geometry = _gather(
            query_state["reference_geometry"], topk_indexes
        )
        reference_class_distribution = _gather(
            query_state["reference_class_distribution"], topk_indexes
        )
        reference_ternary_distribution = _gather(
            query_state["reference_ternary_distribution"], topk_indexes
        )
        reference_valid = _gather(
            query_state["reference_valid"], topk_indexes
        ).bool()
        pre_gap_strength = _gather(
            query_state["pre_gap_strength"], topk_indexes
        )
        pre_gap_presence = _gather(
            query_state["pre_gap_presence"], topk_indexes
        )
        pre_gap_uncertainty = _gather(
            query_state["pre_gap_uncertainty"], topk_indexes
        )
        pre_gap_source_evidence = _gather(
            query_state["pre_gap_source_evidence"], topk_indexes
        )
        gap_active = _gather(
            query_state["gap_active"], topk_indexes
        ).bool()
        gap_age = _gather(query_state["gap_age"], topk_indexes)
        reacquisition_consumed = _gather(
            query_state["reacquisition_consumed"], topk_indexes
        ).bool()

        if valid_write_mask is None:
            valid = torch.ones_like(alpha, dtype=torch.bool)
        else:
            valid = _gather(valid_write_mask, topk_indexes).bool()
        alpha = torch.where(valid, alpha, torch.ones_like(alpha))
        beta = torch.where(valid, beta, torch.ones_like(beta))
        source_evidence = torch.where(
            valid.unsqueeze(-1),
            source_evidence,
            torch.zeros_like(source_evidence),
        )
        provenance = torch.where(valid.unsqueeze(-1), provenance, torch.zeros_like(provenance))
        legacy_provenance = torch.where(
            valid.unsqueeze(-1),
            legacy_provenance,
            torch.zeros_like(legacy_provenance),
        )
        age = torch.where(valid, age, torch.zeros_like(age))
        effective_count = torch.where(valid, effective_count, torch.zeros_like(effective_count))
        observability = torch.where(valid, observability, torch.zeros_like(observability))
        novelty = torch.where(valid, novelty, torch.zeros_like(novelty))
        action = torch.where(
            valid,
            action,
            torch.full_like(action, int(Action.DEFER)),
        )
        reference_feature = torch.where(
            valid.unsqueeze(-1),
            reference_feature,
            torch.zeros_like(reference_feature),
        )
        reference_geometry = torch.where(
            valid.unsqueeze(-1),
            reference_geometry,
            torch.zeros_like(reference_geometry),
        )
        reference_class_distribution = torch.where(
            valid.unsqueeze(-1),
            reference_class_distribution,
            torch.zeros_like(reference_class_distribution),
        )
        reference_ternary_distribution = torch.where(
            valid.unsqueeze(-1),
            reference_ternary_distribution,
            torch.zeros_like(reference_ternary_distribution),
        )
        reference_valid = reference_valid & valid
        pre_gap_strength = torch.where(
            valid, pre_gap_strength, torch.zeros_like(pre_gap_strength)
        )
        pre_gap_presence = torch.where(
            valid, pre_gap_presence, torch.zeros_like(pre_gap_presence)
        )
        pre_gap_uncertainty = torch.where(
            valid, pre_gap_uncertainty, torch.zeros_like(pre_gap_uncertainty)
        )
        pre_gap_source_evidence = torch.where(
            valid.unsqueeze(-1),
            pre_gap_source_evidence,
            torch.zeros_like(pre_gap_source_evidence),
        )
        gap_active = gap_active & valid
        gap_age = torch.where(valid, gap_age, torch.zeros_like(gap_age))
        reacquisition_consumed = reacquisition_consumed & valid

        self.alpha = torch.cat((alpha.detach(), self.alpha), dim=1)
        self.beta = torch.cat((beta.detach(), self.beta), dim=1)
        self.source_evidence = torch.cat(
            (source_evidence.detach(), self.source_evidence), dim=1
        )
        self.provenance = torch.cat((provenance.detach(), self.provenance), dim=1)
        self.legacy_provenance = torch.cat(
            (legacy_provenance.detach(), self.legacy_provenance), dim=1
        )
        self.age = torch.cat((age.detach(), self.age), dim=1)
        self.effective_count = torch.cat((effective_count.detach(), self.effective_count), dim=1)
        self.observability = torch.cat((observability.detach(), self.observability), dim=1)
        self.novelty = torch.cat((novelty.detach(), self.novelty), dim=1)
        self.action = torch.cat((action.detach(), self.action), dim=1)
        self.reference_feature = torch.cat(
            (reference_feature.detach(), self.reference_feature), dim=1
        )
        self.reference_geometry = torch.cat(
            (reference_geometry.detach(), self.reference_geometry), dim=1
        )
        self.reference_class_distribution = torch.cat(
            (
                reference_class_distribution.detach(),
                self.reference_class_distribution,
            ),
            dim=1,
        )
        self.reference_ternary_distribution = torch.cat(
            (
                reference_ternary_distribution.detach(),
                self.reference_ternary_distribution,
            ),
            dim=1,
        )
        self.reference_valid = torch.cat(
            (reference_valid.detach(), self.reference_valid), dim=1
        )
        self.pre_gap_strength = torch.cat(
            (pre_gap_strength.detach(), self.pre_gap_strength), dim=1
        )
        self.pre_gap_presence = torch.cat(
            (pre_gap_presence.detach(), self.pre_gap_presence), dim=1
        )
        self.pre_gap_uncertainty = torch.cat(
            (pre_gap_uncertainty.detach(), self.pre_gap_uncertainty), dim=1
        )
        self.pre_gap_source_evidence = torch.cat(
            (
                pre_gap_source_evidence.detach(),
                self.pre_gap_source_evidence,
            ),
            dim=1,
        )
        self.gap_active = torch.cat(
            (gap_active.detach(), self.gap_active), dim=1
        )
        self.gap_age = torch.cat(
            (gap_age.detach(), self.gap_age), dim=1
        )
        self.reacquisition_consumed = torch.cat(
            (
                reacquisition_consumed.detach(),
                self.reacquisition_consumed,
            ),
            dim=1,
        )

    @staticmethod
    def summarize(query_state: Dict[str, Tensor]) -> Dict[str, float]:
        action = query_state["action"]
        query_count = max(int(query_state["strength"].numel()), 1)
        violation_count = int(
            query_state["conservation_violation_mask"].sum().detach().cpu()
        )
        unsupported_count = int(
            query_state["unsupported_growth"].sum().detach().cpu()
        )
        summary = {
            "mean_observability": float(query_state["observability"].detach().mean().cpu()),
            "mean_novelty": float(query_state["novelty"].detach().mean().cpu()),
            "mean_uncertainty": float(query_state["uncertainty"].detach().mean().cpu()),
            "mean_existence": float(query_state["existence_probability"].detach().mean().cpu()),
            "max_evidence_inflation": float(
                query_state["conservation_ratio"].detach().max().cpu()
            ),
            "max_conservation_violation": float(
                query_state["conservation_violation"].detach().max().cpu()
            ),
            "max_abs_conservation_residual": float(
                query_state["conservation_residual"].detach().abs().max().cpu()
            ),
            "conservation_residual_mean": float(
                query_state["conservation_residual"].detach().mean().cpu()
            ),
            "conservation_residual_abs_max": float(
                query_state["conservation_residual"].detach().abs().max().cpu()
            ),
            "conservation_violation_count": violation_count,
            "conservation_violation_ratio": violation_count / query_count,
            "unsupported_growth_count": unsupported_count,
            "unsupported_growth_ratio": unsupported_count / query_count,
            "keep_count": int((action == int(Action.KEEP)).sum().detach().cpu()),
            "recover_count": int((action == int(Action.RECOVER)).sum().detach().cpu()),
            "defer_count": int((action == int(Action.DEFER)).sum().detach().cpu()),
        }
        if "source_mass_residual" in query_state:
            source_residual = query_state["source_mass_residual"]
            source_violation = query_state["source_mass_violation"]
            zero_increment = query_state["zero_source_increment"]
            source_count = max(int(source_residual.numel()), 1)
            source_violation_count = int(
                source_violation.sum().detach().cpu()
            )
            zero_increment_count = int(
                zero_increment.sum().detach().cpu()
            )
            summary.update(
                {
                    "source_mass_residual_mean": float(
                        source_residual.detach().mean().cpu()
                    ),
                    "source_mass_residual_abs_max": float(
                        source_residual.detach().abs().max().cpu()
                    ),
                    "source_mass_violation_count": source_violation_count,
                    "source_mass_violation_ratio": (
                        source_violation_count / source_count
                    ),
                    "zero_source_increment_count": zero_increment_count,
                    "zero_source_increment_ratio": (
                        zero_increment_count / source_count
                    ),
                }
            )
        if "novelty_gain" in query_state:
            for key in (
                "source_innovation",
                "feature_innovation",
                "geometry_innovation",
                "semantic_innovation",
                "temporal_reacquisition",
                "combined_reliability",
                "conflict",
                "novelty_gain",
                "positive_novelty_gain",
                "negative_novelty_gain",
            ):
                summary[f"{key}_mean"] = float(
                    query_state[key].detach().mean().cpu()
                )
            summary["reacquired_query_count"] = int(
                query_state["is_reacquired_query"].sum().detach().cpu()
            )
        return summary
