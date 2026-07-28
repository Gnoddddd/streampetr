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
            "version": 3,
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
        if version not in (1, 2, 3):
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
        if version == 3:
            if int(state.get("feature_dim", -1)) != self.feature_dim:
                raise ValueError("runtime feature_dim does not match this ledger")
            if int(state.get("class_dim", -1)) != self.class_dim:
                raise ValueError("runtime class_dim does not match this ledger")
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
            elif name == "reference_valid":
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

        if self.alpha is None or num_propagated <= 0:
            return {
                "alpha": alpha,
                "beta": beta,
                "source_evidence": source_evidence,
                "provenance": provenance,
                "legacy_provenance": legacy_provenance,
                "age": age,
                "reference_feature": reference_feature,
                "reference_geometry": reference_geometry,
                "reference_class_distribution": reference_class_distribution,
                "reference_ternary_distribution": reference_ternary_distribution,
                "reference_valid": reference_valid,
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
        return {
            "alpha": alpha,
            "beta": beta,
            "source_evidence": source_evidence,
            "provenance": provenance,
            "legacy_provenance": legacy_provenance,
            "age": age,
            "reference_feature": reference_feature,
            "reference_geometry": reference_geometry,
            "reference_class_distribution": reference_class_distribution,
            "reference_ternary_distribution": reference_ternary_distribution,
            "reference_valid": reference_valid,
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
            "reference_class_distribution": current_class_probability.detach(),
            "reference_ternary_distribution": ternary_probabilities.detach(),
            "reference_valid": current_reference_valid,
        }

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
