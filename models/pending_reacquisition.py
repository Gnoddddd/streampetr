"""Scene-local pending identities for confirmed temporal reacquisition."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


class PendingReacquisitionTracker(nn.Module):
    """Match reacquisition candidates across frames without query-index identity.

    The tracker is deliberately non-learned and GT-free.  Every runtime tensor
    is a non-persistent buffer, so it follows ``model.to(device)`` but is absent
    from ordinary training checkpoints.
    """

    _STATE_NAMES = (
        "active",
        "runtime_id",
        "predicted_class",
        "center",
        "velocity",
        "timestamp",
        "age",
        "streak",
        "proposed_bonus",
        "prior_alpha",
        "prior_beta",
        "prior_source_evidence",
        "query_source",
        "next_runtime_id",
    )

    def __init__(
        self,
        capacity: int,
        num_sources: int,
        confirmation_frames: int = 2,
        pending_max_age: int = 3,
        class_consistency_required: bool = True,
        center_distance_threshold: float = 2.0,
        motion_distance_threshold: float = 2.0,
        minimum_confirmation_score: float = 0.075,
        minimum_confirmation_reliability: float = 0.65,
        enable_confirmation: bool = True,
    ) -> None:
        super().__init__()
        self.capacity = max(int(capacity), 1)
        self.num_sources = int(num_sources)
        self.confirmation_frames = max(int(confirmation_frames), 2)
        self.pending_max_age = max(int(pending_max_age), 1)
        self.class_consistency_required = bool(class_consistency_required)
        self.center_distance_threshold = float(center_distance_threshold)
        self.motion_distance_threshold = float(motion_distance_threshold)
        self.minimum_confirmation_score = float(minimum_confirmation_score)
        self.minimum_confirmation_reliability = float(
            minimum_confirmation_reliability
        )
        self.enable_confirmation = bool(enable_confirmation)
        if (
            self.center_distance_threshold <= 0.0
            or self.motion_distance_threshold <= 0.0
        ):
            raise ValueError("pending spatial thresholds must be positive")
        for name in self._STATE_NAMES:
            self.register_buffer(name, None, persistent=False)
        self._scene_tokens: Optional[Tuple[str, ...]] = None
        self._register_state_dict_hook(self._strip_runtime_checkpoint_state)

    @staticmethod
    def _strip_runtime_checkpoint_state(
        module: nn.Module,
        destination: Dict[str, Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
    ) -> None:
        del module, local_metadata
        for name in PendingReacquisitionTracker._STATE_NAMES:
            destination.pop(prefix + name, None)

    def reset(self) -> None:
        for name in self._STATE_NAMES:
            setattr(self, name, None)
        self._scene_tokens = None

    def _initialize(self, reference: Tensor, batch_size: int) -> None:
        device = reference.device
        dtype = reference.dtype
        shape = (batch_size, self.capacity)
        self.active = torch.zeros(shape, device=device, dtype=torch.bool)
        self.runtime_id = torch.full(
            shape, -1, device=device, dtype=torch.long
        )
        self.predicted_class = torch.full(
            shape, -1, device=device, dtype=torch.long
        )
        self.center = torch.zeros(
            batch_size, self.capacity, 3, device=device, dtype=dtype
        )
        self.velocity = torch.zeros(
            batch_size, self.capacity, 2, device=device, dtype=dtype
        )
        self.timestamp = torch.zeros(shape, device=device, dtype=torch.float64)
        self.age = torch.zeros(shape, device=device, dtype=torch.long)
        self.streak = torch.zeros(shape, device=device, dtype=torch.long)
        self.proposed_bonus = torch.zeros(shape, device=device, dtype=dtype)
        self.prior_alpha = torch.ones(shape, device=device, dtype=dtype)
        self.prior_beta = torch.ones(shape, device=device, dtype=dtype)
        self.prior_source_evidence = torch.zeros(
            batch_size,
            self.capacity,
            self.num_sources,
            device=device,
            dtype=dtype,
        )
        self.query_source = torch.zeros(shape, device=device, dtype=torch.long)
        self.next_runtime_id = torch.zeros(
            (), device=device, dtype=torch.long
        )

    def _clear_rows(self, rows: Tensor) -> None:
        if not rows.any():
            return
        self.active[rows] = False
        self.runtime_id[rows] = -1
        self.predicted_class[rows] = -1
        self.center[rows] = 0
        self.velocity[rows] = 0
        self.timestamp[rows] = 0
        self.age[rows] = 0
        self.streak[rows] = 0
        self.proposed_bonus[rows] = 0
        self.prior_alpha[rows] = 1
        self.prior_beta[rows] = 1
        self.prior_source_evidence[rows] = 0
        self.query_source[rows] = 0

    def pre_update(
        self,
        reference: Tensor,
        scene_tokens: Sequence[str],
        reset_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Initialize state and clear only batch rows whose scene changed."""
        batch_size = reference.shape[0]
        tokens = tuple(str(token) for token in scene_tokens)
        if len(tokens) != batch_size:
            raise ValueError("scene token count must match batch size")
        if self.active is None or self.active.shape[0] != batch_size:
            self._initialize(reference, batch_size)
            self._scene_tokens = tokens
            return torch.ones(
                batch_size, device=reference.device, dtype=torch.bool
            )
        previous = self._scene_tokens or tuple("" for _ in tokens)
        changed = torch.tensor(
            [left != right for left, right in zip(previous, tokens)],
            device=reference.device,
            dtype=torch.bool,
        )
        if reset_mask is not None:
            changed = changed | reset_mask.to(
                device=reference.device, dtype=torch.bool
            ).reshape(batch_size)
        self._clear_rows(changed)
        self._scene_tokens = tokens
        return changed

    @staticmethod
    def _distance(left: Tensor, right: Tensor) -> Tensor:
        values = left.float() - right.float()
        return torch.linalg.vector_norm(values, dim=-1)

    def _reject_slot(self, batch: int, slot: int) -> int:
        runtime_id = int(self.runtime_id[batch, slot].item())
        self.active[batch, slot] = False
        return runtime_id

    def step(
        self,
        *,
        scene_tokens: Sequence[str],
        seed_mask: Tensor,
        predicted_class: Tensor,
        center: Tensor,
        velocity: Tensor,
        score: Tensor,
        reliability: Tensor,
        proposed_bonus: Tensor,
        prior_alpha: Tensor,
        prior_beta: Tensor,
        prior_source_evidence: Tensor,
        timestamp: Tensor,
        query_source: Tensor,
        reset_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Match pending identities, add new seeds, and expose ready queries."""
        if seed_mask.ndim != 2:
            raise ValueError("seed_mask must use [B,Q] layout")
        batch_size, num_queries = seed_mask.shape
        if center.shape != (batch_size, num_queries, 3):
            raise ValueError("center must use [B,Q,3] layout")
        self.pre_update(center, scene_tokens, reset_mask=reset_mask)
        timestamp = timestamp.reshape(batch_size).to(
            device=center.device, dtype=torch.float64
        )
        inputs = (
            predicted_class,
            center,
            velocity,
            score,
            reliability,
            proposed_bonus,
            prior_alpha,
            prior_beta,
            prior_source_evidence,
            query_source,
        )
        (
            predicted_class,
            center,
            velocity,
            score,
            reliability,
            proposed_bonus,
            prior_alpha,
            prior_beta,
            prior_source_evidence,
            query_source,
        ) = tuple(value.detach() for value in inputs)
        center = torch.nan_to_num(center)
        velocity = torch.nan_to_num(velocity)
        score = torch.nan_to_num(score)
        reliability = torch.nan_to_num(reliability)
        proposed_bonus = torch.nan_to_num(proposed_bonus)

        pending_mask = torch.zeros_like(seed_mask, dtype=torch.bool)
        ready_mask = torch.zeros_like(seed_mask, dtype=torch.bool)
        rejected_query_mask = torch.zeros_like(seed_mask, dtype=torch.bool)
        slot_for_query = torch.full_like(seed_mask, -1, dtype=torch.long)
        runtime_id_for_query = torch.full_like(
            seed_mask, -1, dtype=torch.long
        )
        confirmation_bonus = torch.zeros_like(score)
        confirmation_prior_alpha = torch.ones_like(prior_alpha)
        confirmation_prior_beta = torch.ones_like(prior_beta)
        confirmation_prior_source = torch.zeros_like(prior_source_evidence)
        candidate_created = torch.zeros_like(seed_mask, dtype=torch.bool)
        rejected_count = torch.zeros(
            batch_size, device=center.device, dtype=torch.long
        )
        expired_count = torch.zeros_like(rejected_count)

        for batch in range(batch_size):
            used_queries = torch.zeros(
                num_queries, device=center.device, dtype=torch.bool
            )
            active_slots = self.active[batch].nonzero(
                as_tuple=False
            ).flatten().tolist()
            for slot in active_slots:
                self.age[batch, slot] += 1
                if int(self.age[batch, slot].item()) > self.pending_max_age:
                    self._reject_slot(batch, slot)
                    expired_count[batch] += 1
                    continue

                delta_time = (
                    timestamp[batch] - self.timestamp[batch, slot]
                ).float().clamp_min(0.0)
                predicted_center = self.center[batch, slot].clone()
                predicted_center[:2] = (
                    predicted_center[:2]
                    + self.velocity[batch, slot] * delta_time
                )
                center_distance = self._distance(
                    center[batch], self.center[batch, slot]
                )
                motion_distance = self._distance(
                    center[batch], predicted_center
                )
                class_ok = (
                    predicted_class[batch]
                    == self.predicted_class[batch, slot]
                )
                score_ok = (
                    score[batch] >= self.minimum_confirmation_score
                )
                reliability_ok = (
                    reliability[batch]
                    >= self.minimum_confirmation_reliability
                )
                available = ~used_queries
                spatial_ok = (
                    center_distance <= self.center_distance_threshold
                ) & (
                    motion_distance <= self.motion_distance_threshold
                )
                valid = available & spatial_ok & score_ok & reliability_ok
                if self.class_consistency_required:
                    valid = valid & class_ok

                if valid.any():
                    cost = center_distance + motion_distance
                    cost = torch.where(
                        valid, cost, torch.full_like(cost, float("inf"))
                    )
                    query = int(cost.argmin().item())
                    used_queries[query] = True
                    pending_mask[batch, query] = True
                    slot_for_query[batch, query] = slot
                    runtime_id_for_query[batch, query] = self.runtime_id[
                        batch, slot
                    ]
                    self.streak[batch, slot] += 1
                    self.center[batch, slot] = center[batch, query]
                    self.velocity[batch, slot] = velocity[batch, query]
                    self.timestamp[batch, slot] = timestamp[batch]
                    if (
                        self.enable_confirmation
                        and int(self.streak[batch, slot].item())
                        >= self.confirmation_frames
                    ):
                        ready_mask[batch, query] = True
                        confirmation_bonus[batch, query] = (
                            self.proposed_bonus[batch, slot]
                        )
                        confirmation_prior_alpha[batch, query] = (
                            self.prior_alpha[batch, slot]
                        )
                        confirmation_prior_beta[batch, query] = (
                            self.prior_beta[batch, slot]
                        )
                        confirmation_prior_source[batch, query] = (
                            self.prior_source_evidence[batch, slot]
                        )
                    continue

                # A geometrically nearby query that fails class/quality is a
                # rejected identity. A same-class discontinuity is also
                # rejected instead of being allowed to bind after reordering.
                available_indexes = available.nonzero(
                    as_tuple=False
                ).flatten()
                if available_indexes.numel() == 0:
                    continue
                nearest = int(
                    motion_distance[available_indexes].argmin().item()
                )
                query = int(available_indexes[nearest].item())
                rejected_query_mask[batch, query] = True
                used_queries[query] = True
                self._reject_slot(batch, slot)
                rejected_count[batch] += 1

            for query in seed_mask[batch].nonzero(
                as_tuple=False
            ).flatten().tolist():
                if used_queries[query]:
                    continue
                free = (~self.active[batch]).nonzero(
                    as_tuple=False
                ).flatten()
                if free.numel() == 0:
                    # Deterministically expire the oldest slot.
                    slot = int(self.age[batch].argmax().item())
                    self._reject_slot(batch, slot)
                    expired_count[batch] += 1
                else:
                    slot = int(free[0].item())
                runtime_id = int(self.next_runtime_id.item())
                self.next_runtime_id.add_(1)
                self.active[batch, slot] = True
                self.runtime_id[batch, slot] = runtime_id
                self.predicted_class[batch, slot] = predicted_class[
                    batch, query
                ]
                self.center[batch, slot] = center[batch, query]
                self.velocity[batch, slot] = velocity[batch, query]
                self.timestamp[batch, slot] = timestamp[batch]
                self.age[batch, slot] = 0
                self.streak[batch, slot] = 1
                self.proposed_bonus[batch, slot] = proposed_bonus[
                    batch, query
                ]
                self.prior_alpha[batch, slot] = prior_alpha[batch, query]
                self.prior_beta[batch, slot] = prior_beta[batch, query]
                self.prior_source_evidence[batch, slot] = (
                    prior_source_evidence[batch, query]
                )
                self.query_source[batch, slot] = query_source[batch, query]
                pending_mask[batch, query] = True
                slot_for_query[batch, query] = slot
                runtime_id_for_query[batch, query] = runtime_id
                candidate_created[batch, query] = True
                used_queries[query] = True

        return {
            "candidate_mask": seed_mask.bool(),
            "candidate_created": candidate_created,
            "pending_mask": pending_mask,
            "confirmation_ready_mask": ready_mask,
            "rejected_query_mask": rejected_query_mask,
            "isolation_mask": pending_mask | rejected_query_mask,
            "slot_for_query": slot_for_query,
            "runtime_id": runtime_id_for_query,
            "confirmation_bonus": confirmation_bonus,
            "confirmation_prior_alpha": confirmation_prior_alpha,
            "confirmation_prior_beta": confirmation_prior_beta,
            "confirmation_prior_source_evidence": confirmation_prior_source,
            "rejected_count": rejected_count,
            "expired_count": expired_count,
            "active_pending_count": self.active.sum(dim=1),
        }

    def finalize(
        self,
        confirmed_mask: Tensor,
        slot_for_query: Tensor,
    ) -> Tensor:
        """Consume each confirmed identity once and return its runtime IDs."""
        confirmed_ids = torch.full_like(
            slot_for_query, -1, dtype=torch.long
        )
        for batch, query in confirmed_mask.nonzero(
            as_tuple=False
        ).tolist():
            slot = int(slot_for_query[batch, query].item())
            if slot < 0 or not bool(self.active[batch, slot]):
                continue
            confirmed_ids[batch, query] = self.runtime_id[batch, slot]
            self.active[batch, slot] = False
        return confirmed_ids

    def export_runtime_state(self, to_cpu: bool = True) -> Dict[str, Any]:
        buffers: Dict[str, Optional[Tensor]] = {}
        for name in self._STATE_NAMES:
            value = getattr(self, name)
            if value is None:
                buffers[name] = None
            else:
                clone = value.detach().clone()
                buffers[name] = clone.cpu() if to_cpu else clone
        return {
            "version": 1,
            "capacity": self.capacity,
            "num_sources": self.num_sources,
            "scene_tokens": self._scene_tokens,
            "buffers": buffers,
        }

    def load_runtime_state(
        self,
        state: Dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> None:
        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported pending runtime-state version")
        if int(state.get("capacity", -1)) != self.capacity:
            raise ValueError("pending capacity mismatch")
        if int(state.get("num_sources", -1)) != self.num_sources:
            raise ValueError("pending source dimension mismatch")
        buffers = state.get("buffers")
        if not isinstance(buffers, dict):
            raise ValueError("pending runtime state requires buffers")
        for name in self._STATE_NAMES:
            value = buffers.get(name)
            if value is not None:
                value = value.detach().clone()
                if device is not None:
                    value = value.to(device)
            setattr(self, name, value)
        tokens = state.get("scene_tokens")
        self._scene_tokens = (
            tuple(str(token) for token in tokens)
            if tokens is not None
            else None
        )
