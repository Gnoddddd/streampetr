"""StreamPETR integration for evidence-conserving 3D detection.

This module is imported only inside the dedicated ``streampetr`` environment.
It subclasses the pinned official StreamPETR head and changes exactly the two
critical points identified in the research plan:

1. Hungarian-unmatched queries are background only in observable regions.
2. Temporal memory writes are governed by provenance-aware Beta evidence.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmdet.models import HEADS

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import StreamPETRHead
from projects.mmdet3d_plugin.models.utils.misc import (
    topk_gather,
    transform_reference_points,
)
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d
from mmdet.models.utils.transformer import inverse_sigmoid

from .evidence_ledger import EvidenceLedger
from .keep_recover_defer import KeepRecoverDeferPolicy
from .observability_head import GeometricObservabilityHead
from .ray_denoising import prepare_ray_denoising
from .temporal_update import EvidenceConservingTemporalUpdate
from .ternary_objectness import (
    ObservabilityConditionedTernaryLoss,
    TernaryObjectnessHead,
    build_ternary_targets,
    observability_conditioned_background_weights,
)


def _clone_branch(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


@HEADS.register_module()
class EvidenceConservingStreamPETRHead(StreamPETRHead):
    """StreamPETR head with observability-conditioned evidence conservation."""

    def __init__(
        self,
        *args,
        num_cameras: int = 6,
        observability_cfg: Optional[Dict] = None,
        temporal_update_cfg: Optional[Dict] = None,
        policy_cfg: Optional[Dict] = None,
        ternary_hidden_dims: Optional[int] = None,
        ternary_loss_weight: float = 1.0,
        background_observability_floor: float = 0.0,
        evidence_warmup_steps: int = 200,
        enable_observability_conditioning: bool = True,
        enable_evidence_memory: bool = True,
        evidence_probability_source: str = "ternary",
        calibrate_detection_scores: bool = True,
        trace_enabled: bool = True,
        enable_source_ledger: bool = False,
        source_decay: Optional[float] = None,
        source_mass_tolerance: float = 1e-5,
        use_source_ledger_for_evidence: bool = False,
        use_source_ledger_for_policy: bool = False,
        source_camera_names: Optional[Sequence[str]] = None,
        innovation_cfg: Optional[Dict] = None,
        innovation_warmup_iters: int = 0,
        innovation_transition_iters: int = 0,
        enable_ray_denoising: bool = False,
        raydn_group: int = 1,
        raydn_num: int = 5,
        raydn_alpha: float = 8.0,
        raydn_beta: float = 2.0,
        raydn_radius: float = 3.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_cameras = int(num_cameras)
        self.background_observability_floor = float(background_observability_floor)
        self.evidence_warmup_steps = max(int(evidence_warmup_steps), 0)
        self.enable_observability_conditioning = bool(
            enable_observability_conditioning
        )
        self.enable_evidence_memory = bool(enable_evidence_memory)
        self.evidence_probability_source = str(evidence_probability_source).lower()
        if self.evidence_probability_source not in {"ternary", "classification"}:
            raise ValueError(
                "evidence_probability_source must be 'ternary' or "
                "'classification'"
            )
        self.calibrate_detection_scores = bool(calibrate_detection_scores)
        self.trace_enabled = bool(trace_enabled)
        self.enable_ray_denoising = bool(enable_ray_denoising)
        self.raydn_group = int(raydn_group)
        self.raydn_num = int(raydn_num)
        self.raydn_alpha = float(raydn_alpha)
        self.raydn_beta = float(raydn_beta)
        self.raydn_radius = float(raydn_radius)
        if source_camera_names is None:
            source_camera_names = tuple(
                f"CAMERA_{index}" for index in range(self.num_cameras)
            )
        self.source_camera_names = tuple(str(name) for name in source_camera_names)
        if len(self.source_camera_names) != self.num_cameras:
            raise ValueError("source_camera_names length must match num_cameras")

        observability_cfg = dict(observability_cfg or {})
        observability_cfg.setdefault("num_cameras", self.num_cameras)
        observability_cfg.setdefault("embed_dims", self.embed_dims)
        self.observability_head = GeometricObservabilityHead(**observability_cfg)

        prototype = TernaryObjectnessHead(
            embed_dims=self.embed_dims,
            hidden_dims=ternary_hidden_dims,
        )
        self.ternary_branches = _clone_branch(prototype, self.num_pred)
        self.ternary_loss = ObservabilityConditionedTernaryLoss(
            loss_weight=ternary_loss_weight
        )

        temporal_update = EvidenceConservingTemporalUpdate(
            **dict(temporal_update_cfg or {})
        )
        policy = KeepRecoverDeferPolicy(**dict(policy_cfg or {}))
        self.evidence_ledger = EvidenceLedger(
            memory_len=self.memory_len,
            num_cameras=self.num_cameras,
            temporal_update=temporal_update,
            policy=policy,
            enable_source_ledger=enable_source_ledger,
            source_decay=source_decay,
            source_mass_tolerance=source_mass_tolerance,
            use_source_ledger_for_evidence=use_source_ledger_for_evidence,
            use_source_ledger_for_policy=use_source_ledger_for_policy,
            feature_dim=self.embed_dims,
            class_dim=int(
                getattr(
                    self,
                    "cls_out_channels",
                    kwargs.get("num_classes", 0),
                )
            ),
            innovation_cfg=innovation_cfg,
            innovation_warmup_iters=innovation_warmup_iters,
            innovation_transition_iters=innovation_transition_iters,
        )
        # Persist the warm-up counter so resume-from does not silently restart
        # evidence gating from zero.
        self.register_buffer(
            "evidence_step", torch.zeros((), dtype=torch.long), persistent=True
        )
        self._last_evidence_summary: Dict[str, float] = {}
        self._last_evidence_diagnostics: Dict[str, Any] = {}
        # Parent __init__ called reset_memory before the ledger existed.
        self.reset_memory()

    def reset_memory(self) -> None:
        super().reset_memory()
        if hasattr(self, "evidence_ledger"):
            self.evidence_ledger.reset()
        self._last_evidence_summary = {}
        self._last_evidence_diagnostics = {}

    def pre_update_memory(
        self,
        data: Dict[str, Tensor],
        scene_tokens: Optional[Sequence[str]] = None,
    ) -> None:
        super().pre_update_memory(data)
        self.evidence_ledger.pre_update(
            data["prev_exists"],
            scene_tokens=scene_tokens,
            geometry_transform=data.get("ego_pose_inv"),
        )

    @staticmethod
    def _camera_state(
        data: Dict[str, Tensor],
        key: str,
        batch_size: int,
        num_cameras: int,
        reference: Tensor,
        default: float,
    ) -> Tensor:
        value = data.get(key)
        if value is None:
            return reference.new_full((batch_size, num_cameras), default)
        value = value.to(device=reference.device, dtype=reference.dtype)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        return value

    @staticmethod
    def _image_hw_from_metas(
        img_metas: Sequence[Dict],
        num_cameras: int,
        reference: Tensor,
    ) -> Tensor:
        rows: List[List[Tuple[float, float]]] = []
        for meta in img_metas:
            shapes = meta.get("pad_shape", meta.get("img_shape"))
            if shapes is None:
                raise KeyError("img_metas must contain pad_shape or img_shape")
            if isinstance(shapes, tuple):
                shapes = [shapes for _ in range(num_cameras)]
            if len(shapes) != num_cameras:
                raise ValueError(
                    f"Expected {num_cameras} image shapes, got {len(shapes)}"
                )
            rows.append([(float(shape[0]), float(shape[1])) for shape in shapes])
        return reference.new_tensor(rows)

    def _evidence_blend(self) -> float:
        if not self.training or self.evidence_warmup_steps == 0:
            return 1.0
        return min(float(self.evidence_step.item()) / self.evidence_warmup_steps, 1.0)

    def _update_memory_with_evidence(
        self,
        data: Dict[str, Tensor],
        rec_ego_pose: Tensor,
        all_cls_scores: Tensor,
        all_bbox_preds: Tensor,
        outs_dec: Tensor,
        mask_dict: Optional[Dict],
        ternary_probabilities: Tensor,
        observability_output: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        pad_size = int(mask_dict["pad_size"]) if mask_dict and mask_dict.get("pad_size", 0) > 0 else 0
        rec_cls = all_cls_scores[-1, :, pad_size:]
        rec_bbox = all_bbox_preds[-1, :, pad_size:]
        rec_memory = outs_dec[-1, :, pad_size:]
        rec_pose = rec_ego_pose[:, pad_size:]
        rec_ternary = ternary_probabilities[-1, :, pad_size:]
        rec_observability = observability_output["observability"][-1, :, pad_size:]
        rec_source = observability_output["source_vector"][-1, :, pad_size:]
        rec_raw_source = observability_output.get(
            "per_camera",
            observability_output["source_vector"],
        )[-1, :, pad_size:]
        rec_fresh = observability_output["fresh_ratio"][-1, :, pad_size:]
        rec_effective = observability_output["effective_count"][-1, :, pad_size:]
        rec_class_probability = rec_cls.sigmoid()
        rec_class_probability = rec_class_probability / (
            rec_class_probability.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        )
        if rec_bbox.shape[-1] >= 8:
            rec_yaw = torch.atan2(rec_bbox[..., 6], rec_bbox[..., 7])
        else:
            rec_yaw = torch.zeros_like(rec_bbox[..., 0])
        rec_geometry = torch.cat(
            (
                rec_bbox[..., :3],
                rec_yaw.unsqueeze(-1),
                rec_bbox[..., -2:],
            ),
            dim=-1,
        )
        camera_quality = self._camera_state(
            data,
            "camera_quality",
            rec_cls.shape[0],
            self.num_cameras,
            rec_cls,
            1.0,
        )
        rec_source_quality = (
            rec_source
            * camera_quality.unsqueeze(1)
        ).sum(dim=-1).clamp(0.0, 1.0)
        rec_camera_coverage = (
            rec_raw_source > self.evidence_ledger.novelty_eps
        ).sum(dim=-1).to(rec_cls.dtype)

        num_queries = rec_cls.shape[1]
        num_base_queries = min(self.num_query, num_queries)
        num_propagated = min(self.num_propagated, max(num_queries - num_base_queries, 0))
        query_state = self.evidence_ledger.update_queries(
            rec_ternary,
            rec_observability,
            rec_source,
            rec_fresh,
            rec_effective,
            num_base_queries=num_base_queries,
            num_propagated=num_propagated,
            use_strong_negative=(self.evidence_probability_source == "ternary"),
            raw_source_vector=rec_raw_source,
            current_feature=rec_memory,
            current_geometry=rec_geometry,
            current_class_probability=rec_class_probability,
            source_quality=rec_source_quality,
            camera_coverage=rec_camera_coverage,
            innovation_step=int(self.evidence_step.item()),
        )

        bootstrap_eps = float(
            self.evidence_ledger.temporal_update.eps
        )
        query_state["bootstrap_mask"] = (
            query_state["prior_strength"] <= bootstrap_eps
        )

        diagnostic_keys = (
            "alpha",
            "beta",
            "strength",
            "prior_strength",
            "no_new_evidence_strength",
            "bootstrap_mask",
            "existence_probability",
            "uncertainty",
            "observability",
            "novelty",
            "effective_count",
            "evidence_gate",
            "reliable_observation",
            "no_new_evidence",
            "actual_added_positive_evidence",
            "actual_added_negative_evidence",
            "conservation_ratio",
            "conservation_violation",
            "conservation_violation_mask",
            "conservation_residual",
            "unsupported_growth",
            "current_source_vector",
            "current_source_distribution",
            "current_source_increment",
            "source_evidence",
            "source_strength",
            "provenance",
            "previous_source_strength",
            "source_mass_residual",
            "source_mass_violation",
            "zero_source_increment",
            "age",
            "action",
            "score_scale",
            "write_mask",
            "source_innovation",
            "feature_innovation",
            "geometry_innovation",
            "class_semantic_innovation",
            "ternary_semantic_innovation",
            "semantic_innovation",
            "temporal_reacquisition",
            "is_new_query",
            "is_continuous_observation",
            "is_reacquired_query",
            "valid_feature_pair",
            "valid_geometry",
            "center_residual",
            "yaw_residual",
            "velocity_residual",
            "observation_reliability",
            "source_reliability",
            "semantic_reliability",
            "geometric_reliability",
            "combined_reliability",
            "semantic_conflict",
            "geometry_conflict",
            "existence_conflict",
            "conflict",
            "base_innovation",
            "reacquired_innovation",
            "compatible_innovation",
            "reliable_innovation",
            "novelty_gain",
            "positive_novelty_gain",
            "negative_novelty_gain",
            "positive_reliability",
            "negative_reliability",
            "negative_visibility_gate",
            "strength_saturation",
            "innovation_transition",
        )
        self._last_evidence_diagnostics = {
            key: query_state[key].detach().cpu()
            for key in diagnostic_keys
            if key in query_state
        }
        if self.evidence_ledger.enable_source_ledger:
            self._last_evidence_diagnostics[
                "source_camera_names"
            ] = self.source_camera_names

        if not self.enable_evidence_memory:
            super().post_update_memory(
                data,
                rec_ego_pose,
                all_cls_scores,
                all_bbox_preds,
                outs_dec,
                mask_dict,
            )
            if self.trace_enabled:
                self._last_evidence_summary = self.evidence_ledger.summarize(
                    query_state
                )
                self._last_evidence_summary["evidence_memory_enabled"] = 0.0
                self._last_evidence_summary["scene_reset"] = float(
                    self.evidence_ledger.last_scene_reset
                )
                self._last_evidence_summary["scene_reset_count"] = int(
                    self.evidence_ledger.scene_reset_count
                )
            return query_state

        class_confidence = rec_cls.sigmoid().amax(dim=-1)
        evidence_score = (
            class_confidence
            * query_state["existence_probability"]
            * query_state["score_scale"]
        )
        blend = self._evidence_blend()
        policy_ranking_score = (
            (1.0 - blend) * class_confidence
            + blend * evidence_score
        )
        # Queries without prior evidence must bootstrap from the official
        # classification score. Otherwise an empty ledger causes a permanent
        # DEFER -> no write -> empty ledger deadlock.
        ranking_score = torch.where(
            query_state["bootstrap_mask"],
            class_confidence,
            policy_ranking_score,
        )
        k = min(self.topk_proposals, ranking_score.shape[1])
        topk_indexes = torch.topk(
            ranking_score.unsqueeze(-1), k, dim=1
        ).indices

        rec_timestamp = torch.zeros_like(
            ranking_score.unsqueeze(-1), dtype=torch.float64
        )
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_bbox[..., :3], topk_indexes).detach()
        rec_velocity = topk_gather(rec_bbox[..., -2:], topk_indexes).detach()
        rec_memory_topk = topk_gather(rec_memory, topk_indexes).detach()
        rec_pose_topk = topk_gather(rec_pose, topk_indexes)
        policy_write_mask = topk_gather(
            query_state["write_mask"].unsqueeze(-1).to(
                rec_memory_topk.dtype
            ),
            topk_indexes,
        )
        bootstrap_write_mask = topk_gather(
            query_state["bootstrap_mask"].unsqueeze(-1).to(
                rec_memory_topk.dtype
            ),
            topk_indexes,
        )
        warmup_active = self.training and blend < 1.0
        write_mask = (
            torch.ones_like(policy_write_mask)
            if warmup_active
            else torch.maximum(
                policy_write_mask,
                bootstrap_write_mask,
            )
        )

        # During warm-up, retain the official memory-writing behavior so the
        # ternary branch can learn before hard Defer gating is activated.
        # Afterwards, Defer does not delete the current output, but it cannot
        # write a confident state back into temporal memory.
        rec_memory_topk = rec_memory_topk * write_mask
        rec_reference_points = rec_reference_points * write_mask
        rec_velocity = rec_velocity * write_mask
        if rec_pose_topk.numel() > 0:
            identity = torch.eye(
                4, device=rec_pose_topk.device, dtype=rec_pose_topk.dtype
            ).view(1, 1, 4, 4)
            rec_pose_topk = torch.where(
                write_mask.unsqueeze(-1).bool(), rec_pose_topk, identity
            )

        self.memory_embedding = torch.cat(
            (rec_memory_topk, self.memory_embedding), dim=1
        )
        self.memory_timestamp = torch.cat(
            (rec_timestamp, self.memory_timestamp), dim=1
        )
        self.memory_egopose = torch.cat(
            (rec_pose_topk, self.memory_egopose), dim=1
        )
        self.memory_reference_point = torch.cat(
            (rec_reference_points, self.memory_reference_point), dim=1
        )
        self.memory_velo = torch.cat(
            (rec_velocity, self.memory_velo), dim=1
        )
        self.memory_reference_point = transform_reference_points(
            self.memory_reference_point, data["ego_pose"], reverse=False
        )
        self.memory_timestamp -= data["timestamp"].unsqueeze(-1).unsqueeze(-1)
        self.memory_egopose = data["ego_pose"].unsqueeze(1) @ self.memory_egopose

        valid_query_write_mask = (
            torch.ones_like(
                query_state["write_mask"],
                dtype=torch.bool,
            )
            if warmup_active
            else (
                query_state["write_mask"]
                | query_state["bootstrap_mask"]
            )
        )
        # Read-only evaluation diagnostics. These tensors are detached CPU
        # copies and never participate in prediction or memory updates.
        selected_query_mask = torch.zeros_like(
            query_state["write_mask"], dtype=torch.bool
        )
        selected_query_mask.scatter_(
            1, topk_indexes.squeeze(-1), True
        )
        self._last_evidence_diagnostics.update(
            {
                "reference_geometry": query_state[
                    "reference_geometry"
                ].detach().cpu(),
                "reference_class_distribution": query_state[
                    "reference_class_distribution"
                ].detach().cpu(),
                "topk_selected_mask": selected_query_mask.detach().cpu(),
                "actual_memory_write_mask": (
                    selected_query_mask & valid_query_write_mask
                ).detach().cpu(),
            }
        )
        self.evidence_ledger.commit_topk(
            query_state,
            topk_indexes,
            valid_write_mask=valid_query_write_mask,
        )
        self.evidence_ledger.transform_reference_geometry(data["ego_pose"])
        if self.trace_enabled:
            self._last_evidence_summary = self.evidence_ledger.summarize(query_state)
            self._last_evidence_summary["evidence_blend"] = float(blend)
            self._last_evidence_summary["evidence_memory_enabled"] = 1.0
            self._last_evidence_summary["warmup_active"] = float(warmup_active)
            self._last_evidence_summary["bootstrap_count"] = int(
                query_state["bootstrap_mask"].sum().detach().cpu()
            )
            self._last_evidence_summary["scene_reset"] = float(
                self.evidence_ledger.last_scene_reset
            )
            self._last_evidence_summary["scene_reset_count"] = int(
                self.evidence_ledger.scene_reset_count
            )
        return query_state

    def forward(
        self,
        memory_center: Tensor,
        img_metas: Sequence[Dict],
        topk_indexes: Optional[Tensor] = None,
        **data,
    ) -> Dict[str, Tensor]:
        scene_tokens = [
            str(meta.get("scene_token", ""))
            for meta in img_metas
        ]
        self.pre_update_memory(data, scene_tokens=scene_tokens)
        if self.training:
            self.evidence_step.add_(1)

        x = data["img_feats"]
        batch_size, num_cameras, channels, height, width = x.shape
        num_tokens = num_cameras * height * width
        memory = x.permute(0, 1, 3, 4, 2).reshape(
            batch_size, num_tokens, channels
        )
        memory = topk_gather(memory, topk_indexes)
        pos_embed, cone = self.position_embeding(
            data, memory_center, topk_indexes, img_metas
        )
        memory = self.memory_embed(memory)
        memory = self.spatial_alignment(memory, cone)
        pos_embed = self.featurized_pe(pos_embed, memory)

        reference_points = self.reference_points.weight
        if self.training and self.enable_ray_denoising:
            reference_points, attn_mask, mask_dict = prepare_ray_denoising(
                self,
                batch_size,
                reference_points,
                img_metas,
                data,
                raydn_group=self.raydn_group,
                raydn_num=self.raydn_num,
                raydn_alpha=self.raydn_alpha,
                raydn_beta=self.raydn_beta,
                raydn_radius=self.raydn_radius,
            )
        else:
            reference_points, attn_mask, mask_dict = self.prepare_for_dn(
                batch_size, reference_points, img_metas
            )
        query_pos = self.query_embedding(pos2posemb3d(reference_points))
        tgt = torch.zeros_like(query_pos)
        (
            tgt,
            query_pos,
            reference_points,
            temp_memory,
            temp_pos,
            rec_ego_pose,
        ) = self.temporal_alignment(query_pos, tgt, reference_points)

        outs_dec, _ = self.transformer(
            memory,
            tgt,
            query_pos,
            pos_embed,
            attn_mask,
            temp_memory,
            temp_pos,
        )
        outs_dec = torch.nan_to_num(outs_dec)

        outputs_classes = []
        outputs_coords = []
        outputs_ternary = []
        for level in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            outputs_class = self.cls_branches[level](outs_dec[level])
            tmp = self.reg_branches[level](outs_dec[level])
            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp)
            outputs_ternary.append(self.ternary_branches[level](outs_dec[level]))

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_ternary_logits = torch.stack(outputs_ternary)
        all_bbox_preds[..., 0:3] = (
            all_bbox_preds[..., 0:3]
            * (self.pc_range[3:6] - self.pc_range[0:3])
            + self.pc_range[0:3]
        )

        image_hw = self._image_hw_from_metas(
            img_metas, num_cameras, all_bbox_preds
        )
        camera_online = self._camera_state(
            data,
            "camera_online_mask",
            batch_size,
            num_cameras,
            all_bbox_preds,
            1.0,
        )
        camera_quality = self._camera_state(
            data,
            "camera_quality",
            batch_size,
            num_cameras,
            all_bbox_preds,
            1.0,
        )
        camera_fresh = self._camera_state(
            data,
            "camera_fresh_mask",
            batch_size,
            num_cameras,
            all_bbox_preds,
            1.0,
        )
        observability_output = self.observability_head(
            all_bbox_preds[..., :3],
            data["lidar2img"],
            image_hw,
            camera_online_mask=camera_online,
            camera_quality=camera_quality,
            camera_fresh_mask=camera_fresh,
            query_features=outs_dec,
        )
        if self.evidence_probability_source == "ternary":
            all_ternary_probabilities = all_ternary_logits.softmax(dim=-1)
        else:
            class_presence = all_cls_scores.sigmoid().amax(dim=-1)
            all_ternary_probabilities = torch.stack(
                (
                    class_presence,
                    1.0 - class_presence,
                    torch.zeros_like(class_presence),
                ),
                dim=-1,
            )
        query_state = self._update_memory_with_evidence(
            data,
            rec_ego_pose,
            all_cls_scores,
            all_bbox_preds,
            outs_dec,
            mask_dict,
            all_ternary_probabilities,
            observability_output,
        )

        pad_size = int(mask_dict["pad_size"]) if mask_dict and mask_dict.get("pad_size", 0) > 0 else 0
        if pad_size > 0:
            mask_dict["output_known_lbs_bboxes"] = (
                all_cls_scores[:, :, :pad_size],
                all_bbox_preds[:, :, :pad_size],
            )
            all_cls_scores = all_cls_scores[:, :, pad_size:]
            all_bbox_preds = all_bbox_preds[:, :, pad_size:]
            all_ternary_logits = all_ternary_logits[:, :, pad_size:]
            for key in list(observability_output):
                observability_output[key] = observability_output[key][:, :, pad_size:]

        return {
            "all_cls_scores": all_cls_scores,
            "all_bbox_preds": all_bbox_preds,
            "dn_mask_dict": mask_dict if pad_size > 0 else None,
            "all_ternary_logits": all_ternary_logits,
            "all_observability": observability_output["observability"],
            "all_camera_support": observability_output["source_vector"],
            "all_effective_count": observability_output["effective_count"],
            "all_fresh_ratio": observability_output["fresh_ratio"],
            "evidence_presence": query_state["existence_probability"],
            "evidence_uncertainty": query_state["uncertainty"],
            "evidence_action": query_state["action"],
            "evidence_score_scale": query_state["score_scale"],
            "evidence_bootstrap_mask": query_state["bootstrap_mask"],
            "evidence_novelty": query_state["novelty"],
            "evidence_strength": query_state["strength"],
            "evidence_conservation_residual": query_state[
                "conservation_residual"
            ],
        }

    def loss_single_evidence(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        ternary_logits: Tensor,
        observability: Tensor,
        gt_bboxes_list: Sequence[Tensor],
        gt_labels_list: Sequence[Tensor],
        gt_bboxes_ignore_list: Optional[Sequence[Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            gt_labels_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum(ind.numel() for ind in pos_inds_list)

        ternary_targets = []
        ternary_weights = []
        for image_index in range(num_imgs):
            supervision_observability = (
                observability[image_index]
                if self.enable_observability_conditioning
                else torch.ones_like(observability[image_index])
            )
            label_weights_list[image_index] = observability_conditioned_background_weights(
                label_weights_list[image_index],
                neg_inds_list[image_index],
                supervision_observability,
                floor=self.background_observability_floor,
            )
            targets, weights = build_ternary_targets(
                cls_scores.shape[1],
                pos_inds_list[image_index],
                neg_inds_list[image_index],
                supervision_observability,
                ternary_logits.dtype,
                ternary_logits.device,
            )
            ternary_targets.append(targets)
            ternary_weights.append(weights)

        labels = torch.cat(labels_list, dim=0)
        label_weights = torch.cat(label_weights_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        bbox_weights = torch.cat(bbox_weights_list, dim=0)
        cls_scores_flat = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = max(float(num_total_pos), 1.0)
        if self.sync_cls_avg_factor:
            cls_avg_factor = float(
                reduce_mean(cls_scores_flat.new_tensor([cls_avg_factor])).item()
            )
        loss_cls = self.loss_cls(
            cls_scores_flat,
            labels,
            label_weights,
            avg_factor=max(cls_avg_factor, 1.0),
        )

        num_total_pos_tensor = loss_cls.new_tensor([num_total_pos])
        regression_avg = torch.clamp(
            reduce_mean(num_total_pos_tensor), min=1
        ).item()
        bbox_preds_flat = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_targets = normalize_bbox(bbox_targets, self.pc_range)
        finite = torch.isfinite(normalized_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds_flat[finite, :10],
            normalized_targets[finite, :10],
            bbox_weights[finite, :10],
            avg_factor=regression_avg,
        )
        loss_ternary = self.ternary_loss(
            ternary_logits.reshape(-1, 3),
            torch.cat(ternary_targets, dim=0),
            torch.cat(ternary_weights, dim=0),
        )
        return (
            torch.nan_to_num(loss_cls),
            torch.nan_to_num(loss_bbox),
            torch.nan_to_num(loss_ternary),
        )

    @force_fp32(apply_to=("preds_dicts",))
    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        preds_dicts,
        gt_bboxes_ignore=None,
    ):
        if gt_bboxes_ignore is not None:
            raise AssertionError("gt_bboxes_ignore is not supported")
        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]
        all_ternary_logits = preds_dicts["all_ternary_logits"]
        all_observability = preds_dicts["all_observability"]
        num_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_tensor = [
            torch.cat((boxes.gravity_center, boxes.tensor[:, 3:]), dim=1).to(device)
            for boxes in gt_bboxes_list
        ]
        all_gt_bboxes = [gt_bboxes_tensor for _ in range(num_layers)]
        all_gt_labels = [gt_labels_list for _ in range(num_layers)]
        all_ignore = [None for _ in range(num_layers)]
        losses_cls, losses_bbox, losses_ternary = multi_apply(
            self.loss_single_evidence,
            all_cls_scores,
            all_bbox_preds,
            all_ternary_logits,
            all_observability,
            all_gt_bboxes,
            all_gt_labels,
            all_ignore,
        )
        loss_dict = {
            "loss_cls": losses_cls[-1],
            "loss_bbox": losses_bbox[-1],
            "loss_ternary": losses_ternary[-1],
        }
        for index, (loss_cls, loss_bbox, loss_ternary) in enumerate(
            zip(losses_cls[:-1], losses_bbox[:-1], losses_ternary[:-1])
        ):
            loss_dict[f"d{index}.loss_cls"] = loss_cls
            loss_dict[f"d{index}.loss_bbox"] = loss_bbox
            loss_dict[f"d{index}.loss_ternary"] = loss_ternary

        if preds_dicts["dn_mask_dict"] is not None:
            (
                known_labels,
                known_bboxs,
                output_known_class,
                output_known_coord,
                num_tgt,
            ) = self.prepare_for_loss(preds_dicts["dn_mask_dict"])
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.dn_loss_single,
                output_known_class,
                output_known_coord,
                [known_bboxs for _ in range(num_layers)],
                [known_labels for _ in range(num_layers)],
                [num_tgt for _ in range(num_layers)],
            )
            loss_dict["dn_loss_cls"] = dn_losses_cls[-1]
            loss_dict["dn_loss_bbox"] = dn_losses_bbox[-1]
            for index, (loss_cls, loss_bbox) in enumerate(
                zip(dn_losses_cls[:-1], dn_losses_bbox[:-1])
            ):
                loss_dict[f"d{index}.dn_loss_cls"] = loss_cls
                loss_dict[f"d{index}.dn_loss_bbox"] = loss_bbox
        elif self.with_dn:
            # Keep the official key contract when a batch contains no DN target.
            loss_dict["dn_loss_cls"] = losses_cls[-1].detach() * 0.0
            loss_dict["dn_loss_bbox"] = losses_bbox[-1].detach() * 0.0
        return loss_dict

    @force_fp32(apply_to=("preds_dicts",))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        if not self.calibrate_detection_scores:
            return super().get_bboxes(preds_dicts, img_metas, rescale=rescale)
        calibrated = dict(preds_dicts)
        logits = preds_dicts["all_cls_scores"].clone()
        presence = preds_dicts.get("evidence_presence")
        score_scale = preds_dicts.get("evidence_score_scale")
        bootstrap_mask = preds_dicts.get(
            "evidence_bootstrap_mask"
        )
        if presence is not None and score_scale is not None:
            scale = (
                presence * score_scale
            ).clamp(0.0, 1.0)
            if bootstrap_mask is not None:
                # No evidence prior exists yet, so retain the official detector
                # score rather than suppressing it as an evidence-based DEFER.
                scale = torch.where(
                    bootstrap_mask.bool(),
                    torch.ones_like(scale),
                    scale,
                )
            scale = scale.unsqueeze(-1)
            probability = (
                logits[-1].sigmoid() * scale
            ).clamp(1e-6, 1.0 - 1e-6)
            logits[-1] = torch.log(probability / (1.0 - probability))
        calibrated["all_cls_scores"] = logits
        return super().get_bboxes(calibrated, img_metas, rescale=rescale)

    def get_last_evidence_summary(self) -> Dict[str, float]:
        return dict(self._last_evidence_summary)

    def get_last_evidence_diagnostics(self) -> Dict[str, Any]:
        return {
            key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in self._last_evidence_diagnostics.items()
        }
