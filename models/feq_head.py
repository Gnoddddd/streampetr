"""Project-side, parameter-free StreamPETR FEQ training adapter."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models import HEADS
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import StreamPETRHead

from .feq_losses import (
    PRESENT, UNOBSERVED, adjacent_survival_loss, geometric_auxiliary_cost,
    greedy_auxiliary_assignment, supervision_weights, topk_boundary_loss,
)


@HEADS.register_module()
class FEQStreamPETRHead(StreamPETRHead):
    """The stock head plus optional losses; inference/state keys stay identical."""

    def __init__(self, *args, enable_feq_core=False, feq_otm_weight=0.0,
                 feq_boundary_weight=0.0, feq_max_aux=3,
                 feq_boundary_margin=0.10, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_feq_core = bool(enable_feq_core)
        self.feq_otm_weight = float(feq_otm_weight)
        self.feq_boundary_weight = float(feq_boundary_weight)
        self.feq_max_aux = int(feq_max_aux)
        self.feq_boundary_margin = float(feq_boundary_margin)
        self._feq_context = None
        self._last_feq_summary: Dict[str, float] = {}
        self._last_feq_diagnostics: List[Dict] = []

    def forward(self, memory_center, img_metas, topk_indexes=None, **data):
        if self.training and self.enable_feq_core:
            self._feq_context = {
                "img_metas": img_metas,
                "lidar2img": data.get("lidar2img"),
                "online": data.get("camera_online_mask"),
                "fresh": data.get("camera_fresh_mask"),
            }
        else:
            self._feq_context = None
        return super().forward(memory_center, img_metas, topk_indexes, **data)

    def _states(self, gt_boxes, batch):
        device = gt_boxes.device
        count = len(gt_boxes)
        states = torch.full((count,), PRESENT, device=device, dtype=torch.long)
        history = torch.zeros(count, device=device, dtype=torch.bool)
        if not count or not self._feq_context:
            return states, history
        online = self._feq_context.get("online")
        fresh = self._feq_context.get("fresh")
        matrices = self._feq_context.get("lidar2img")
        if online is None or matrices is None:
            return states, history
        online = online[batch] if online.ndim > 1 else online
        if fresh is not None:
            fresh = fresh[batch] if fresh.ndim > 1 else fresh
            online = online * fresh
        matrices = matrices[batch]
        xyz1 = torch.cat((gt_boxes[:, :3], gt_boxes.new_ones(count, 1)), -1)
        projected = torch.einsum("cij,nj->cni", matrices.to(device), xyz1)
        depth = projected[..., 2]
        uv = projected[..., :2] / depth.unsqueeze(-1).clamp_min(1e-5)
        meta = self._feq_context["img_metas"][batch]
        shapes = meta.get("pad_shape", meta.get("img_shape"))
        if isinstance(shapes, tuple): shapes = [shapes] * matrices.shape[0]
        visible = torch.zeros(count, dtype=torch.bool, device=device)
        for camera, shape in enumerate(shapes):
            inside = (depth[camera] > 1e-3) & (uv[camera, :, 0] >= 0) & \
                (uv[camera, :, 0] < shape[1]) & (uv[camera, :, 1] >= 0) & \
                (uv[camera, :, 1] < shape[0]) & (online[camera] > 0.5)
            visible |= inside
        states[~visible] = UNOBSERVED
        history_centers = meta.get("feq_history_centers", [])
        if len(history_centers):
            old = gt_boxes.new_tensor(history_centers)
            history = torch.cdist(gt_boxes[:, :3], old[:, :3]).min(dim=1).values < 0.25
        return states, history

    def _main_queries(self, logits, boxes, gt, labels):
        assign = self.assigner.assign(boxes.detach(), logits.detach(), gt, labels,
                                      None, self.match_costs, self.match_with_velo)
        main = labels.new_full((len(gt),), -1)
        for query in torch.nonzero(assign.gt_inds > 0, as_tuple=False).flatten():
            main[int(assign.gt_inds[query]) - 1] = query
        return main

    def _layer_sets(self, logits, boxes, gt, labels, eligible):
        main = self._main_queries(logits, boxes, gt, labels)
        norm_gt = normalize_bbox(gt, self.pc_range)
        geometry_cost = geometric_auxiliary_cost(boxes, norm_gt, self.pc_range)
        aux, conflicts = greedy_auxiliary_assignment(
            geometry_cost, main, eligible, self.feq_max_aux)
        original_aux, _ = greedy_auxiliary_assignment(
            geometry_cost - logits.detach().sigmoid()[:, labels],
            main, eligible, self.feq_max_aux,
        )
        sets = []
        for index, values in enumerate(aux):
            sets.append(([int(main[index])] if int(main[index]) >= 0 else []) + values)
        return sets, aux, original_aux, conflicts, norm_gt, main

    @force_fp32(apply_to=("preds_dicts",))
    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore=None):
        original = super().loss(gt_bboxes_list, gt_labels_list, preds_dicts,
                                gt_bboxes_ignore)
        if not (self.training and self.enable_feq_core):
            self._last_feq_summary = {}
            self._last_feq_diagnostics = []
            return original
        logits_layers = preds_dicts["all_cls_scores"]
        box_layers = preds_dicts["all_bbox_preds"]
        converted = [torch.cat((box.gravity_center, box.tensor[:, 3:]), 1).to(
            gt_labels_list[i].device) for i, box in enumerate(gt_bboxes_list)]
        otm_terms, boundary_terms, survival_values = [], [], []
        diagnostics: List[Dict] = []
        aux_count = conflict_count = comparisons = gt_count = 0
        for batch, (gt, labels) in enumerate(zip(converted, gt_labels_list)):
            if not len(gt): continue
            states, history = self._states(gt, batch)
            weights = supervision_weights(states, history).to(gt)
            eligible = weights > 0
            layer_sets = []
            meta = self._feq_context["img_metas"][batch]
            online = self._feq_context.get("online")
            fresh = self._feq_context.get("fresh")
            online_row = online[batch] if online is not None and online.ndim > 1 else online
            fresh_row = fresh[batch] if fresh is not None and fresh.ndim > 1 else fresh
            is_fault = bool(
                (online_row is not None and torch.any(online_row < 0.5)) or
                (fresh_row is not None and torch.any(fresh_row < 0.5))
            )
            for layer, (logits, boxes) in enumerate(zip(
                    logits_layers[:, batch], box_layers[:, batch])):
                sets, aux, original_aux, conflicts, norm_gt, main = self._layer_sets(
                    logits, boxes, gt, labels, eligible)
                layer_sets.append(sets)
                conflict_count += conflicts
                boundary, boundary_details = topk_boundary_loss(
                    logits, labels, sets, weights,
                    int(getattr(self.bbox_coder, "max_num", 100)),
                    self.feq_boundary_margin,
                )
                boundary_terms.append(boundary)
                boundary_by_gt = {int(item["gt"]): item for item in boundary_details}
                for target, queries in enumerate(aux):
                    gt_otm_terms = []
                    for query in queries:
                        cls = F.binary_cross_entropy_with_logits(
                            logits[query, labels[target]], logits.new_tensor(1.0))
                        dims = list(range(10)) if states[target] == PRESENT else [0, 1, 2, 8, 9]
                        reg = F.smooth_l1_loss(boxes[query, dims], norm_gt[target, dims])
                        term = weights[target] * (cls + reg)
                        otm_terms.append(term); gt_otm_terms.append(term.detach())
                        aux_count += 1
                    boundary_item = boundary_by_gt.get(target)
                    main_query = int(main[target])
                    geo_scores = [float(logits[q, labels[target]].sigmoid().detach()) for q in queries]
                    original_scores = [float(logits[q, labels[target]].sigmoid().detach())
                                       for q in original_aux[target]]
                    near_duplicate = False
                    if main_query >= 0 and queries:
                        center_delta = torch.linalg.vector_norm(
                            boxes[queries, :3] - boxes[main_query, :3], dim=-1)
                        box_delta = (boxes[queries, :10] - boxes[main_query, :10]).abs().mean(-1)
                        near_duplicate = bool(torch.any((center_delta < 0.5) & (box_delta < 0.05)))
                    diagnostics.append({
                        "sample_token": str(meta.get("sample_idx", "")),
                        "scene_token": str(meta.get("scene_token", "")),
                        "batch": batch, "layer": layer, "gt": target,
                        "state": "Present" if int(states[target]) == PRESENT else "Unobserved",
                        "reliable_history": bool(history[target]), "fault": is_fault,
                        "weight": float(weights[target]), "main_query": main_query,
                        "main_exists": main_query >= 0,
                        "geometric_aux_count": len(queries),
                        "original_aux_count": len(original_aux[target]),
                        "geometric_aux_queries": list(queries),
                        "original_aux_queries": list(original_aux[target]),
                        "geometric_aux_scores": geo_scores,
                        "original_aux_scores": original_scores,
                        "main_query_overlap": main_query in queries,
                        "near_duplicate_box": near_duplicate,
                        "competition": conflicts,
                        "otm_loss": float(torch.stack(gt_otm_terms).mean()) if gt_otm_terms else 0.0,
                        "s_pos": float(boundary_item["s_pos"]) if boundary_item else 0.0,
                        "s_k": float(boundary_item["s_k"]) if boundary_item else 0.0,
                        "boundary_gap": float(boundary_item["gap"]) if boundary_item else 0.0,
                        "boundary_violation": bool(boundary_item["violation"]) if boundary_item else False,
                        "positive_in_topk": bool(boundary_item["positive_in_topk"]) if boundary_item else False,
                        "boundary_loss": float(boundary_item["weighted_loss"]) if boundary_item else 0.0,
                    })
            with torch.no_grad():
                survive, count = adjacent_survival_loss(
                    list(logits_layers[:, batch]), list(box_layers[:, batch]), labels,
                    gt[:, :3], layer_sets, weights, 0.05)
            survival_values.append(survive.detach()); comparisons += count; gt_count += len(gt)
        zero = logits_layers.sum() * 0.0
        raw_otm = torch.stack(otm_terms).mean() if otm_terms else zero
        raw_boundary = torch.stack(boundary_terms).mean() if boundary_terms else zero
        raw_survive = torch.stack(survival_values).mean() if survival_values else zero.detach()
        original["loss_feq_otm"] = raw_otm * self.feq_otm_weight
        original["loss_feq_boundary"] = raw_boundary * self.feq_boundary_weight
        original["feq_raw_otm"] = raw_otm.detach()
        original["feq_raw_boundary"] = raw_boundary.detach()
        original["feq_survival_diagnostic"] = raw_survive
        self._last_feq_summary = {"aux_queries": aux_count, "competition": conflict_count,
                                  "survival_comparisons": comparisons, "gt": gt_count}
        self._last_feq_diagnostics = diagnostics
        return original
