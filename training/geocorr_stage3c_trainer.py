"""Training primitives for GeoCorr Stage 3-C.

The module deliberately keeps data loading and the frozen StreamPETR image
backbone outside the trainable model.  Clean/history tensors enter detached;
only the dirty descriptor adapter and recovery MLP own parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from models.geocorr_recovery import (
    GeoCorrStage3BOutput,
    GeoCorrStage3BRecovery,
    ObjectCentricCorrelationField,
    correspondence_distillation,
    masked_softmax,
    recovery_feature_loss,
)


@dataclass(frozen=True)
class Stage3CConfig:
    feature_dim: int
    candidate_offsets: Tuple[Tuple[float, float], ...]
    top_k: Union[int, str]
    temperature: float
    lambda_corr: float
    lambda_rec: float


@dataclass(frozen=True)
class Stage3CForward:
    recovery: GeoCorrStage3BOutput
    loss_corr: Tensor
    loss_rec: Tensor
    correspondence_diagnostics: Mapping[str, Tensor]


@dataclass(frozen=True)
class Stage3CLoss:
    total: Tensor
    detection: Tensor
    correspondence: Tensor
    recovery: Tensor
    detection_components: Mapping[str, Tensor]


class GeoCorrStage3CModel(nn.Module):
    """The complete trainable Stage 3-C sidecar, excluding StreamPETR."""

    def __init__(self, config: Stage3CConfig) -> None:
        super().__init__()
        if config.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.config = config
        self.correlation = ObjectCentricCorrelationField(
            config.candidate_offsets,
            feature_dim=config.feature_dim,
            detach_teacher=True,
            detach_history=True,
            detach_dirty_base=True,
        )
        self.recovery = GeoCorrStage3BRecovery(
            feature_dim=config.feature_dim,
            top_k=config.top_k,
            recovery_zero_init=True,
        )

    def forward(
        self,
        current_clean_tokens: Tensor,
        current_dirty_tokens: Tensor,
        history_clean_tokens: Tensor,
        current_valid_mask: Tensor,
        history_valid_mask: Tensor,
        current_dirty_fpn: Tensor,
        projected_center_coords: Tensor,
        projected_center_valid: Tensor,
        current_clean_fpn: Optional[Tensor] = None,
        object_valid_mask: Optional[Tensor] = None,
    ) -> Stage3CForward:
        # These branches are frozen teachers even if a caller accidentally
        # supplies tensors carrying an autograd history.
        clean_tokens = current_clean_tokens.detach()
        history_tokens = history_clean_tokens.detach()
        dirty_base = current_dirty_fpn.detach()
        corr = self.correlation(
            clean_tokens,
            current_dirty_tokens.detach(),
            history_tokens,
            current_valid_mask,
            history_valid_mask,
            object_valid_mask=object_valid_mask,
        )
        loss_corr, corr_diagnostics = correspondence_distillation(
            corr["teacher_logits"],
            corr["student_logits"],
            corr["valid_mask"],
            self.config.temperature,
            self.correlation.center_candidate_index,
        )
        recovery = self.recovery(
            q_dirty=corr["dirty_descriptor"],
            q_clean=corr["clean_descriptor"],
            historical_features=corr["history_descriptor"],
            p_student=corr_diagnostics["student_probs"],
            correspondence_valid_mask=corr["valid_mask"],
            p_teacher=corr_diagnostics["teacher_probs"],
            current_clean_fpn=(
                None if current_clean_fpn is None else current_clean_fpn.detach()
            ),
            current_dirty_fpn=dirty_base,
            projected_center_coords=projected_center_coords,
            projected_center_valid=projected_center_valid,
        )
        loss_rec = recovery_feature_loss(
            recovery.q_recovered, recovery.q_clean, recovery.query_valid
        )
        return Stage3CForward(recovery, loss_corr, loss_rec, corr_diagnostics)


    def inference_forward(
        self,
        current_dirty_tokens: Tensor,
        history_clean_tokens: Tensor,
        current_valid_mask: Tensor,
        history_valid_mask: Tensor,
        current_dirty_fpn: Tensor,
        projected_center_coords: Tensor,
        projected_center_valid: Tensor,
        object_valid_mask: Optional[Tensor] = None,
    ) -> GeoCorrStage3BOutput:
        """Recover a dirty current FPN without reading a clean current frame.

        Training uses the clean current branch only as a teacher for L_corr/L_rec.
        Formal fault inference must be deployable, so this path constructs only the
        student correspondence from current dirty descriptors and previous clean
        history. No clean-current tensor is accepted by the signature.
        """
        if current_dirty_tokens.ndim != 5 or history_clean_tokens.ndim != 6:
            raise ValueError("current/history tokens must be 5D/6D")
        if history_clean_tokens.shape[3] != 1:
            raise ValueError("GeoCorr V1 inference supports exactly one history frame")
        if current_valid_mask.shape != current_dirty_tokens.shape[:-1]:
            raise ValueError("current_valid_mask shape does not match dirty tokens")
        if history_valid_mask.shape != history_clean_tokens.shape[:-1]:
            raise ValueError("history_valid_mask shape does not match history tokens")
        if current_dirty_tokens.shape[:3] != history_clean_tokens.shape[:3]:
            raise ValueError("current/history B,N,J dimensions must match")

        center = self.correlation.center_candidate_index
        dirty_query = current_dirty_tokens[:, :, center].detach()
        history_key = history_clean_tokens.detach()
        q_dirty = self.correlation.adapter(dirty_query)
        history_descriptor = torch.nn.functional.normalize(history_key, dim=-1)
        student_logits = torch.einsum(
            "bnvc,bnjtwc->bnvjtw", q_dirty, history_descriptor
        )
        if self.correlation.geometry_prior_beta:
            prior = (
                -self.correlation.geometry_prior_beta
                * self.correlation.candidate_offsets.square().sum(-1)
            )
            student_logits = student_logits + prior.to(student_logits).reshape(
                1, 1, 1, -1, 1, 1
            )

        query_valid = current_valid_mask[:, :, center]
        if object_valid_mask is not None:
            if object_valid_mask.shape != query_valid.shape[:2]:
                raise ValueError("object_valid_mask must have shape [B,N]")
            query_valid = query_valid & object_valid_mask[:, :, None]
        valid_mask = (
            query_valid[:, :, :, None, None, None]
            & history_valid_mask[:, :, None]
        )
        p_student, _ = masked_softmax(
            student_logits, valid_mask, self.config.temperature
        )
        return self.recovery(
            q_dirty=q_dirty,
            q_clean=q_dirty.detach(),
            historical_features=history_descriptor,
            p_student=p_student,
            correspondence_valid_mask=valid_mask,
            p_teacher=None,
            current_clean_fpn=None,
            current_dirty_fpn=current_dirty_fpn.detach(),
            projected_center_coords=projected_center_coords,
            projected_center_valid=projected_center_valid,
        )

def freeze_detector(detector: nn.Module) -> None:
    """Freeze every detector parameter without disabling downstream autograd."""
    for parameter in detector.parameters():
        parameter.requires_grad_(False)


def trainable_named_parameters(
    geocorr: nn.Module,
) -> List[Tuple[str, nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in geocorr.named_parameters()
        if parameter.requires_grad
    ]


def build_optimizer(
    detector: nn.Module, geocorr: nn.Module, lr: float
) -> torch.optim.Optimizer:
    """Build an AdamW optimizer and hard-fail on detector parameter leakage."""
    if lr <= 0:
        raise ValueError("learning rate must be positive")
    detector_parameters = list(detector.named_parameters())
    still_trainable = [name for name, value in detector_parameters if value.requires_grad]
    if still_trainable:
        raise RuntimeError("StreamPETR parameters are not frozen: %s" % still_trainable)
    named = trainable_named_parameters(geocorr)
    if not named:
        raise RuntimeError("GeoCorr has no trainable parameters")
    detector_ids = {id(value) for _, value in detector_parameters}
    leaked = [name for name, value in named if id(value) in detector_ids]
    if leaked:
        raise RuntimeError("StreamPETR parameters entered GeoCorr optimizer: %s" % leaked)
    optimizer = torch.optim.AdamW([value for _, value in named], lr=lr)
    optimized_ids = {
        id(value) for group in optimizer.param_groups for value in group["params"]
    }
    detector_overlap = [
        name for name, value in detector_parameters if id(value) in optimized_ids
    ]
    if detector_overlap:
        raise RuntimeError(
            "StreamPETR parameters entered optimizer: %s" % detector_overlap
        )
    return optimizer


def parameter_report(detector: nn.Module, geocorr: nn.Module) -> Dict[str, Any]:
    trainable = trainable_named_parameters(geocorr)
    frozen = [name for name, _ in detector.named_parameters()]
    return {
        "trainable_modules": sorted({name.rsplit(".", 1)[0] for name, _ in trainable}),
        "trainable_parameters": [name for name, _ in trainable],
        "frozen_parameters": frozen,
        "trainable_parameter_count": sum(value.numel() for _, value in trainable),
    }


def frozen_detector_parameter_checksum(detector: nn.Module) -> str:
    """Return a deterministic checksum over names, dtypes, shapes, and values."""
    digest = hashlib.sha256()
    for name, parameter in detector.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _loss_scalar(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.mean()
    if isinstance(value, (list, tuple)) and value:
        return sum((_loss_scalar(item) for item in value[1:]), _loss_scalar(value[0]))
    raise TypeError("official loss component must be a tensor or non-empty tensor list")


def reduce_official_detection_losses(
    losses: Mapping[str, Any]
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Apply MMCV's loss-key convention while preserving every component."""
    components = {name: _loss_scalar(value) for name, value in losses.items()}
    selected = [value for name, value in components.items() if "loss" in name]
    if not selected:
        raise ValueError("official detector returned no loss components")
    total = sum(selected[1:], selected[0])
    return total, components


def streampetr_detection_losses(
    detector: nn.Module,
    recovered_fpn: Tensor,
    batch: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Call StreamPETR's official downstream training loss from recovered FPN.

    Ground truth is consumed only here; the GeoCorr forward signature contains
    no GT argument, making accidental oracle correspondence structurally hard.
    """
    required = (
        "gt_bboxes_3d", "gt_labels_3d", "gt_bboxes", "gt_labels",
        "img_metas", "centers2d", "depths",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError("training batch lacks official GT fields: %s" % missing)
    downstream = {
        name: value for name, value in batch.items()
        if name not in required and name != "img_feats"
    }
    downstream["img_feats"] = recovered_fpn
    # Do not use no_grad here: frozen detector operations must differentiate
    # with respect to recovered_fpn and hence the GeoCorr sidecar.
    return detector.forward_pts_train(
        batch["gt_bboxes_3d"],
        batch["gt_labels_3d"],
        batch["gt_bboxes"],
        batch["gt_labels"],
        batch["img_metas"],
        batch["centers2d"],
        batch["depths"],
        requires_grad=True,
        return_losses=True,
        **downstream,
    )


def compose_losses(
    official_losses: Mapping[str, Any],
    loss_corr: Tensor,
    loss_rec: Tensor,
    lambda_corr: float,
    lambda_rec: float,
) -> Stage3CLoss:
    loss_det, components = reduce_official_detection_losses(official_losses)
    total = loss_det + float(lambda_corr) * loss_corr + float(lambda_rec) * loss_rec
    return Stage3CLoss(total, loss_det, loss_corr, loss_rec, components)


def assert_finite_tensor(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError("non-finite tensor: %s" % name)


def assert_finite_parameters(module: nn.Module, label: str) -> None:
    for name, parameter in module.named_parameters():
        assert_finite_tensor("%s parameter %s" % (label, name), parameter)


def assert_finite_gradients(module: nn.Module, label: str) -> None:
    for name, parameter in module.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            assert_finite_tensor("%s gradient %s" % (label, name), parameter.grad)


def module_grad_norm(module: nn.Module) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return math.sqrt(float(sum(value.item() for value in squares)))


def snapshot_parameters(module: nn.Module) -> Dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.named_parameters() if value.requires_grad
    }


def parameter_update_norm(module: nn.Module, before: Mapping[str, Tensor]) -> float:
    total = 0.0
    for name, value in module.named_parameters():
        if value.requires_grad:
            total += float((value.detach().cpu() - before[name]).float().square().sum())
    return math.sqrt(total)


def tensor_l2_mean(value: Tensor, mask: Optional[Tensor] = None) -> float:
    norms = torch.linalg.norm(value.detach().float(), dim=-1)
    if mask is not None:
        selected = norms.masked_select(mask.bool())
        return float(selected.mean().item()) if selected.numel() else 0.0
    return float(norms.mean().item()) if norms.numel() else 0.0


def forward_diagnostics(output: Stage3CForward) -> Dict[str, float]:
    recovery = output.recovery
    valid = recovery.query_valid
    confidence = recovery.confidence.detach().squeeze(-1).masked_select(valid)
    zeros = {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    stats = zeros if not confidence.numel() else {
        "min": float(confidence.min().item()),
        "mean": float(confidence.mean().item()),
        "median": float(confidence.median().item()),
        "max": float(confidence.max().item()),
    }
    recovered_fpn = recovery.recovered_current_fpn
    dirty_fpn = recovery.current_dirty_fpn
    if recovered_fpn is None or dirty_fpn is None:
        raise RuntimeError("Stage 3-C requires recovered and dirty FPN tensors")
    return {
        "confidence_min": stats["min"],
        "confidence_mean": stats["mean"],
        "confidence_median": stats["median"],
        "confidence_max": stats["max"],
        "query_valid_ratio": float(valid.float().mean().item()),
        "delta_q_l2": tensor_l2_mean(recovery.delta_q, valid),
        "g_delta_q_l2": tensor_l2_mean(
            recovery.confidence * recovery.delta_q, valid
        ),
        "weighted_delta_q_l2": tensor_l2_mean(
            recovery.confidence * recovery.delta_q, valid
        ),
        "fpn_residual_l2": float(
            torch.linalg.norm(
                (recovered_fpn - dirty_fpn).detach().float()
            ).item()
        ),
    }


class ZeroGradientMonitor:
    """Hard-fail after a configurable run of silent zero-gradient steps."""

    def __init__(self, patience: int = 10, tolerance: float = 0.0) -> None:
        if patience <= 0:
            raise ValueError("zero-gradient patience must be positive")
        self.patience = int(patience)
        self.tolerance = float(tolerance)
        self.counts: Dict[str, int] = {}

    def update(self, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            self.counts[name] = self.counts.get(name, 0) + 1 if value <= self.tolerance else 0
            if self.counts[name] >= self.patience:
                raise RuntimeError("trainable module has zero gradient: %s" % name)


class JsonlLogger:
    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, values: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(values), sort_keys=True) + "\n")


def save_checkpoint(
    path: Union[str, Path],
    geocorr: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    config: Stage3CConfig,
    detector_checksum: str,
    pair_index: int = 0,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "geocorr": geocorr.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "pair_index": int(pair_index),
        "config": asdict(config),
        "detector_checksum": detector_checksum,
    }, str(target))


def load_checkpoint(
    path: Union[str, Path],
    geocorr: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_detector_checksum: str,
) -> Tuple[int, int, int]:
    payload = torch.load(str(path), map_location="cpu")
    if payload.get("detector_checksum") != expected_detector_checksum:
        raise RuntimeError("resume checkpoint was made with a different frozen detector")
    geocorr.load_state_dict(payload["geocorr"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return (
        int(payload["step"]), int(payload["epoch"]),
        int(payload.get("pair_index", 0)),
    )


def epoch_permutation(
    length: int, seed: int, epoch: int, shuffle: bool = True
) -> List[int]:
    """Return the reproducible per-epoch manifest permutation.

    Resume stores the cursor within this order. Rebuilding the order from
    seed + epoch restores the exact remaining suffix without serializing a
    large permutation into every checkpoint.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    order = list(range(length))
    if shuffle:
        random.Random(int(seed) + int(epoch)).shuffle(order)
    return order

def limit_pairs(records: Sequence[Any], max_pairs: Optional[int]) -> List[Any]:
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    return list(records if max_pairs is None else records[:max_pairs])


def reached_max_steps(step: int, max_steps: Optional[int]) -> bool:
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    return max_steps is not None and step >= max_steps


def deterministic_train_pipeline(pipeline: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Make StreamPETR train geometry deterministic while retaining GT transforms."""
    result = deepcopy(list(pipeline))
    seen_image = False
    seen_global = False
    for transform in result:
        kind = transform.get("type")
        if kind == "ResizeCropFlipRotImage":
            augmentation = deepcopy(transform["data_aug_conf"])
            height, width = augmentation["H"], augmentation["W"]
            final_height, final_width = augmentation["final_dim"]
            resize = max(final_height / height, final_width / width)
            augmentation.update(
                resize_lim=(resize, resize),
                bot_pct_lim=(0.0, 0.0),
                rot_lim=(0.0, 0.0),
                rand_flip=False,
            )
            transform["data_aug_conf"] = augmentation
            # Keep training=True so 2D GT receives the exact same transform.
            transform["training"] = True
            seen_image = True
        elif kind == "GlobalRotScaleTransImage":
            # Upstream's implementation samples even when training=False.
            transform.update(
                rot_range=(0.0, 0.0),
                scale_ratio_range=(1.0, 1.0),
                translation_std=(0.0, 0.0, 0.0),
            )
            seen_global = True
    if not seen_image or not seen_global:
        raise ValueError("expected StreamPETR image and global geometry transforms")
    return result


def _unwrap_data_container(value: Any) -> Any:
    """Unwrap MMCV-like ``.data`` wrappers without touching tensor data."""
    while not isinstance(value, (Tensor, np.ndarray, Mapping, list, tuple)) and hasattr(
        value, "data"
    ):
        value = value.data
    return value


def _value_type(value: Any) -> str:
    return "%s.%s" % (type(value).__module__, type(value).__name__)


def _value_shape(value: Any) -> Optional[Tuple[int, ...]]:
    if isinstance(value, (Tensor, np.ndarray)):
        return tuple(value.shape)
    return None


def _comparison_error(
    path: str,
    reason: str,
    clean_value: Any,
    dirty_value: Any,
    max_abs_diff: Optional[float] = None,
) -> RuntimeError:
    details = [
        "clean/dirty preprocessing mismatch at %s: %s" % (path, reason),
        "clean type=%s" % _value_type(clean_value),
        "dirty type=%s" % _value_type(dirty_value),
    ]
    clean_shape = _value_shape(clean_value)
    dirty_shape = _value_shape(dirty_value)
    if clean_shape is not None:
        details.append("clean shape=%s" % (clean_shape,))
    if dirty_shape is not None:
        details.append("dirty shape=%s" % (dirty_shape,))
    if max_abs_diff is not None:
        details.append("max abs diff=%g" % max_abs_diff)
    return RuntimeError("; ".join(details))


def _numeric_tensor(value: Any) -> Optional[Tensor]:
    """Convert exactly one numeric Tensor/ndarray leaf, never a container."""
    if isinstance(value, Tensor):
        return value.detach()
    if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
        return torch.from_numpy(value)
    return None


def _max_abs_diff(clean_value: Tensor, dirty_value: Tensor) -> float:
    if clean_value.numel() == 0:
        return 0.0
    if clean_value.dtype == torch.bool or dirty_value.dtype == torch.bool:
        return float((clean_value != dirty_value).to(torch.float32).max().item())
    dtype = (
        torch.complex128
        if torch.is_complex(clean_value) or torch.is_complex(dirty_value)
        else torch.float64
    )
    left = clean_value.detach().to(dtype=dtype, device="cpu")
    right = dirty_value.detach().to(dtype=dtype, device="cpu")
    return float((left - right).abs().max().item())


def _assert_nested_equal(name: str, clean_value: Any, dirty_value: Any) -> None:
    """Strictly compare paired metadata while retaining its nested structure."""
    clean_value = _unwrap_data_container(clean_value)
    dirty_value = _unwrap_data_container(dirty_value)

    clean_tensor = _numeric_tensor(clean_value)
    dirty_tensor = _numeric_tensor(dirty_value)
    if clean_tensor is not None and dirty_tensor is not None:
        if clean_tensor.shape != dirty_tensor.shape:
            raise _comparison_error(
                name, "shape mismatch", clean_value, dirty_value
            )
        floating = (
            torch.is_floating_point(clean_tensor)
            or torch.is_floating_point(dirty_tensor)
            or torch.is_complex(clean_tensor)
            or torch.is_complex(dirty_tensor)
        )
        if floating:
            common_dtype = torch.promote_types(clean_tensor.dtype, dirty_tensor.dtype)
            equal = torch.allclose(
                clean_tensor.to(dtype=common_dtype, device="cpu"),
                dirty_tensor.to(dtype=common_dtype, device="cpu"),
                atol=1e-6,
                rtol=1e-6,
            )
        else:
            equal = torch.equal(clean_tensor.cpu(), dirty_tensor.cpu())
        if not bool(equal):
            raise _comparison_error(
                name,
                "numeric mismatch",
                clean_value,
                dirty_value,
                _max_abs_diff(clean_tensor, dirty_tensor),
            )
        return

    if isinstance(clean_value, np.ndarray) and isinstance(dirty_value, np.ndarray):
        if clean_value.shape != dirty_value.shape:
            raise _comparison_error(
                name, "shape mismatch", clean_value, dirty_value
            )
        if not np.array_equal(clean_value, dirty_value):
            raise _comparison_error(name, "array mismatch", clean_value, dirty_value)
        return

    if isinstance(clean_value, Mapping) or isinstance(dirty_value, Mapping):
        if not isinstance(clean_value, Mapping) or not isinstance(dirty_value, Mapping):
            raise _comparison_error(name, "container type mismatch", clean_value, dirty_value)
        if set(clean_value) != set(dirty_value):
            raise _comparison_error(name, "dictionary key mismatch", clean_value, dirty_value)
        for key in clean_value:
            _assert_nested_equal("%s[%r]" % (name, key), clean_value[key], dirty_value[key])
        return

    if isinstance(clean_value, (list, tuple)) or isinstance(dirty_value, (list, tuple)):
        if type(clean_value) is not type(dirty_value):
            raise _comparison_error(name, "sequence type mismatch", clean_value, dirty_value)
        if len(clean_value) != len(dirty_value):
            raise _comparison_error(name, "sequence length mismatch", clean_value, dirty_value)
        for index, (clean_item, dirty_item) in enumerate(zip(clean_value, dirty_value)):
            _assert_nested_equal("%s[%d]" % (name, index), clean_item, dirty_item)
        return

    if clean_value != dirty_value:
        raise _comparison_error(name, "scalar mismatch", clean_value, dirty_value)


def _assert_nested_shape_equal(name: str, clean_value: Any, dirty_value: Any) -> None:
    """Compare image container/leaf shapes without comparing corrupted pixels."""
    clean_value = _unwrap_data_container(clean_value)
    dirty_value = _unwrap_data_container(dirty_value)
    if isinstance(clean_value, (Tensor, np.ndarray)) or isinstance(dirty_value, (Tensor, np.ndarray)):
        if not isinstance(clean_value, (Tensor, np.ndarray)) or not isinstance(
            dirty_value, (Tensor, np.ndarray)
        ) or _value_shape(clean_value) != _value_shape(dirty_value):
            raise _comparison_error(name, "shape mismatch", clean_value, dirty_value)
        return
    if isinstance(clean_value, Mapping) or isinstance(dirty_value, Mapping):
        if not isinstance(clean_value, Mapping) or not isinstance(dirty_value, Mapping):
            raise _comparison_error(name, "container type mismatch", clean_value, dirty_value)
        if set(clean_value) != set(dirty_value):
            raise _comparison_error(name, "dictionary key mismatch", clean_value, dirty_value)
        for key in clean_value:
            _assert_nested_shape_equal("%s[%r]" % (name, key), clean_value[key], dirty_value[key])
        return
    if isinstance(clean_value, (list, tuple)) or isinstance(dirty_value, (list, tuple)):
        if type(clean_value) is not type(dirty_value):
            raise _comparison_error(name, "sequence type mismatch", clean_value, dirty_value)
        if len(clean_value) != len(dirty_value):
            raise _comparison_error(name, "sequence length mismatch", clean_value, dirty_value)
        for index, (clean_item, dirty_item) in enumerate(zip(clean_value, dirty_value)):
            _assert_nested_shape_equal("%s[%d]" % (name, index), clean_item, dirty_item)
        return
    if type(clean_value) is not type(dirty_value):
        raise _comparison_error(name, "scalar type mismatch", clean_value, dirty_value)


def assert_synchronized_preprocessing(
    clean: Mapping[str, Any], dirty: Mapping[str, Any]
) -> None:
    """Strictly validate paired geometry while allowing different image pixels."""
    for name in (
        "lidar2img", "intrinsics", "extrinsics", "ego_pose", "ego_pose_inv"
    ):
        if name not in clean or name not in dirty:
            raise KeyError("paired batch lacks geometry field: %s" % name)
        _assert_nested_equal(name, clean[name], dirty[name])
    if "img" not in clean or "img" not in dirty:
        raise KeyError("paired batch lacks image field: img")
    _assert_nested_shape_equal("img", clean["img"], dirty["img"])


def gpu_memory_megabytes(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))


def now() -> float:
    return time.perf_counter()


__all__ = [
    "GeoCorrStage3CModel", "JsonlLogger", "Stage3CConfig", "Stage3CForward",
    "Stage3CLoss", "ZeroGradientMonitor", "assert_finite_gradients",
    "assert_finite_parameters", "assert_finite_tensor", "build_optimizer",
    "compose_losses", "forward_diagnostics", "freeze_detector",
    "frozen_detector_parameter_checksum", "gpu_memory_megabytes",
    "epoch_permutation", "limit_pairs", "load_checkpoint", "module_grad_norm", "now",
    "parameter_report", "parameter_update_norm", "reached_max_steps",
    "reduce_official_detection_losses", "save_checkpoint",
    "snapshot_parameters", "streampetr_detection_losses",
    "trainable_named_parameters",
    "deterministic_train_pipeline", "assert_synchronized_preprocessing",
]
