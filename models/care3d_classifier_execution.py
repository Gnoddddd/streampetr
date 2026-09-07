"""CARE-3D P1 classifier execution compatibility helpers.

The StreamPETR final classification branch is numerically shape-sensitive on the
formal CUDA stack.  P1 therefore keeps the deployed ``[1, 900, D]`` classifier
invocation shape whenever it replays a standalone batch of object queries.

The shim changes only 2-D standalone calls. Native 3-D detector calls pass
through to the original ``forward`` method unchanged, preserving hooks, module
identity, checkpoint keys and the deployed detector graph.
"""

from __future__ import annotations

import types
from typing import Any

import torch
from torch import Tensor, nn


CLASSIFIER_EXECUTION_POLICY = "packed_deployment_shape_900_v1"
DEPLOYED_QUERY_COUNT = 900


def packed_deployment_shape_forward(
    original_forward,
    query: Tensor,
    *args: Any,
    query_count: int = DEPLOYED_QUERY_COUNT,
    **kwargs: Any,
) -> Tensor:
    """Replay a 2-D query batch through the deployed 900-query classifier shape.

    ``query`` has shape ``[B, D]``. It is placed in the first ``B`` rows of a
    differentiable ``[1, query_count, D]`` tensor; the remaining rows are zeros.
    The classifier is executed once at the deployed GEMM shape and the first
    ``B`` output rows are returned.  Concatenation is used instead of an
    in-place copy so gradients from routed P1 queries propagate exactly.
    """

    if query.ndim != 2:
        raise ValueError("packed deployment replay expects [B, D] queries")
    batch, width = int(query.shape[0]), int(query.shape[1])
    if batch <= 0:
        raise ValueError("packed deployment replay requires a non-empty batch")
    if batch > int(query_count):
        raise ValueError(
            f"packed deployment replay batch {batch} exceeds query_count {query_count}"
        )
    padding = query.new_zeros((int(query_count) - batch, width))
    packed = torch.cat((query, padding), dim=0).unsqueeze(0)
    logits = original_forward(packed, *args, **kwargs)
    if logits.ndim != 3 or int(logits.shape[0]) != 1 \
            or int(logits.shape[1]) != int(query_count):
        raise RuntimeError("classifier output shape changed under deployment-shape replay")
    return logits[0, :batch]


def install_deployment_shape_forward(
    classifier: nn.Module,
    *,
    query_count: int = DEPLOYED_QUERY_COUNT,
) -> nn.Module:
    """Patch one classifier instance while preserving native 3-D calls exactly."""

    installed = getattr(classifier, "_care3d_classifier_execution_policy", None)
    if installed is not None:
        if installed != CLASSIFIER_EXECUTION_POLICY:
            raise RuntimeError(f"classifier already has incompatible execution policy: {installed}")
        return classifier

    original_forward = classifier.forward

    def _forward(self, query: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        if query.ndim == 2:
            return packed_deployment_shape_forward(
                original_forward,
                query,
                *args,
                query_count=int(query_count),
                **kwargs,
            )
        return original_forward(query, *args, **kwargs)

    classifier.forward = types.MethodType(_forward, classifier)
    classifier._care3d_classifier_execution_policy = CLASSIFIER_EXECUTION_POLICY
    classifier._care3d_deployed_query_count = int(query_count)
    return classifier
