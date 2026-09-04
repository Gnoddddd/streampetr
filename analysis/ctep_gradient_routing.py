"""Fixed-classifier CTEP routing helpers for the frozen StreamPETR graph."""

from __future__ import annotations

from collections import OrderedDict
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def frozen_sequential_classifier(module: nn.Sequential, query: torch.Tensor) -> torch.Tensor:
    """Run the real classifier mapping while detaching all weights and buffers.

    StreamPETR's configured classification branch is deliberately handled by
    supported primitive types only.  Refusing an unknown child prevents a
    silent approximation of the model's real mapping.
    """

    value = query
    for child in module:
        if isinstance(child, nn.Linear):
            bias = None if child.bias is None else child.bias.detach()
            value = F.linear(value, child.weight.detach(), bias)
        elif isinstance(child, nn.LayerNorm):
            weight = None if child.weight is None else child.weight.detach()
            bias = None if child.bias is None else child.bias.detach()
            value = F.layer_norm(value, child.normalized_shape, weight, bias, child.eps)
        elif isinstance(child, nn.ReLU):
            value = F.relu(value, inplace=False)
        else:
            raise TypeError(f"unsupported classification child: {type(child).__name__}")
    return value


def stream_petr_parameter_groups(head) -> tuple[OrderedDict, OrderedDict]:
    """Resolve stopped-head and predeclared real upstream parameter views."""

    named = OrderedDict(head.named_parameters())
    classifier = OrderedDict(
        (name, parameter)
        for name, parameter in named.items()
        if name.startswith("cls_branches.0.")
    )
    groups = OrderedDict()
    groups["final_decoder_layer_5"] = OrderedDict(
        (name, parameter)
        for name, parameter in named.items()
        if name.startswith("transformer.decoder.layers.5.")
    )
    groups["final_decoder_temporal_self_attention"] = OrderedDict(
        (name, parameter)
        for name, parameter in named.items()
        if name.startswith("transformer.decoder.layers.5.attentions.0.")
    )
    temporal_pattern = re.compile(
        r"^transformer\.decoder\.layers\.[0-5]\.attentions\.0\."
    )
    groups["all_decoder_temporal_self_attention"] = OrderedDict(
        (name, parameter)
        for name, parameter in named.items()
        if temporal_pattern.match(name)
    )
    alignment_prefixes = (
        "query_embedding.",
        "time_embedding.",
        "ego_pose_pe.",
        "ego_pose_memory.",
    )
    groups["temporal_alignment_modules"] = OrderedDict(
        (name, parameter)
        for name, parameter in named.items()
        if name.startswith(alignment_prefixes)
    )
    if not classifier:
        raise RuntimeError("shared classification-head parameter group is empty")
    empty = [name for name, parameters in groups.items() if not parameters]
    if empty:
        raise RuntimeError(f"empty upstream parameter groups: {empty}")
    branches = list(head.cls_branches)
    if not branches or any(branch is not branches[0] for branch in branches[1:]):
        raise RuntimeError("configured StreamPETR classification branch is not shared")
    return classifier, groups


def unique_parameters(*mappings) -> list[torch.nn.Parameter]:
    """Return identity-deduplicated parameters, preserving declared order."""

    result = []
    seen = set()
    for mapping in mappings:
        values = mapping.values() if hasattr(mapping, "values") else mapping
        for parameter in values:
            if id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    return result

