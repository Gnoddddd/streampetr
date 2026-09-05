import math

import torch

from analysis.temporal_state_attribution import (
    assert_one_component_swap,
    decide_attribution,
    explanation_ratio,
    swap_one_component,
)


def test_swap_replaces_exactly_one_tensor():
    base = {"embedding": torch.tensor([1.0]), "position": torch.tensor([2.0])}
    donor = {"embedding": torch.tensor([3.0]), "position": torch.tensor([4.0])}
    swapped = swap_one_component(base, donor, "embedding")
    assert assert_one_component_swap(base, donor, swapped, "embedding") == 2.0
    assert torch.equal(swapped["embedding"], donor["embedding"])
    assert torch.equal(swapped["position"], base["position"])
    assert swapped["embedding"].data_ptr() != donor["embedding"].data_ptr()


def test_explanation_ratio_keeps_raw_overshoot_and_clips_readable_value():
    overshoot = explanation_ratio(.3, .2)
    assert math.isclose(overshoot["raw"], 1.5)
    assert overshoot["clipped"] == 1.0
    assert explanation_ratio(-.1, -.2) == {"raw": .5, "clipped": .5}


def record(component, ratio=.7, core=True):
    value = {"component": component,
             "bd_lost": .04, "bd_retained": .002,
             "bd_enrichment": .038, "bd_ci_low": .02,
             "bd_enrichment_ci_low": .02, "bd_cross_protocol": True,
             "bd_spos_ratio": ratio, "bd_topk_ratio": ratio, "bd_tp_ratio": ratio,
             "ca_lost": -.04, "ca_retained": -.002,
             "ca_enrichment": -.038, "ca_ci_high": -.02,
             "ca_enrichment_ci_high": -.02, "ca_cross_protocol": True,
             "ca_spos_ratio": ratio, "ca_topk_ratio": ratio, "ca_tp_ratio": ratio}
    if not core:
        value["bd_cross_protocol"] = False
    return value


def test_decision_selects_component_that_passes_both_arms():
    result = decide_attribution([record("embedding"), record("position", core=False)])
    assert result["decision"] == "GO_DOMINANT_TEMPORAL_STATE_COMPONENT"
    assert result["selected_components"] == ["embedding"]


def test_decision_allows_two_sparse_core_components():
    result = decide_attribution([record("embedding", .3), record("position", .3)])
    assert result["decision"] == "GO_SPARSE_TEMPORAL_STATE_COMPONENT_SET"
    assert result["selected_components"] == ["embedding", "position"]


def test_decision_rejects_retained_sensitive_component():
    value = record("embedding")
    value.update({"bd_retained": .03, "ca_retained": -.03,
                  "bd_enrichment": .01, "ca_enrichment": -.01})
    result = decide_attribution([value])
    assert result["decision"] == "NO_GO_TEMPORAL_STATE_ATTRIBUTION"
