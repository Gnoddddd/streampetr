import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from analysis.care3d_p2a2_lineage import (
    FROZEN_P2A0_CONFIG,
    evaluate_assignments,
    lineage_assignments,
    lineage_first_assign,
    paired_delta_bootstrap,
    query_origin,
    recompute_topk_indexes,
    topk_gather_exact,
    verify_post_update_memory,
)
from analysis.care3d_p2a_association import (
    assert_query_layout,
    hungarian_with_unmatched,
)
from scripts.export_care3d_p2a2_lineage_r0 import (
    parse_args,
    parse_lineage_split,
    require_engineering_smoke,
)


def frozen_topk(*queries: int) -> torch.Tensor:
    remaining = [query for query in range(900) if query not in queries]
    values = list(queries) + remaining[: 256 - len(queries)]
    return torch.tensor(values, dtype=torch.long).view(1, 256, 1)


def test_topk_ancestry_maps_position_to_644_plus_j():
    indexes = frozen_topk(31, 700, 18)
    result = lineage_assignments([700, 31, 18], indexes)
    assert result["lineage_available"].tolist() == [True, True, True]
    assert result["lineage_position"].tolist() == [1, 0, 2]
    assert result["lineage_child_query"].tolist() == [645, 644, 646]


def test_anchor_outside_topk_has_no_lineage_child():
    result = lineage_assignments([899], frozen_topk(1, 2, 3))
    assert result["anchor_topk_member"].tolist() == [False]
    assert result["lineage_available"].tolist() == [False]
    assert result["lineage_position"].tolist() == [-1]
    assert result["lineage_child_query"].tolist() == [-1]


def test_lineage_children_are_unique_within_frame():
    with pytest.raises(RuntimeError, match="anchor queries must be unique"):
        lineage_assignments([10, 10], frozen_topk(10))
    result = lineage_assignments([10, 11], frozen_topk(10, 11))
    children = result["lineage_child_query"]
    assert len(children) == len(set(children.tolist()))


def test_lineage_reservation_removes_child_from_frozen_fallback_pool():
    cost = torch.full((2, 900), float("inf"))
    cost[0, 50] = 0.1
    cost[1, 644] = 0.01  # Best candidate, but reserved by row 0 lineage.
    cost[1, 7] = 0.02
    result = lineage_first_assign([50, 899], frozen_topk(50), cost)
    assert result["hybrid_selected_query"].tolist() == [644, 7]
    assert result["hybrid_source"].tolist() == ["lineage", "p2a0_fallback"]
    assert result["fallback_selected_cost"][1] == pytest.approx(0.02)


def test_hybrid_assignment_is_frame_level_one_to_one():
    cost = torch.full((4, 900), float("inf"))
    cost[2, 8] = 0.1
    cost[2, 9] = 0.2
    cost[3, 8] = 0.11
    cost[3, 9] = 0.12
    result = lineage_first_assign([20, 21, 898, 899], frozen_topk(20, 21), cost)
    selected = result["hybrid_selected_query"]
    real = selected[selected >= 0]
    assert len(real) == len(set(real.tolist()))
    assert set(real.tolist()) == {644, 645, 8, 9}


def test_gt_clean_future_and_oracle_cannot_enter_online_assignment():
    parameters = set(inspect.signature(lineage_first_assign).parameters)
    forbidden = {
        "gt", "ground_truth", "oracle_query", "oracle_queries",
        "target_clean_query_index", "clean_future", "clean_output",
    }
    assert parameters.isdisjoint(forbidden)
    assert "oracle_queries" in inspect.signature(evaluate_assignments).parameters


def test_cli_cannot_access_probe_val_or_probe_test():
    assert parse_lineage_split("probe_train") == "probe_train"
    with pytest.raises(Exception):
        parse_lineage_split("probe_val")
    with pytest.raises(Exception):
        parse_lineage_split("probe_test")
    assert parse_args(["--split", "probe_train"]).split == "probe_train"
    with pytest.raises(SystemExit):
        parse_args(["--split", "probe_val"])
    with pytest.raises(SystemExit):
        parse_args(["--split", "probe_test"])


def test_probe_train_entrypoint_is_locked_by_engineering_smoke(monkeypatch, tmp_path):
    import scripts.export_care3d_p2a2_lineage_r0 as exporter

    monkeypatch.setattr(exporter, "REPORT", tmp_path)
    with pytest.raises(RuntimeError, match="locked pending engineering smoke"):
        require_engineering_smoke()


def test_query_layout_is_exactly_644_plus_256_equals_900():
    assert_query_layout(644, 256, 900)
    assert query_origin(0) == "current"
    assert query_origin(643) == "current"
    assert query_origin(644) == "propagated"
    assert query_origin(899) == "propagated"
    with pytest.raises(ValueError):
        query_origin(900)


def paired_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "scene_token": ["s1", "s1", "s2", "s2"],
        "instance_token": ["i1", "i2", "i3", "i4"],
        "p2a0_exact": [0, 1, 0, 1],
        "hybrid_exact": [1, 1, 1, 0],
        "p2a0_wrong": [1, 0, 1, 0],
        "hybrid_wrong": [0, 0, 0, 1],
    })


def test_paired_delta_statistics_use_rowwise_method_differences():
    result = paired_delta_bootstrap(
        paired_fixture(), cluster_column="scene_token", repetitions=5000, seed=71
    )
    assert result["delta_exact"]["estimate"] == pytest.approx(0.25)
    assert result["delta_wrong"]["estimate"] == pytest.approx(-0.25)
    assert result["delta_exact"]["finite_bootstraps"] == 5000


def test_cluster_bootstrap_is_deterministic():
    first = paired_delta_bootstrap(
        paired_fixture(), cluster_column="instance_token", repetitions=5000, seed=99
    )
    second = paired_delta_bootstrap(
        paired_fixture(), cluster_column="instance_token", repetitions=5000, seed=99
    )
    assert first == second
    with pytest.raises(ValueError):
        paired_delta_bootstrap(
            paired_fixture(), cluster_column="row", repetitions=5000, seed=99
        )


def test_frozen_p2a0_baseline_matches_unmodified_original_solver():
    torch.manual_seed(19)
    cost = torch.rand(3, 900)
    indexes = frozen_topk(0, 1)
    direct = hungarian_with_unmatched(cost, FROZEN_P2A0_CONFIG.max_cost)
    hybrid = lineage_first_assign([0, 1, 899], indexes, cost)
    assert np.array_equal(direct["selected_query"], hybrid["p2a0_selected_query"])
    assert np.allclose(
        direct["selected_cost"], hybrid["p2a0_selected_cost"],
        rtol=0.0, atol=0.0, equal_nan=True,
    )


def test_topk_recomputation_and_memory_prefix_equivalence_are_exact():
    torch.manual_seed(23)
    logits = torch.randn(1, 900, 10)
    outs_dec = torch.randn(1, 900, 256)
    indexes = recompute_topk_indexes(logits, 256)
    prefix = topk_gather_exact(outs_dec, indexes)
    old = torch.randn(1, 1024, 256)
    memory = torch.cat([prefix, old], dim=1)
    check = verify_post_update_memory(
        outs_dec, logits, memory, topk_proposals=256
    )
    assert check["torch_equal"] is True
    assert check["max_abs_diff"] == 0.0
