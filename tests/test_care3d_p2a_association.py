import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from analysis.care3d_p2a_association import (
    P2A_QUERY_COLLISION_POLICY,
    AssociationConfig,
    association_cost_components,
    association_grid,
    assert_query_layout,
    assignment_rows,
    baseline_configs,
    filter_p2a_rows,
    hungarian_with_unmatched,
    oracle_diagnostics,
    p2a_protocol_gate,
    select_global_config,
    transform_lidar_centers_between_frames,
    weighted_cost,
)


def test_p2a_query_layout_is_frozen_to_644_plus_256():
    assert_query_layout(644, 256, 900)
    with pytest.raises(RuntimeError):
        assert_query_layout(644, 255, 899)


def test_p2a_collision_policy_excludes_all_shared_anchor_or_target_rows():
    frame = pd.DataFrame({
        "sample_id": ["a", "b", "c", "d", "e", "f"],
        "anchor_frame_idx": [2, 2, 2, 3, 3, 4],
        "target_frame_idx": [3, 3, 3, 4, 4, 5],
        "anchor_query_index": [10, 10, 11, 12, 13, 14],
        "target_clean_query_index": [20, 21, 22, 30, 30, 31],
    })
    arrays = {
        "object_features": np.arange(12, dtype=np.float32).reshape(6, 2),
        "labels": np.arange(6, dtype=np.int64),
        "scalar": np.asarray(7),
    }
    filtered, packed, audit = filter_p2a_rows(frame, arrays)
    assert audit["policy"] == P2A_QUERY_COLLISION_POLICY
    assert filtered.sample_id.tolist() == ["c", "f"]
    assert packed["labels"].tolist() == [2, 5]
    assert int(packed["scalar"]) == 7
    assert audit["anchor_query_collision_excluded_rows"] == 2
    assert audit["target_query_collision_excluded_rows"] == 2
    assert audit["total_excluded_rows"] == 4


def test_lidar_center_transform_uses_only_pose_geometry():
    identity = np.eye(3)
    source = {
        "lidar2ego_rotation": identity,
        "lidar2ego_translation": np.zeros(3),
        "ego2global_rotation": identity,
        "ego2global_translation": np.asarray([10.0, 0.0, 0.0]),
    }
    target = {
        "lidar2ego_rotation": identity,
        "lidar2ego_translation": np.zeros(3),
        "ego2global_rotation": identity,
        "ego2global_translation": np.asarray([8.0, 0.0, 0.0]),
    }
    value = transform_lidar_centers_between_frames(
        np.asarray([[1.0, 2.0, 0.0]], dtype=np.float32), source, target
    )
    assert np.allclose(value, np.asarray([[3.0, 2.0, 0.0]], dtype=np.float32))


def test_association_cost_is_vectorized_and_has_no_oracle_or_gt_argument():
    parameters = set(inspect.signature(association_cost_components).parameters)
    forbidden = {
        "oracle_query", "oracle_queries", "target_clean_query_index",
        "gt", "gt_center", "gt_class", "clean_future", "clean_target_query",
    }
    assert parameters.isdisjoint(forbidden)

    anchors = torch.zeros(2, 256)
    anchors[0, 0] = 1.0
    anchors[1, 1] = 1.0
    fault = torch.zeros(900, 256)
    fault[0, 0] = 1.0
    fault[1, 1] = 1.0
    anchor_centers = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    fault_centers = torch.zeros(900, 3)
    fault_centers[0] = torch.tensor([0.0, 0.0, 0.0])
    fault_centers[1] = torch.tensor([1.0, 0.0, 0.0])
    fault_centers[2:] = torch.tensor([20.0, 0.0, 0.0])
    logits = torch.zeros(900, 10)
    logits[0, 3] = 8.0
    logits[1, 4] = 8.0
    classes = torch.tensor([3, 4])

    components = association_cost_components(
        anchors, anchor_centers, classes, fault, logits, fault_centers
    )
    assert components["geometry"].shape == (2, 900)
    assert components["embedding"].shape == (2, 900)
    assert components["class"].shape == (2, 900)
    assert components["geometry_allowed"].shape == (2, 900)
    assert components["embedding"][0, 0].item() == pytest.approx(0.0, abs=1e-7)
    assert components["embedding"][1, 1].item() == pytest.approx(0.0, abs=1e-7)
    assert not bool(components["geometry_allowed"][0, 2])

    config = AssociationConfig(0.5, 0.3, 0.2, 0.45)
    cost = weighted_cost(components, config)
    assert torch.isfinite(cost[:, :2]).all()
    assert torch.isinf(cost[:, 2:]).all()


def test_hungarian_is_one_to_one_and_supports_unmatched():
    cost = torch.full((3, 900), float("inf"))
    cost[0, 0] = 0.10
    cost[0, 1] = 0.25
    cost[1, 0] = 0.11
    cost[1, 1] = 0.12
    # Row 2 has no feasible real query and must use its private dummy.
    assignment = hungarian_with_unmatched(cost, 0.30)
    selected = assignment["selected_query"]
    assert selected[2] == -1
    real = selected[selected >= 0]
    assert len(real) == len(set(real.tolist()))
    assert set(real.tolist()) == {0, 1}


def test_oracle_diagnostics_and_assignment_outcomes_are_offline_only():
    cost = torch.full((2, 900), float("inf"))
    cost[0, 4] = 0.10
    cost[0, 5] = 0.20
    cost[1, 8] = 0.30
    cost[1, 9] = 0.15
    oracle = np.asarray([4, 8], dtype=np.int64)
    diagnostics = oracle_diagnostics(cost, oracle)
    assert diagnostics["oracle_rank"].tolist() == [1, 2]
    assert diagnostics["correct_vs_best_wrong_margin"][0] == pytest.approx(0.10)
    assert diagnostics["correct_vs_best_wrong_margin"][1] == pytest.approx(-0.15)

    assignment = {
        "selected_query": np.asarray([4, 9]),
        "selected_cost": np.asarray([0.10, 0.15]),
        "matched": np.asarray([True, True]),
    }
    rows = assignment_rows(assignment, oracle, diagnostics)
    assert rows["exact_match"].tolist() == [1, 0]
    assert rows["wrong_match"].tolist() == [0, 1]
    assert rows["unmatched"].tolist() == [0, 0]


def test_frozen_grid_has_15_global_configs_and_baselines_share_threshold():
    grid = association_grid()
    assert len(grid) == 15
    assert len({config.config_id for config in grid}) == 15
    baselines = baseline_configs(0.45)
    assert [name for name, _ in baselines] == [
        "geometry_only", "embedding_only", "class_geometry"
    ]
    assert all(config.max_cost == 0.45 for _, config in baselines)


def test_train_selection_uses_macro_recall_then_frozen_tiebreaks():
    rows = []
    configs = association_grid()
    for config in configs:
        for protocol in ("blur_back", "crash_back", "dark_back"):
            exact_n = 70
            wrong_n = 10
            accepted_n = 80
            accepted_cost_sum = 16.0
            # Make one preregistered config uniquely best on the primary metric.
            if config.config_id == configs[7].config_id:
                exact_n = 75
            rows.append({
                "protocol": protocol,
                "config_id": config.config_id,
                "rows": 100,
                "exact_n": exact_n,
                "wrong_n": wrong_n,
                "accepted_n": accepted_n,
                "accepted_cost_sum": accepted_cost_sum,
            })
    result = select_global_config(pd.DataFrame(rows))
    assert result["selected"]["config_id"] == configs[7].config_id
    assert "probe" not in result["selection_rule"]


def test_p2a_gate_requires_recall_both_cluster_bounds_and_wrong_control():
    point = {"exact_recall": 0.75, "wrong_match_rate": 0.08}
    scene = {"ci_low": 0.60}
    instance = {"ci_low": 0.58}
    flags = p2a_protocol_gate(
        point,
        scene,
        instance,
        min_exact_recall=0.70,
        min_cluster_ci_low=0.50,
        max_wrong_match_rate=0.10,
    )
    assert flags["protocol_pass"] is True
    failed = dict(point)
    failed["wrong_match_rate"] = 0.11
    flags = p2a_protocol_gate(
        failed,
        scene,
        instance,
        min_exact_recall=0.70,
        min_cluster_ci_low=0.50,
        max_wrong_match_rate=0.10,
    )
    assert flags["wrong_match_pass"] is False
    assert flags["protocol_pass"] is False
