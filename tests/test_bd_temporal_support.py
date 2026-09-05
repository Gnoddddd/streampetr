import pandas as pd

from analysis.bd_temporal_support import classify_protocol, two_stage_cluster_bootstrap


def ci(low, high):
    return {"estimate": (low + high) / 2, "ci_low": low, "ci_high": high}


def test_preregistered_mechanism_patterns():
    assert classify_protocol(ci(.1, .2), ci(-.005, .005), ci(.02, .1), ci(-.1, .1))["mechanism"] == "clean_history_compensation"
    assert classify_protocol(ci(-.005, .005), ci(-.2, -.1), ci(-.1, .1), ci(-.1, -.02))["mechanism"] == "fault_history_contamination"
    assert classify_protocol(ci(.1, .2), ci(-.2, -.1), ci(.02, .1), ci(-.1, -.02))["mechanism"] == "both"


def test_two_stage_bootstrap_keeps_cluster_counts():
    rows = pd.DataFrame({
        "population": ["lost"] * 4,
        "scene_token": ["a", "a", "b", "b"],
        "instance_token": ["x", "x", "y", "z"],
        "value": [1.0, 3.0, 5.0, 7.0],
    })
    result = two_stage_cluster_bootstrap(rows, "value", "lost", n_boot=50, seed=1)
    assert result["n_gt"] == 4
    assert result["n_trajectories"] == 3
    assert result["n_scenes"] == 2
