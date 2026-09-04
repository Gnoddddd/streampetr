import math

from analysis.temporal_state_counterfactual import (
    cluster_bootstrap_median,
    cluster_bootstrap_spearman,
    temporal_decision,
)


def test_cluster_bootstrap_is_seeded_and_preserves_frame_clusters():
    rows = [
        {"protocol": "p", "sample": "a", "value": 1.0},
        {"protocol": "p", "sample": "a", "value": 3.0},
        {"protocol": "p", "sample": "b", "value": 5.0},
    ]
    first = cluster_bootstrap_median(rows, "value", ("protocol", "sample"), 7, 100)
    second = cluster_bootstrap_median(rows, "value", ("protocol", "sample"), 7, 100)
    assert first == second
    assert first["estimate"] == 3.0


def test_cluster_bootstrap_spearman_tracks_direction():
    rows = [{"sample": str(i), "age": i, "effect": i} for i in range(1, 8)]
    value = cluster_bootstrap_spearman(rows, "age", "effect", ("sample",), 9, 100)
    assert math.isclose(value["estimate"], 1.0)
    assert value["ci_low"] > 0


def _inputs():
    protocols = {p: {"bd": .02, "ca": -.02} for p in ("d", "b", "c")}
    age = {p: {"bd": .4, "ca": -.4} for p in protocols}
    pooled = {"bd_ci_low": .1, "ca_ci_high": -.1}
    equivalence = {"bd_ci_low": -.002, "bd_ci_high": .002,
                   "ca_ci_low": -.002, "ca_ci_high": .002,
                   "bd_topk_discordance": 0, "bd_tp_discordance": 0,
                   "ca_topk_discordance": 0, "ca_tp_discordance": 0}
    return protocols, age, pooled, equivalence


def test_temporal_decision_supports_strong_clean_history_benefit():
    protocols, age, pooled, equivalence = _inputs()
    value = temporal_decision(
        {"median": .02, "ci_low": .01, "topk_event_rate": .3, "tp_event_rate": .3},
        {"median": -.005, "ci_high": .01, "topk_event_rate": 0, "tp_event_rate": 0},
        protocols, age, pooled, equivalence,
    )
    assert value["bd_arm"]
    assert value["decision"] == "GO_TEMPORAL_STATE_CONTAMINATION"


def test_temporal_decision_closes_stage4_on_equivalence():
    protocols, age, pooled, equivalence = _inputs()
    protocols = {p: {"bd": .001, "ca": -.001} for p in protocols}
    age = {p: {"bd": float("nan"), "ca": float("nan")} for p in age}
    value = temporal_decision(
        {"median": .001, "ci_low": -.002, "topk_event_rate": 0, "tp_event_rate": 0},
        {"median": -.001, "ci_high": .002, "topk_event_rate": 0, "tp_event_rate": 0},
        protocols, age, {"bd_ci_low": -1, "ca_ci_high": 1}, equivalence,
    )
    assert value["current_state_equivalent"]
    assert value["decision"] == "NO_GO_CURRENT_OBSERVATION_DOMINANT_CLOSE_STAGE4"
