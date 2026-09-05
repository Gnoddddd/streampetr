import math

from analysis.fault_assignment_gradient import (
    bootstrap_selection_difference,
    scalar_gradient_relation,
    select_equal_budget,
    selection_metrics,
    vector_gradient_relation,
)


def test_scalar_gradient_relation_detects_harmful_reversal():
    value = scalar_gradient_relation(0.2, -0.5)
    assert value["harmful_current"]
    assert value["gradient_conflict"]
    assert value["harmful_reversed"]
    assert math.isclose(value["combined_update"], 0.3)


def test_vector_gradient_relation_tracks_same_gt_projection():
    value = vector_gradient_relation([-2.0, 0.0], [1.0, 0.0])
    assert value["current_conflict"]
    assert not value["combined_desired_positive"]
    assert value["current_aux_cosine"] == -1.0
    value = vector_gradient_relation([0.0, 0.0], [1.0, 0.0])
    assert value["combined_desired_positive"]
    assert value["current_aux_cosine"] == 0.0


def _rows():
    return [
        {"sample_token": "s1", "gt_token": "g1", "pair_cost": 0.1,
         "non_same_layer_count": 0, "final_non_same": False, "fault_margin": 0.3,
         "lost_degraded": False, "boundary_crossing": False,
         "retained": True, "easy_retained": True},
        {"sample_token": "s2", "gt_token": "g2", "pair_cost": 0.4,
         "non_same_layer_count": 6, "final_non_same": True, "fault_margin": -0.1,
         "lost_degraded": True, "boundary_crossing": True,
         "retained": False, "easy_retained": False},
        {"sample_token": "s3", "gt_token": "g3", "pair_cost": 0.2,
         "non_same_layer_count": 3, "final_non_same": True, "fault_margin": 0.1,
         "lost_degraded": False, "boundary_crossing": False,
         "retained": True, "easy_retained": False},
    ]


def test_equal_budget_selection_separates_generic_and_selective():
    selected = select_equal_budget(_rows(), 1)
    assert selected["generic"] == {("s1", "g1")}
    assert selected["selective"] == {("s2", "g2")}
    generic = selection_metrics(_rows(), selected["generic"])
    selective = selection_metrics(_rows(), selected["selective"])
    assert generic["easy_retained"] == 1.0
    assert selective["lost_degraded"] == 1.0
    assert selective["concentration"] > generic["concentration"]


def test_selection_bootstrap_is_seeded_and_positive():
    selected = select_equal_budget(_rows(), 1)
    first = bootstrap_selection_difference(
        _rows(), selected["generic"], selected["selective"],
        "concentration", 7, 200,
    )
    second = bootstrap_selection_difference(
        _rows(), selected["generic"], selected["selective"],
        "concentration", 7, 200,
    )
    assert first == second
    assert first["estimate"] == 2.0
