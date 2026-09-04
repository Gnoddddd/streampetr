import numpy as np
import pandas as pd

from analysis.fault_stress_reserve import risk_bin, stress_reserve, two_stage_bootstrap


def test_stress_reserve_is_fixed_margin_difference():
    value = stress_reserve(.8, .2, .5)
    assert np.allclose([value["R_K"], value["J_p"], value["M_p"]], [.6, .3, -.3])


def test_fixed_risk_bins_are_left_closed():
    assert risk_bin([-.3, -.2, -.1, 0, .1, .2]).tolist() == [0, 1, 2, 3, 4, 5]


def test_two_stage_bootstrap_is_reproducible():
    rows = pd.DataFrame({"scene_token": ["a", "a", "b", "b"],
                         "instance_token": ["1", "2", "3", "4"],
                         "x": [1., 2., 3., 4.]})
    statistic = lambda values: float(np.mean(values[:, 0].astype(float)))
    left = two_stage_bootstrap(rows, ["x"], statistic, n_boot=20, seed=7)
    right = two_stage_bootstrap(rows, ["x"], statistic, n_boot=20, seed=7)
    assert np.array_equal(left, right)
