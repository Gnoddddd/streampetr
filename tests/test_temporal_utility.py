import math

import pandas as pd

from analysis.temporal_utility import auroc, classify_trajectory, utility_values


def test_utility_identity_and_retention_gate():
    result = utility_values(.8, .5, .2)
    assert math.isclose(result["U_available"], .6)
    assert math.isclose(result["U_realized"], .3)
    assert math.isclose(result["TU_loss"], .3)
    assert math.isclose(result["TU_retention"], .5)
    assert math.isnan(utility_values(.1, .2, .2)["TU_retention"])


def test_prospective_trajectory_classification():
    lost = pd.DataFrame({"frame_idx": [3, 4], "A_tp": [True, True], "D_tp": [True, False]})
    retained = pd.DataFrame({"frame_idx": [3, 4], "A_tp": [True, False], "D_tp": [True, True]})
    ambiguous = pd.DataFrame({"frame_idx": [3], "A_tp": [False], "D_tp": [False]})
    assert classify_trajectory(lost) == ("future_lost", 4)
    assert classify_trajectory(retained) == ("always_retained", None)
    assert classify_trajectory(ambiguous) == ("ambiguous_clean_failure", None)


def test_auroc_ties():
    assert auroc([0, 1], [0, 1]) == 1.0
    assert auroc([0, 1], [1, 0]) == 0.0
    assert auroc([0, 1], [1, 1]) == 0.5
