import numpy as np
import pandas as pd

from analysis.prospective_failure_decodability import (
    classification_metrics, ece_equal_width, train_standardize,
)


def test_perfect_probabilities_have_perfect_ranking_and_zero_calibration_error():
    result = classification_metrics([0, 0, 1, 1], [0., 0., 1., 1.])
    assert result["auprc"] == 1.
    assert result["auroc"] == 1.
    assert result["brier"] == 0.
    assert result["ece10"] == 0.


def test_ece_uses_fixed_equal_width_bins():
    assert np.isclose(ece_equal_width([0, 1], [.25, .75], 10), .25)


def test_standardization_uses_train_statistics_and_guards_constant_columns():
    train = np.asarray([[1., 4.], [3., 4.]])
    other = np.asarray([[5., 4.]])
    train_z, other_z, mean, scale = train_standardize(train, other)
    assert np.allclose(mean, [2., 4.])
    assert np.allclose(scale, [1., 1.])
    assert np.allclose(train_z, [[-1., 0.], [1., 0.]])
    assert np.allclose(other_z, [[3., 0.]])
