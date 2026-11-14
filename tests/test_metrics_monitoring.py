import numpy as np
from credit_scoring.metrics import calculate_metrics
from credit_scoring.monitoring import calculate_psi


def test_metrics():
    result = calculate_metrics([0, 1, 1, 0], [0.1, 0.8, 0.7, 0.2])
    assert result["roc_auc"] == 1.0
    assert result["f1"] == 1.0


def test_psi_same_distribution():
    values = np.linspace(0, 1, 1000)
    assert calculate_psi(values, values) < 1e-8


def test_psi_shift():
    a = np.random.default_rng(1).normal(0, 1, 1000)
    b = np.random.default_rng(2).normal(2, 1, 1000)
    assert calculate_psi(a, b) > 0.1
