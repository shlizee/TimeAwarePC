"""Regression test: partial_corr should be shift-invariant.

Adding a constant to the input data should not change the partial correlation,
because the underlying linear regression now includes an intercept term.
"""
import os
import sys

# Insert project root before any installed timeawarepc so the tests run
# against the in-repo source, not any pre-installed package version.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from timeawarepc.pcalg import partial_corr as partial_corr_pcalg
from timeawarepc.pcalg_helpers import partial_corr as partial_corr_helpers


def _make_test_data(seed: int = 0, n: int = 200):
    rng = np.random.default_rng(seed)
    # 3 variables; var 2 is a linear combination of vars 0 and 1 plus noise
    data = rng.standard_normal((n, 3))
    data[:, 2] = data[:, 0] + 0.5 * data[:, 1] + 0.1 * rng.standard_normal(n)
    return data


def test_partial_corr_pcalg_shift_invariant():
    data = _make_test_data()
    r_orig = partial_corr_pcalg(0, 2, {1}, data)
    r_shift = partial_corr_pcalg(0, 2, {1}, data + 10.0)
    assert np.isclose(r_orig, r_shift, atol=1e-8), (
        f"partial_corr (pcalg) not shift-invariant: {r_orig} vs {r_shift}"
    )


def test_partial_corr_helpers_shift_invariant():
    data = _make_test_data()
    r_orig = partial_corr_helpers(data, 0, 2, {1})
    r_shift = partial_corr_helpers(data + 10.0, 0, 2, {1})
    assert np.isclose(r_orig, r_shift, atol=1e-8), (
        f"partial_corr (helpers) not shift-invariant: {r_orig} vs {r_shift}"
    )


def test_partial_corr_pcalg_matches_helpers():
    """Both partial_corr implementations should agree."""
    data = _make_test_data()
    r1 = partial_corr_pcalg(0, 2, {1}, data)
    r2 = partial_corr_helpers(data, 0, 2, {1})
    assert np.isclose(r1, r2, atol=1e-8), (
        f"partial_corr disagreement between pcalg and helpers: {r1} vs {r2}"
    )


if __name__ == "__main__":
    test_partial_corr_pcalg_shift_invariant()
    test_partial_corr_helpers_shift_invariant()
    test_partial_corr_pcalg_matches_helpers()
    print("All tests passed.")
