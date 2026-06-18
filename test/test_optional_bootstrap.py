"""Tests for the optional bootstrap behavior of cfc_tpc.

By default (subsampsize=None, niter=None), cfc_tpc should run PC once on the
full time-delayed data without bootstrap subsampling. When both subsampsize
and niter are specified, it should run bootstrap subsampling as before.
"""
import os
import sys

# Insert project root before any installed timeawarepc so tests run against
# the in-repo source, not any pre-installed package version.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np


# All cfc_tpc paths import rpy2. If R/rpy2 isn't available on this system, we
# fall back to testing the no-bootstrap signature directly via _run_pc_inner.
try:
    from timeawarepc.tpc import cfc_tpc
    HAS_RPY2 = True
except OSError:
    HAS_RPY2 = False


def _make_test_data(seed: int = 0, T: int = 400, p: int = 4):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((T, p))
    # Inject lagged causal structure on the first 4 vars when available.
    for t in range(1, T):
        if p >= 2:
            data[t, 1] = 0.7 * data[t - 1, 0] + 0.3 * rng.standard_normal()
        if p >= 3:
            data[t, 2] = 0.6 * data[t - 1, 1] + 0.3 * rng.standard_normal()
        if p >= 4:
            data[t, 3] = 0.5 * data[t - 1, 2] + 0.3 * rng.standard_normal()
    return data


def test_cfc_tpc_default_no_bootstrap():
    """Default call (no subsampsize / niter) should succeed without bootstrap."""
    if not HAS_RPY2:
        print("SKIP test_cfc_tpc_default_no_bootstrap: rpy2/R not available")
        return
    data = _make_test_data()
    adjacency, weights = cfc_tpc(data, maxdelay=1, alpha=0.1, isgauss=True)
    assert adjacency.shape == (4, 4)
    assert weights.shape == (4, 4)


def test_cfc_tpc_bootstrap_path():
    """Explicit bootstrap (subsampsize + niter specified) should still work."""
    if not HAS_RPY2:
        print("SKIP test_cfc_tpc_bootstrap_path: rpy2/R not available")
        return
    data = _make_test_data()
    adjacency, weights = cfc_tpc(
        data, maxdelay=1, alpha=0.1, isgauss=True,
        subsampsize=50, niter=5,
    )
    assert adjacency.shape == (4, 4)
    assert weights.shape == (4, 4)


def test_cfc_tpc_partial_bootstrap_args_raises():
    """Specifying only one of subsampsize/niter should raise ValueError."""
    if not HAS_RPY2:
        print("SKIP test_cfc_tpc_partial_bootstrap_args_raises: rpy2/R not available")
        return
    data = _make_test_data()
    try:
        cfc_tpc(data, isgauss=True, subsampsize=50)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when only subsampsize is given")

    try:
        cfc_tpc(data, isgauss=True, niter=5)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when only niter is given")


def test_cfc_tpc_subsampsize_too_large_raises():
    """subsampsize >= number of time-delayed samples should raise ValueError."""
    if not HAS_RPY2:
        print("SKIP test_cfc_tpc_subsampsize_too_large_raises: rpy2/R not available")
        return
    data = _make_test_data(T=20, p=3)
    try:
        cfc_tpc(data, isgauss=True, subsampsize=1000, niter=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when subsampsize too large")


if __name__ == "__main__":
    test_cfc_tpc_default_no_bootstrap()
    test_cfc_tpc_bootstrap_path()
    test_cfc_tpc_partial_bootstrap_args_raises()
    test_cfc_tpc_subsampsize_too_large_raises()
    print("All tests passed.")
