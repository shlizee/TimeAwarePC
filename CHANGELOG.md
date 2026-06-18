# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed (BREAKING)
- `cfc_tpc` now defaults to **no bootstrap subsampling**. The full time-delayed data is passed to PC in a single call. Previously, `subsampsize=50` and `niter=25` were the defaults, which truncated each PC call to a 50-row window and aggregated across 25 bootstrap iterations.
- New signature: `cfc_tpc(data, maxdelay=1, alpha=0.1, isgauss=False, subsampsize=None, niter=None, thresh=0.25)`.
  - To enable bootstrap subsampling (original behavior), pass BOTH `subsampsize` and `niter` explicitly.
  - Passing only one raises `ValueError` for clarity.
  - When bootstrap is active and `subsampsize >= n_time_delayed_samples`, raises `ValueError`.
- Refactored the inner PC call into a private helper `_run_pc_inner(sample, alpha, isgauss)` shared by both code paths.

### Migration
- Old call: `cfc_tpc(data)` → bootstrap with subsampsize=50, niter=25.
- New equivalent: `cfc_tpc(data, subsampsize=50, niter=25)`.
- For the new no-bootstrap default behavior: `cfc_tpc(data)` is sufficient.

### Fixed
- `partial_corr` (in both `timeawarepc/pcalg.py` and `timeawarepc/pcalg_helpers.py`) now fits an intercept in the conditioning regression, making the partial correlation shift-invariant (invariant to adding a constant to the data). Previously, the regression was forced through the origin, which biased the residuals when the data was not mean-centered.

### Added
- `test/test_optional_bootstrap.py`: regression tests for the new no-bootstrap default, the bootstrap path, and argument-validation errors.
- `test/test_partial_corr_shift_invariance.py`: regression test verifying `partial_corr` produces the same result before and after a constant shift of the data.
