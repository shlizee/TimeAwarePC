# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- `partial_corr` (in both `timeawarepc/pcalg.py` and `timeawarepc/pcalg_helpers.py`) now fits an intercept in the conditioning regression, making the partial correlation shift-invariant (invariant to adding a constant to the data). Previously, the regression was forced through the origin, which biased the residuals when the data was not mean-centered.
- Added regression test (`test/test_partial_corr_shift_invariance.py`) verifying that `partial_corr` produces the same result before and after a constant shift of the data.
