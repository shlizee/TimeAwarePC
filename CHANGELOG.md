# Changelog

All notable changes to this project will be documented in this file.

## [v2.0.2]

### Fixed (rpy2 modernization)
- Replaced the deprecated `pandas2ri.activate()` calls in `cfc_tpc` / `cfc_pc` with a `localconverter(default_converter + pandas2ri.converter)` context (scoped narrowly to the pandas→R conversion). `pandas2ri.activate()` raises in rpy2 ≥ 3.5.11.
- Switched `rlc.TaggedList(..., tags=...)` to `rlc.NamedList(..., names=...)` to match the current rpy2 API.
- Wrapped `pd.Index` column labels in `robjects.StrVector` before passing to `kpcalg.kpc`; the default converter no longer auto-converts `pd.Index`.

### Removed
- `environment.yml` no longer caps `rpy2` to `<3.6`; the code is now compatible with current rpy2 versions.

### Changed
- `install_r_deps.R` now also installs CRAN deps `energy`, `kernlab`, `RSpectra` so `kpcalg` builds cleanly from the CRAN archive.

## [v2.0.1]

### Fixed
- `__version__` in `timeawarepc/__init__.py` is now correctly synced to the package version (was stuck at `0.0.1` since release).
- Relaxed `rpy2` pin from `==3.5.11` to `>=3.5.11`, so `pip install timeawarepc` inside a conda env does not force a downgrade of an already-installed `rpy2`.

### Added
- Top-level `environment.yml` for one-command conda install (Python, R, rpy2, all R deps except kpcalg).
- `install_r_deps.R` that installs `kpcalg` from CRAN archive (the only R package not on conda channels).
- README install section now leads with the conda env install path; manual install kept as alternative.

### Changed (docs build)
- ReadTheDocs conda env: bumped Python `3.7` → `3.9`, added `matplotlib` for `conf.py`. Improves docs-build compatibility with modern packaging.
- `docs/source/conf.py`: `release` now reads from `timeawarepc.__version__` instead of being hardcoded to `2022`.

## [v2.0.0]

### Changed
- `cfc_tpc` now defaults to **no bootstrap subsampling**. The full time-delayed data is passed to PC in a single call. Previously, `subsampsize=50` and `niter=25` were the defaults, which truncated each PC call to a 50-row window and aggregated across 25 bootstrap iterations.
- New signature: `cfc_tpc(data, maxdelay=1, alpha=0.1, isgauss=False, subsampsize=None, niter=None, thresh=0.25)`.
  - To use bootstrap stability scoring, pass BOTH `subsampsize` and `niter` together.
  - Passing only one raises `ValueError` for clarity.
  - When bootstrap is active and `subsampsize >= n_time_delayed_samples`, raises `ValueError`.
- `find_cfc` now exposes optional `subsampsize` and `niter` arguments that are passed through to `cfc_tpc`. Defaults match the new no-bootstrap behavior.
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
