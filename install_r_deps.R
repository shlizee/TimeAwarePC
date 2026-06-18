# Install kpcalg + its CRAN dependencies that are not already on conda.
# Run AFTER creating the conda env with environment.yml:
#   conda env create -f environment.yml
#   conda activate timeawarepc
#   Rscript install_r_deps.R
#
# graph, RBGL, and pcalg are already installed by the conda env.

options(repos = c(CRAN = "https://cloud.r-project.org"))

# kpcalg pulls these from CRAN; install them first so the build does not fail.
kpcalg_cran_deps <- c("energy", "kernlab", "RSpectra")
missing_deps <- setdiff(kpcalg_cran_deps,
                       rownames(installed.packages()))
if (length(missing_deps)) {
  cat("Installing kpcalg CRAN deps:", paste(missing_deps, collapse = ", "), "\n")
  install.packages(missing_deps)
}

# kpcalg is archived on CRAN; install from the archive URL.
kpcalg_url <- "https://cran.r-project.org/src/contrib/Archive/kpcalg/kpcalg_1.0.1.tar.gz"

if (!requireNamespace("kpcalg", quietly = TRUE)) {
  cat("Installing kpcalg from CRAN archive...\n")
  install.packages(kpcalg_url, repos = NULL, type = "source")
} else {
  cat("kpcalg already installed.\n")
}

if (requireNamespace("kpcalg", quietly = TRUE)) {
  cat("Done. kpcalg installed. Try: library(kpcalg)\n")
} else {
  cat("ERROR: kpcalg install failed. See messages above.\n")
  quit(status = 1)
}
