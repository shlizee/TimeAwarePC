# Install kpcalg (only R package not on conda channels).
# Run AFTER creating the conda env with environment.yml:
#   conda env create -f environment.yml
#   conda activate timeawarepc
#   Rscript install_r_deps.R
#
# graph, RBGL, and pcalg are already installed by the conda env.

# Use a CRAN mirror that does not require interactive selection
options(repos = c(CRAN = "https://cloud.r-project.org"))

# kpcalg is archived on CRAN; install from the archive URL.
kpcalg_url <- "https://cran.r-project.org/src/contrib/Archive/kpcalg/kpcalg_1.0.1.tar.gz"

# Skip if already installed
if (!requireNamespace("kpcalg", quietly = TRUE)) {
  cat("Installing kpcalg from CRAN archive...\n")
  install.packages(kpcalg_url, repos = NULL, type = "source")
} else {
  cat("kpcalg already installed.\n")
}

cat("Done. Try: library(kpcalg)\n")
