Installation
============

Recommended: conda environment
------------------------------

The conda env at the repo root handles Python, R, rpy2, and all R dependencies in one command:

.. code-block:: bash

	$ git clone https://github.com/shlizee/TimeAwarePC.git
	$ cd TimeAwarePC
	$ conda env create -f environment.yml
	$ conda activate timeawarepc
	$ Rscript install_r_deps.R   # installs kpcalg from CRAN archive


Manual install (alternative)
----------------------------

If you prefer to install without conda:

Requirements
~~~~~~~~~~~~
- Python >=3.9, <3.11 and R >= 4.0
- Install the R package requirements as follows in R.

.. code-block:: R

	> install.packages("BiocManager")
	> BiocManager::install("graph")
	> BiocManager::install("RBGL")
	> install.packages("pcalg")
	> install.packages(c("energy", "kernlab", "RSpectra"))   # kpcalg CRAN deps
	> install.packages("https://cran.r-project.org/src/contrib/Archive/kpcalg/kpcalg_1.0.1.tar.gz")

Install ``timeawarepc`` using ``pip``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

	$ pip install timeawarepc

To use Granger Causality, also install ``nitime``:

.. code-block:: bash

	$ pip install nitime
