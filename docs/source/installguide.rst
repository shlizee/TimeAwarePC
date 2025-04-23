Installation
============

Installation with ``pip`` is recommended. To use Granger Causality additional dependency of ``nitime`` is required. For detailed instructions, see below.

Requirements
------------
- Python ==3.10.* and R ==4.4.2
- Install the R package requirements as follows in R.

.. code-block:: R

	> install.packages("BiocManager")
	> BiocManager::install("graph")
	> BiocManager::install("RBGL")
	> install.packages("pcalg")
	> install.packages("https://cran.r-project.org/src/contrib/Archive/kpcalg/kpcalg_1.0.1.tar.gz")

Install ``timeawarepc`` using ``pip``
-------------------------------------

.. code-block:: bash

	$ pip install timeawarepc
 
To install ``timeawarepc`` with ``nitime`` also:

.. code-block:: bash

	$ pip install nitime
