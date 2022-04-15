Installation
============

Installation with ``pip`` is recommended. To use Granger Causality additional dependency of ``nitime`` is required. For detailed instructions, see below.

Requirements
------------
- Python >=3.6 and R >=4.0
- Install the R package requirements as follows in R.

.. code-block:: R

	> install.packages("BiocManager")
	> BiocManager::install("graph")
	> BiocManager::install("RBGL")
	> install.packages("pcalg")
	> install.packages("kpcalg")

Install ``timeawarepc`` using ``pip``
-------------------------------------

.. code-block:: bash

	$ pip install timeawarepc
 
To install ``timeawarepc`` with ``nitime`` also:

.. code-block:: bash

	$ pip install nitime