About TimeAwarePC
=================

``TimeAwarePC`` is a Python package that implements the **Time-Aware PC** (TPC) algorithm for finding **Causal Functional Connectivity** (CFC) from time series observations. Essentially, it estimates the causal network between nodes of the time series and the connectivity weights, in a non-parametric manner, using recent developments in directed probabilistic graphical modeling in time series. The package also includes implementations of the PC algorithm and Granger Causality (GC) to estimate the CFC.

Typical use scenario:

- **Neural Connectomics**: The representation of the flow of information between neurons in the brain based on their activity is termed the causal functional connectome (CFC). The CFC is not directly observed and needs to be inferred from neural time series. 

.. image:: Schematic.png
    :align: center
    :width: 500

Benefits of TPC
---------------

TPC algorithm has been shown to outperform other approaches in estimating the CFC from neural time series, in a variety of simulation, benchmark, and real neurobiological datasets. The model outcome of TPC reflects causality of neural interactions such as being non-parametric, exhibits the directed Markov property in a time-series setting, and is predictive of the consequence of counterfactual interventions on the time series. :cite:`biswasshlizerman2022-2`


The package currently supports the following methods:

- :ref:`Time-Aware PC Algorithm <Time-Aware PC Algorithm>`
- :ref:`PC Algorithm <PC Algorithm>`
- :ref:`Granger Causality <Granger Causality>`