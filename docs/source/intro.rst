About TimeAwarePC
===========================

``TimeAwarePC`` is a Python package that implements the **Time-Aware PC** (TPC) algorithm for finding **Causal Functional Connectivity** (CFC) from time series observations. Essentially, it estimates the causal network between nodes of the time series and its connectivity weights, in a non-parametric manner, using recent developments in directed probabilistic graphical modeling for time series. The package also includes implementations of the PC algorithm and Granger Causality (GC) to estimate the CFC.

Typical use case:

- **Neural Connectomics**: The representation of the flow of information between neurons in the brain based on their activity is termed the causal functional connectome. The causal functional connectome is not directly observed and needs to be inferred from neural time series. The model outcome of TPC reflects causality of neural interactions such as being non-parametric, exhibits the directed Markov property in a time-series setting, and is predictive of the consequence of counterfactual interventions on the time series.

.. image:: Schematic.png
   :width: 500

The package currently supports the following methods:

- :ref:`Time-Aware PC Algorithm <Time-Aware PC Algorithm>`
- :ref:`PC Algorithm <PC Algorithm>`
- :ref:`Granger Causality <Granger Causality>`