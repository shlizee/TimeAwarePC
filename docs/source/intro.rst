About TimeAwarePC
===========================

``TimeAwarePC`` is a Python package that implements the **Time-Aware PC** (TPC) algorithm for finding **Causal Functional Connectivity** (CFC) from time series data. Essentially, it estimates the causal network between nodes of the time series and the connectivity weights in a non-parametric manner. This is based on recent developments in directed probabilistic graphical modeling in the time series setting. The package also includes implementations of the PC algorithm and Granger Causality to estimate the CFC.

Typical use case:

- **Neural Connectomics**: The representation of the flow of information between neurons in the brain based on their activity is termed the causal functional connectome. The causal functional connectome is not directly observed and needs to be inferred from neural time series. The model outcome of TPC reflects causality of neural interactions such as being non-parametric, exhibits the directed Markov property in a time-series setting, and is predictive of the consequence of counterfactual interventions on the time series. 

.. image:: Schematic.png
   :width: 600


The advantages of using TPC are summarized in the table below [`1 <https://arxiv.org/abs/2204.04845>`, `2 <https://doi.org/10.3389/fnsys.2022.817962>`].

.. image:: tablesummary.png
   :width: 600

The package currently supports the following methods:

- Time-Aware PC Algorithm
    - :ref:`<Biswas & Shlizerman, 2022 https://arxiv.org/abs/2204.04845>` for details on the TPC Algorithm, comparative study with other approaches, and its applications.
- PC Algorithm
    - :ref:`Kalish & Buhlmann, 2007 <https://jmlr.csail.mit.edu/papers/volume8/kalisch07a/kalisch07a.pdf>`. 
- Granger Causality
    - :ref:`Smith et. al, 2011 <https://www.contrib.andrew.cmu.edu/org/fmri-research/Smith-FMRI-2011.pdf>`.