# CausalFunctionalConnectivity
Python library for finding the causal functional connectivity from time series data.

<p align="center">
<img src="/imgs/Schematic.png" align="middle" width="500" height="250"/>
</p>

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [References](#references)

## Overview

This library implements the Time Aware PC algorithm to find the causal functional connectivity from time series data, which is based on recent developments in directed probabilistic graphical modeling of causal interactions in neural time series. The library also includes classical approaches such as, Granger Causality and the PC algorithm.

## Installation

You can get the latest version of causal-timeaware by cloning the repository::

```
git clone -b main https://github.com/bkrahul1/causal-timeaware.git
cd timeaware-pc
pip install .
```

## Documentation

[Documentation is available at readthedocs.org](https://timeaware-pc.readthedocs.io/en/latest/)


## Contributing

Your help is absolutely welcome! Please do reach out or create a feature branch!

## Citation

Biswas, R., & Shlizerman, E. (2022). Statistical Perspective on Functional and Causal Neural Connectomics: The Time-Aware PC Algorithm. https://arxiv.org/abs/2204.04845v2

Biswas, R., & Shlizerman, E. (2021). Statistical Perspective on Functional and Causal Neural Connectomics: A Comparative Study. Frontiers in Systems Neuroscience. https://doi.org/10.3389/fnsys.2022.817962

Biswas, R., & Shlizerman, E. (2020). Neuro-PC: Causal Functional Connectivity from Neural Dynamics. arXiv preprint, arXiv:2011.03913.

## References

R Clay Reid. (2012) From functional architecture to functional connectomics. Neuron, 75(2):209–217.

Smith, S. M., Miller, K. L., Salimi-Khorshidi, G., Webster, M., Beckmann, C. F., Nichols, T. E., ... & Woolrich, M. W. (2011). Network modelling methods for FMRI. Neuroimage, 54(2), 875-891.

Judea Pearl. (2009) Causality. Cambridge University press.

Markus Kalisch and Peter Bhlmann. (2007) Estimating high-dimensional directed acyclic graphs with the pc-algorithm. In The Journal of Machine Learning Research, Vol. 8, pp. 613-636.

Peter Spirtes, Clark N Glymour, Richard Scheines, and David Heckerman. (2000) Causation, prediction, and search. MIT press.



