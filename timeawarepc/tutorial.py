"""
Use this tutorial to try out main functionalities of this library and test if it has installed properly. 
In this tutorial, you will be estimating causal functional connectivity (CFC) from simulated datasets.
"""
#%%
import networkx as nx
import numpy as np
from numpy.random import default_rng
rng = default_rng(seed=111)
from timeawarepc.tpc import cfc_tpc, cfc_pc
from timeawarepc.simulate_data import *
from timeawarepc.find_cfc import *
#%%
# Simulate a dataset. 
# Specify a data generating model: model = "lingauss" for Linear Gaussian VAR, "nonlinnongauss" for Non-Linear Non-Gaussian VAR, "ctrnn" for Continuous Time Recurrent Neural Network.
# T is the number of time recordings (default 1000), noise is the noise std. deviation (default 1). 
# Note that number of neurons = 4 and max delay of interaction = 1 indices are fixed in this tutorial.
model = 'lingauss'
T=1000
noise = 1

data, CFCtruth = simulate_data(model, T = 1000, noise = 1)

#%%Estimate CFC by find_cfc(), a convenient wrapper around different methods in this library.
#TODO: Specify a method_name. 'TPC' for Time-Aware PC Algorithm, 'PC' for PC Algorithm, 'GC' for Granger Causality.
method_name = 'TPC'#'GC'#'PC'

alpha = 0.05
maxdelay=1
isgauss = (model == 'lingauss')
adjmat, causaleffmat = find_cfc(data,method_name,alpha=alpha,maxdelay=maxdelay,isgauss=isgauss)

#%%Compare the Ground Truth and Estimated CFC.
print('Ground Truth CFC adjacency:')
print(CFCtruth)
print('Estimated CFC adjacency by '+method_name+':')
print(adjmat)

#%%Alternatively, you may estimate CFC using the python functions for individual methods.
#TODO: Specify a method_name. 'TPC' for Time-Aware PC Algorithm, 'PC' for PC Algorithm, 'GC' for Granger Causality.
method_name = 'TPC'#'GC'#'PC'

alpha = 0.05
isgauss = (model == 'lingauss')
if method_name == 'TPC':
    maxdelay=1
    niter = 50
    thresh = 0.25
    adjmat, causaleffmat = cfc_tpc(data,maxdelay=maxdelay,alpha=alpha,niter=niter,thresh=thresh,isgauss=isgauss)
elif method_name == 'PC':
    adjmat, causaleffmat = cfc_pc(data,alpha,isgauss=isgauss)
elif method_name == 'GC':
    from timeawarepc.gc import cfc_gc
    adjmat, causaleffmat = cfc_gc(data,maxdelay,alpha)
#%%Compare the Ground Truth and Estimated CFC.
print('Ground Truth CFC adjacency:')
print(CFCtruth)
print('Estimated CFC adjacency by '+method_name+':')
print(adjmat)

