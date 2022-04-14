"""Convenient wrapper for functions in the library.
"""
from timeawarepc.tpc import *
def find_cfc(data,method_name,alpha=0.05,maxdelay=1,niter=50,thresh=0.25,isgauss=False):
    """Estimate Causal Functional Connectivity (CFC) between nodes from time series.
        This is a wrapper for functions cfc_tpc, cfc_pc, cfc_gc in tpc.py.
        Refer to the individual functions for their details.

    Args:
        data: (numpy.array) of shape (n,p) with n time-recordings for p nodes
        method_name: (string)
            'TPC': Implements TPC Algorithm
            'PC': PC Algorithm
            'GC': Granger Causality
        alpha: (float) Significance level

    (Required args if method_name is 'PC':)
        isgauss: (boolean) Arg for TPC and PC, not GC.
            True: Assume Gaussian Noise distribution, 
            False: Distribution free.

    (Required args if method_name is 'GC':)
        maxdelay: (int) Maximum time-delay of interactions.
    
    (Required args if method_name is 'TPC':)
        maxdelay: (int) Maximum time-delay of interactions. Arg for TPC and GC, not PC.
        isgauss: (boolean) Arg for TPC and PC, not GC.
            True: Assume Gaussian Noise distribution, 
            False: Distribution free.
        subsampsize: (int) Bootstrap window width in TPC.
        niter: (int) Number of bootstrap iterations in TPC
        thresh: (float) Bootstrap stability cut-off in TPC

    Returns:
        adjacency: (numpy.array) Adcajency matrix of estimated CFC by chosen method.
        weights: (numpy.array) Connectivity Weights in the CFC

    """

    if method_name == 'TPC':
        adjacency, weights = cfc_tpc(data,maxdelay=maxdelay,alpha=alpha,niter=niter,thresh=thresh,isgauss=isgauss)
    elif method_name == 'PC':
        adjacency, weights = cfc_pc(data,alpha,isgauss=isgauss)
    elif method_name == 'GC':
        adjacency, weights = cfc_gc(data,maxdelay,alpha)
    return adjacency,weights