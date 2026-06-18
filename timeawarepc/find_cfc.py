"""Convenient wrapper for functions in the library.
"""
__all__ = ["find_cfc"]
from timeawarepc.tpc import *
def find_cfc(data, method_name, alpha=0.05, maxdelay=1, isgauss=False,
             subsampsize=None, niter=None, thresh=0.25):
    """Estimate Causal Functional Connectivity (CFC) between nodes from time series.
        This is a wrapper for functions cfc_tpc, cfc_pc, cfc_gc in tpc.py.
        Refer to the individual functions for their details.

    Args:
        data: (numpy.array) of shape (n,p) with n time-recordings for p nodes
        method_name: (string)
            'TPC': Implements TPC Algorithm,
            'PC': PC Algorithm,
            'GC': Granger Causality.
        alpha: (float) Significance level
        isgauss: (boolean) Arg used for method_name == 'PC' or 'TPC'.
            True: Assume Gaussian Noise distribution,
            False: Distribution free.
        maxdelay: (int) Maximum time-delay of interactions. Arg used for method_name == 'GC' or 'TPC'.
        subsampsize: (int, optional) Bootstrap window width in TPC. If None
            (default), no bootstrap is performed and the full time-delayed
            data is used. Must be specified together with niter.
        niter: (int, optional) Number of bootstrap iterations in TPC. Must be
            specified together with subsampsize.
        thresh: (float) Bootstrap stability cut-off in TPC. Only used when
            bootstrap is active.

    Returns:
        adjacency: (numpy.array) Adcajency matrix of estimated CFC by chosen method.
        weights: (numpy.array) Connectivity Weights in the CFC

    """

    if method_name == 'TPC':
        adjacency, weights = cfc_tpc(
            data, maxdelay=maxdelay, alpha=alpha, isgauss=isgauss,
            subsampsize=subsampsize, niter=niter, thresh=thresh,
        )
    elif method_name == 'PC':
        adjacency, weights = cfc_pc(data,alpha,isgauss=isgauss)
    elif method_name == 'GC':
        from timeawarepc.gc import cfc_gc
        adjacency, weights = cfc_gc(data,maxdelay,alpha)
    return adjacency,weights