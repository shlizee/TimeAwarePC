"""Implements the Time-Aware PC (TPC) Algorithm for finding Causal Functional Connectivity from Time Series.
"""
import time
import numpy as np
import pandas as pd
from timeawarepc.tpc_helpers import *
from timeawarepc.pcalg import estimate_skeleton, estimate_cpdag, ci_test_gauss
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.rlike.container as rlc
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
import random
import networkx as nx
import re
import numpy as np
import warnings
import logging
_logger = logging.getLogger(__name__)
#%%
def _run_pc_inner(sample, alpha, isgauss):
    """Run PC on a single (sub)sample of time-delayed data, return CPDAG as nx.DiGraph."""
    if not isgauss:
        d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
        kpcalg = importr('kpcalg', robject_translations=d)
        sample_pd = pd.DataFrame(sample)
        # Convert pandas -> R DataFrame inside the converter context; do all
        # subsequent R calls outside so results stay as R objects (.rx2 etc.).
        with localconverter(default_converter + pandas2ri.converter):
            df = robjects.conversion.py2rpy(sample_pd)
        base = importr("base")
        out = kpcalg.kpc(**{
            'suffStat': rlc.NamedList((df, "hsic.perm"), names=('data', 'ic.method')),
            'indepTest': kpcalg.kernelCItest,
            'alpha': alpha,
            'labels': robjects.StrVector(sample_pd.columns.astype(str).tolist()),
            'u2pd': "relaxed",
            'skel.method': "stable",
            'verbose': robjects.r('F'),
        })
        dollar = base.__dict__["@"]
        graphobj = dollar(out, "graph")
        graph = importr("graph")
        graphedges = graph.edges(graphobj)
        graphedgespy = {int(key): np.array(re.findall(r'-?\d+\.?\d*', str(graphedges.rx2(str(key))))).astype(int)
                        for key in graphedges.names}
        g = nx.DiGraph(graphedgespy)
    else:
        (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                                         data_matrix=sample,
                                         alpha=alpha, method='stable')
        g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    return g


def cfc_tpc(data, maxdelay=1, alpha=0.1, isgauss=False,
            subsampsize=None, niter=None, thresh=0.25):
    """Estimate Causal Functional Connectivity using TPC Algorithm.

    By default (subsampsize=None, niter=None), TPC runs PC once on the full
    time-delayed data — no bootstrap subsampling. To enable bootstrap
    subsampling for stability scoring, pass both `subsampsize` and `niter`.

    Args:
        data: (numpy.array) of shape (n,p) with n time-recordings for p nodes.
        maxdelay: (int) Maximum time-delay of interactions.
        alpha: (float) Significance level for conditional independence tests.
        isgauss: (boolean)
            True: Assume Gaussian Noise distribution.
            False: Distribution-free.
        subsampsize: (int, optional) Bootstrap window width. If None (default),
            no bootstrap is performed and the full time-delayed data is used.
            Must be specified together with `niter`.
        niter: (int, optional) Number of bootstrap iterations. Must be specified
            together with `subsampsize`. Ignored when subsampsize is None.
        thresh: (float) Bootstrap stability cut-off. Only used when bootstrap
            is active (both subsampsize and niter specified).

    Returns:
        adjacency: (numpy.array) Adjacency matrix of shape (p,p) for estimated CFC by TPC Algorithm.
        weights: (numpy.array) Connectivity Weight matrix of shape (p,p).

    Biswas, Rahul and Shlizerman, Eli (2022). Statistical perspective on functional and causal neural connectomics: the time-aware pc algorithm. arXiv preprint arXiv:2204.04845.
    """
    if (subsampsize is None) != (niter is None):
        raise ValueError(
            "subsampsize and niter must both be specified (for bootstrap) "
            "or both be None (for no bootstrap)."
        )
    use_bootstrap = subsampsize is not None

    start_time = time.time()
    data_trans = data_transformed(data, maxdelay)  # Step 1: Time-Delayed Samples
    _logger.debug("Data transformed in " + str(time.time() - start_time))

    if not use_bootstrap:
        # No-bootstrap path: run PC once on the full time-delayed data.
        g = _run_pc_inner(data_trans, alpha, isgauss)
        causaleff = causaleff_ida(g, data_trans)
        G, causaleffin, _ = return_finaledges(g, causaleff, maxdelay, data.shape[1])
        adjacency = np.asarray(G).copy()
        weights = np.asarray(causaleffin, dtype=float)
        # Step 8: Pruning (keep same magnitude-based pruning as bootstrap path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cutoff = np.nanmax(np.abs(weights)) / 10
        adjacency[np.abs(weights) <= cutoff] = 0
        return adjacency, weights

    # Bootstrap path (legacy / stability-scored).
    C_iter, C_cf_iter, C_cf2_iter = [], [], []
    n = data_trans.shape[0]
    if n - subsampsize <= 0:
        raise ValueError(
            f"subsampsize ({subsampsize}) must be smaller than the "
            f"number of time-delayed samples ({n})."
        )

    for inneriter in range(niter):
        start_btrstrp = time.time()
        _logger.debug("Starting bootstrap " + str(inneriter))

        # Step 2: Select random Bootstrap window
        r_idx = random.sample(range(n - subsampsize), 1)[0]
        sample = data_trans[r_idx:(r_idx + subsampsize), :]

        # Step 3: PC
        g = _run_pc_inner(sample, alpha, isgauss)

        # Step 5: Rolled CFC-DPGM
        causaleff = causaleff_ida(g, data_trans)
        G, causaleffin, causaleffin2 = return_finaledges(g, causaleff, maxdelay, data.shape[1])

        C_iter.append(G)
        C_cf_iter.append(causaleffin)
        C_cf2_iter.append(causaleffin2)
        _logger.debug("Bootstrap done in " + str(time.time() - start_btrstrp))

    # Step 6b-c: Robust Edges and Connectivity Weights
    adjacency = (np.mean(np.asarray(C_iter), axis=0) >= thresh).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        weights = np.nanmean(np.where(np.asarray(C_cf_iter) != 0,
                                       np.asarray(C_cf_iter), np.nan), axis=0)
    _logger.debug("CE shape " + str(weights.shape))

    # Step 8: Pruning
    adjacency[np.abs(weights) <= np.nanmax(np.abs(weights)) / 10] = 0

    return adjacency, weights

def cfc_pc(data,alpha,isgauss=False):
    """Estimate Causal Functional Connectivity using PC Algorithm.

    Args:
        data: (numpy.array) of shape (n,p) with n samples for p nodes 
        alpha: (float) Significance level for conditional independence tests
        isgauss: (boolean) 
            True: Assume Gaussian Noise distribution, 
            False: Distribution free.

    Returns:
        adjacency: (numpy.array) Adcajency matrix of shape (p,p) of estimated CFC by PC Algorithm.
        weights: (numpy.array) Connectivity Weight matrix of shape (p,p).

    """
    if not isgauss:
        d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
        kpcalg = importr('kpcalg', robject_translations = d)
        data_trans_pd = pd.DataFrame(data)
        # Convert pandas -> R inside the converter context only.
        with localconverter(default_converter + pandas2ri.converter):
            df = robjects.conversion.py2rpy(data_trans_pd)
        out=kpcalg.kpc(**{'suffStat' : rlc.NamedList((df,"hsic.perm"),names=('data','ic.method')),
        'indepTest' : kpcalg.kernelCItest,
        'alpha' : alpha,
        'labels' : robjects.StrVector(data_trans_pd.columns.astype(str).tolist()),
        'u2pd' : "relaxed",
        'skel.method' : "stable",
        'verbose' : robjects.r('F')})
        base=importr("base")
        dollar = base.__dict__["@"]
        graphobj=dollar(out, "graph")
        graph=importr("graph")
        graphedges=graph.edges(graphobj)
        import re
        graphedgespy={int(key): np.array(re.findall(r'-?\d+\.?\d*', str(graphedges.rx2(key)))[1:]).astype(int) for key in graphedges.names}
        g=nx.DiGraph(graphedgespy)
    else:
        (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                                            data_matrix=data,
                                            alpha=alpha,method='stable')
        g = estimate_cpdag(skel_graph=g, sep_set=sep_set)

    weights = causaleff_ida(g,data)
    adjacency=nx.adjacency_matrix(g).toarray()
    return adjacency, weights*adjacency
# %%
