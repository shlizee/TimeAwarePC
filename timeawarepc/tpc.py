"""Implements the Time-Aware PC (TPC) Algorithm for finding 
Causal Functional Connectivity from Time Series.

Biswas, R., & Shlizerman, E. (2022). Statistical Perspective on 
Functional and Causal Neural Connectomics: The Time-Aware PC Algorithm. 
https://arxiv.org/abs/2204.04845
"""
import time
import numpy as np
import pandas as pd
from timeawarepc.tpc_helpers import *
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.rlike.container as rlc
from rpy2.robjects import pandas2ri
import random
import networkx as nx
import re
import nitime.analysis as nta
import nitime.timeseries as ts
import numpy as np
from timeawarepc.pcalg import *
import warnings
#%%
def cfc_tpc(data,maxdelay=1,subsampsize=50,niter=25,alpha=0.1,thresh=0.25,isgauss=False):
    """Estimate Causal Functional Connectivity using TPC Algorithm.

    Args:
        data: (numpy.array) of shape (n,p) with n time-recordings for p nodes .
        maxdelay: (int) Maximum time-delay of interactions.
        subsampsize: (int) Bootstrap window width.
        niter: (int) Number of bootstrap iterations.
        alpha: (float) Significance level for conditional independence tests.
        thresh: (float) Bootstrap stability cut-off.
        isgauss: (boolean) 
            True: Assume Gaussian Noise distribution. 
            False: Distribution-free.

    Returns:
        adjacency: (numpy.array) Adcajency matrix of shape (p,p) 
                    for estimated CFC by TPC Algorithm.
        weights: (numpy.array) Connectivity Weight matrix of shape (p,p).
    """
    C_iter=[]
    C_cf_iter=[]
    C_cf2_iter=[]
    start_time = time.time()
    data_trans = data_transformed(data, maxdelay)#Step 1: Time-Delayed Samples
    print("Data transformed in "+str(time.time()-start_time))

    #Steps 2-6a:
    for inneriter in range(niter):
        start_btrstrp = time.time()
        print("Starting bootstrap "+str(inneriter))
        n=data_trans.shape[0]

        #Step 2: Select random Bootstrap window
        r_idx = random.sample(range(n-subsampsize),1)[0]
        
        #Step 3: PC
        if not isgauss:
            d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
            kpcalg = importr('kpcalg', robject_translations = d)
            data_trans_pd=pd.DataFrame(data_trans[r_idx:(r_idx+subsampsize),:])
            pandas2ri.activate()
            df = robjects.conversion.py2rpy(data_trans_pd)
            base=importr("base")
            out=kpcalg.kpc(**{'suffStat' : rlc.TaggedList((df,"hsic.perm"),tags=('data','ic.method')),
            'indepTest' : kpcalg.kernelCItest,
            'alpha' : alpha,
            'labels' : data_trans_pd.columns.astype(str),
            'u2pd' : "relaxed",
            'skel.method' : "stable",
            'verbose' : robjects.r('F')})
            dollar = base.__dict__["@"]
            graphobj=dollar(out, "graph")
            graph=importr("graph")
            graphedges=graph.edges(graphobj)#, "matrix")
            graphedgespy={int(key): np.array(re.findall(r'-?\d+\.?\d*', str(graphedges.rx2(key)))[1:]).astype(int) for key in graphedges.names}
            g=nx.DiGraph(graphedgespy)
        else:
            data_trans_pd=data_trans[r_idx:(r_idx+subsampsize),:]
            (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                                                data_matrix=data_trans_pd,
                                                alpha=alpha,method='stable')
            g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
        
        #Step 4: Orient
        g=orient(g,maxdelay,data.shape[1])

        #Step 5: Rolled CFC-DPGM
        causaleff = causaleff_ida(g,data_trans)#Interventional Causal Effects in Unrolled DAG
        G,causaleffin, causaleffin2=return_finaledges(g,causaleff,maxdelay,data.shape[1])#Rolled CFC-DPGM
        
        A_rr=G
        C_iter.append(A_rr)
        C_cf_iter.append(causaleffin)
        C_cf2_iter.append(causaleffin2)
        print("Bootstrap done in "+str(time.time()-start_btrstrp))
        #Step 6a: Repeat Steps 2-5    

    #Step 6b-c: Robust Edges and Connectivity Weights
    adjacency=(np.mean(np.asarray(C_iter),axis=0)>=thresh).astype(int)#Robust Edges
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        weights=np.nanmean(np.where(np.asarray(C_cf_iter)!=0,np.asarray(C_cf_iter),np.nan),axis=0)#Robust Connectivity Weights
    print("CE shape "+str(weights.shape))

    #Step 8: Pruning
    adjacency[np.abs(weights) <= np.nanmax(np.abs(weights))/10]=0

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
        pandas2ri.activate()
        df = robjects.conversion.py2rpy(data_trans_pd)
        out=kpcalg.kpc(**{'suffStat' : rlc.TaggedList((df,"hsic.perm"),tags=('data','ic.method')),#robjects.r('list(data=data_trans, ic.method="hsic.perm")'),#list(data=data_trans, ic.method="hsic.perm"),
        'indepTest' : kpcalg.kernelCItest,
        'alpha' : alpha,
        'labels' : data_trans_pd.columns.astype(str),
        'u2pd' : "relaxed",
        'skel.method' : "stable",
        'verbose' : robjects.r('F')})
        base=importr("base")
        dollar = base.__dict__["@"]
        graphobj=dollar(out, "graph")
        graph=importr("graph")
        graphedges=graph.edges(graphobj)#, "matrix")
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
    return adjacency, weights

def cfc_gc(data,maxdelay,alpha):
    """Estimate Causal Functional Connectivity using Granger Causality.

    Args:
        data: (numpy.array) of shape (n,p) with n samples for p nodes 
        maxdelay: Maximum time-delay of interactions. 
        alpha: (float) Significance level for conditional independence tests

    Returns:
        adjacency: (numpy.array) Adcajency matrix of shape (p,p) of estimated CFC by Granger Causality.
        weights: (numpy.array) Connectivity Weight matrix of shape (p,p).

    """
    TR = 1
    thresh = 0
    time_series = ts.TimeSeries(data.T, sampling_interval=TR)
    order=maxdelay
    G = nta.GrangerAnalyzer(time_series, order=order)
    adj_mat = np.zeros((data.shape[1],data.shape[1]))

    adj_mat=np.mean(np.nan_to_num(G.causality_xy[:, :]),-1)+np.mean(np.nan_to_num(G.causality_yx[:, :]),-1).T
    adjmat1=np.mean(np.nan_to_num(G.causality_xy[:, :]),-1)
    adjmat2=np.mean(np.nan_to_num(G.causality_yx[:, :]),-1)
    adj_mat=adjmat1+adjmat2.T
    weights = adj_mat
    thresh = np.percentile(adj_mat,(1-alpha)*100)
    adjacency=(adj_mat > thresh).astype(int)
    return adjacency, weights
# %%
