import numpy as np
import nitime.analysis as nta
import nitime.timeseries as ts
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