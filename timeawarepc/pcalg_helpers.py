"""Helper functions for PC algorithm.
"""
import numpy as np
from scipy import stats, linalg
def ci_test_gauss(data,A,B,S,**kwargs):
    """Conduct Conditional Independence Test using Fisher's Z-transform 
    for node A conditionally independent of node B given set of nodes in S.

    Args:
        data: (numpy.array) of shape (n,p) with n samples for p nodes 
        A: (int) node index in data
        B: (int) node index in data
        S: (set) set of node indices in data
    
    Returns:
        pval: (float) p-value of the conditional independence test for A and B given S.
    """
    r = partial_corr(data,A,B,S)
    #print(r)
    if r==1:
        pval = 0    
    else:
        z = 0.5 * np.log((1+r)/(1-r))
        T = np.sqrt(data.shape[0]-len(S)-3)*np.abs(z)
        pval = 2*(1 - stats.norm.cdf(T))
    return pval

def partial_corr(data,A,B,S):
    """Find Partial Correlation of var A and var B given set of vars in S.

    Args:
        data: (numpy.array) of shape (n,p) with n samples for p nodes 
        A: (int) node index in data
        B: (int) node index in data
        S: (set) set of node indices in data

    Returns:
        p_corr: Partial correlation between A and B given S.
    """
    p = data.shape[1]
    idx = np.zeros(p, dtype=bool)

    for i in range(p):
        if i in S:
            idx[i]=True
    C=data
    beta_A = linalg.lstsq(C[:,idx], C[:,A])[0]
    beta_B = linalg.lstsq(C[:,idx], C[:,B])[0]

    res_A = C[:,A] - C[:, idx].dot(beta_A)
    res_B = C[:,B] - C[:, idx].dot(beta_B)
    
    p_corr = stats.pearsonr(res_A, res_B)[0]  
    
    return p_corr
