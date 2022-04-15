"""Create simulated time series dataset from three different paradigms: 
Linear Gaussian Vector-AutoRegressive (VAR) Model,
Non-linear Non-Gaussian VAR Model, and 
Continuous Time Recurrent Neural Networks (CTRNN).
"""
import numpy as np
from numpy.random import default_rng
rng = default_rng(seed=111)

def simulate_data(model, T = 1000, noise = 1):
    """Simulate time seris data from one of three simulation paradigms with 4 neurons.

    Args:
        model: (string)
            'lingauss': Linear Gaussian VAR model
            'nonlinnongauss': Non-linear Non-Gaussian VAR model
            'ctrnn': Continuous Time Recurrent Neural Networks
        T: (int) Number of time recordings in the time series
        noise: (float) Noise standard deviation

    Returns: 
        data: (numpy.array) Time series data of shape (n,p) with n time recordings for p nodes
        CFCtruth: (numpy.array) Ground Truth adjacency matrix of shape (p,p).
    """
    n_neurons = 4
    CFCtruth = np.zeros((4,4),dtype = int)
    if model == "lingauss":
        smspikes=np.zeros((n_neurons,T))
        lag=1
        for iter1 in range(n_neurons):
            smspikes[iter1,0]=rng.normal(scale=noise)
        for t in range(1,T):
            smspikes[0,t]=rng.normal(scale=noise)+1
            smspikes[1,t]=rng.normal(scale=noise)-1
            smspikes[2,t]=2*np.sum(smspikes[0,np.max((t-lag,0)):t])+np.sum(smspikes[1,np.max((t-lag,0)):t])+rng.normal(scale=noise)
            smspikes[3,t]=2*np.sum(smspikes[2,np.max((t-lag,0)):t])+rng.normal(scale=noise)
        CFCtruth[0,2]=CFCtruth[1,2]=CFCtruth[2,3]=1
    elif model == "nonlinnongauss":
        smspikes=np.zeros((n_neurons,T))
        lag=1
        for iter1 in range(n_neurons):
            smspikes[iter1,0]=np.random.uniform()
        for t in range(1,T):
            smspikes[0,t]=np.random.uniform(high = noise)
            smspikes[1,t]=np.random.uniform(high = noise)
            smspikes[2,t]=4*np.sum(np.sin(smspikes[0,np.max((t-lag,0)):t]))+3*np.sum(np.sin(smspikes[1,np.max((t-lag,0)):t]))+np.random.uniform(high = noise)
            smspikes[3,t]=3*np.sum(np.sin(smspikes[2,np.max((t-lag,0)):t]))+np.random.uniform(high = noise)
        CFCtruth[0,2]=CFCtruth[1,2]=CFCtruth[2,3]=1
    elif model == 'ctrnn':
        lag=1
        p=4
        w=np.zeros((p,p))
        w[0,2]=10
        w[1,2]=10
        w[2,3]=10
        n_ctrnn=T
        tau=10
        smspikes=simulate_ctrnn(n_ctrnn,p,w,tau,noise)
        CFCtruth[0,2]=CFCtruth[1,2]=CFCtruth[2,3]=CFCtruth[0,0]=CFCtruth[1,1]=CFCtruth[2,2]=CFCtruth[3,3]=1
    data = smspikes.T
    return data,CFCtruth

def simulate_ctrnn(n_ctrnn, p, w, tau, noise=1):
    """Simulates a CTRNN.
    
    Args:
        n_ctrnn: Number of time points in the CTRNN
        p: Number of nodes.
        w: Weight matrix.
        tau: Time constant of the CTRNN.
        noise: Noise level in the CTRNN.
        
    Returns:
        u: (numpy.array) CTRNN time series of shape (p,n_ctrnn).
    """
    e=np.exp(1)
    u=np.zeros((p,n_ctrnn))
    for n in range(n_ctrnn-1):
        for i in range(p):
            In=np.random.normal(1,noise)
            u[i,(n+1)] = u[i,n] - ((e*u[i,n])/tau) + e*np.sum(w[:,i]*np.tanh(u[:,n]))/tau + e*In/tau#np.tanh(u[:,n])
    return u