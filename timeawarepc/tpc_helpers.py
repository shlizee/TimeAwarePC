"""Helper functions for TPC Algorithm
"""
import numpy as np
import networkx as nx
def data_transformed(data, maxdelay):
    """Implements Step 1 (Time Delay) of TPC, 
    to construct the data with time-delayed samples.
    
    Args:
        data: (numpy.array) of shape (n,p) with n time recordings of p nodes.
        maxdelay: (int) Maximum time-delay of interactions

    Returns:
        data2: (numpy.array) Data with columns as Unrolled DAG nodes (column index v*maxdelay+t = node (v,t)) and rows having time-delayed samples.
    """
    n = data.shape[0]
    p = data.shape[1]
    maxdelay1=maxdelay+1
    new_n = int(np.floor((n-maxdelay)/(2*maxdelay1))*(2*maxdelay1))
    data=data[:new_n,:]
    data2=np.zeros((int(new_n/(2*maxdelay1)),p*maxdelay1))
    for i in range(p):
        for j in range(maxdelay1):
            data2[:,maxdelay1*i+j]=data[j::(2*maxdelay1),i]
    return data2

def orient(g,maxdelay,m):
    """Implements Step 4 (Orient) of TPC, 
    to correct future to past edges in Estimated Unrolled DAG
    
    Args:
        g: (networkx.DiGraph) Unrolled Causal DAG estimate outputted by Step 3 (PC) of TPC.
        maxdelay: (int) Maximum time-delay of interaction
        m: (int) Number of neurons/nodes in original data
    
    Returns:
        g1: (networkx.DiGraph) Re-Oriented Unrolled Causal DAG estimate.
    """
    labels=np.arange(0,(maxdelay+1)*m)

    edge=set([])
    labelmat=labels.reshape((m,maxdelay+1))
    for i in range(m):
        for k in range(m):
            for j in range(maxdelay+1):
                for l in range(maxdelay+1):
                    if j<=l:# and j>=l-maxdelay:
                    #if j==l-1:
                        if (labelmat[i,j],labelmat[k,l]) in g.edges or (labelmat[k,l],labelmat[i,j]) in g.edges:
                            edge = edge | {(labelmat[i,j],labelmat[k,l])}
    g1 = nx.DiGraph()
    g1.add_nodes_from(g.nodes)
    g1.add_edges_from(edge)
    return g1

def causaleff_ida(g,data):
    """Estimate interventional causal effects in the Unrolled DAG in TPC.

    Args:
        g: (networkx.DiGraph) Estimated Unrolled Causal DAG outputted by Step 4 (Orient) of TPC. 
        data: (numpy.array) with variables for Unrolled Causal DAG in columns and time-delayed samples in rows

    Returns:
        causaleff: (numpy.array) Interventional Causal Effects among pairs of nodes in the Unrolled Causal DAG.
    """
    Nodes = list(g.nodes)
    causaleff=np.zeros((len(Nodes),len(Nodes)))

    for x in Nodes:
        for y in Nodes:
            if x!=y:
                pa_x = list(g.predecessors(x))
                pa_y = list(g.predecessors(y))
                if x not in pa_x:
                    regressors = pa_x + [x]
                else:
                    regressors = pa_x
                if y in pa_x:
#                    causaleff[x,y] = 0#uncomment this and comment below for legacy version
                    regressors = [item for item in regressors if item != y]
#                else:#uncomment for legacy version
                X=np.asarray(data[:,regressors])
                Y=np.asarray(data[:,y])
                X0=np.hstack((np.ones((X.shape[0],1)),X))
                lm_out = np.linalg.lstsq(X0,Y,rcond=None)[0]
                causaleff[x,y] = lm_out[regressors.index(x)+1]
    return causaleff

def return_finaledges(g,causaleff,maxdelay,m):
    """Implements Step 5 (Rolled CFC-DPGM) of TPC, to obtain 
    the Rolled CFC-DPGM and its weights from the Unrolled Causal DAG

    Args:
        g: (networkx.DiGraph) Oriented Unrolled Causal DAG estimate outputted by Step 4 (Orient) of TPC.
        causaleff: (numpy.array) Estimated Interventional Causal Effects in Unrolled DAG
        maxdelay: (int) Maximum time-delay of interaction
        m: (int) Number of neurons/nodes in original data

    Returns:
        adjacency: (numpy.array) Adjacency matrix for the Rolled CFC-DPGM estimate.
        wdir: (numpy.array) Weights for direct connections in the Rolled CFC-DPGM.
        windir: (numpy.array) Weights for indirect/direct connections in the Rolled CFC-DPGM.
    """
    labels=np.arange(0,(maxdelay+1)*m)
    labelmat=labels.reshape((m,maxdelay+1))
    causaleff1 = np.zeros((m,m))
    causaleff2 = np.zeros((m,m))
    edge_mat = np.zeros((m,m))==1
    g1=g
    for i in range(m):
        for k in range(m):
            acc=[]
            ansacc=[]
            for j in range(maxdelay+1):
                for l in range(maxdelay+1):
                    if j<=l:
                        if (labelmat[i,j],labelmat[k,l]) in g1.edges:
                            edge_mat[i,k]=True
                            acc.append(causaleff[labelmat[i,j],labelmat[k,l]])
                        if labelmat[i,j] in nx.ancestors(g1,labelmat[k,l]):
                            ansacc.append(causaleff[labelmat[i,j],labelmat[k,l]])
            if len(acc)>0:
                causaleff1[i,k] = np.mean(acc)
            if len(ansacc)>0:
                causaleff2[i,k] = np.mean(ansacc)
    adjacency = edge_mat.astype(int)
    wdir = causaleff1
    windir = causaleff2
    return adjacency, wdir, windir