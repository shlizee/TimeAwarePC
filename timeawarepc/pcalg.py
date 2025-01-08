#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A graph generator based on the PC algorithm [Kalisch2007].

[Kalisch2007] Markus Kalisch and Peter Bhlmann. Estimating
high-dimensional directed acyclic graphs with the pc-algorithm. In The
Journal of Machine Learning Research, Vol. 8, pp. 613-636, 2007.

License: BSD
"""

from __future__ import print_function

from itertools import combinations, permutations
import logging
import numpy as np
import networkx as nx

_logger = logging.getLogger(__name__)

def _create_complete_graph(node_ids):
    """Create a complete graph from the list of node ids.

    Args:
        node_ids: a list of node ids

    Returns:
        An undirected graph (as a networkx.Graph)
    """
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        g.add_edge(i, j)
    return g

def estimate_skeleton(indep_test_func, data_matrix, alpha, **kwargs):
    """Estimate a skeleton graph from the statistis information.

    Args:
        indep_test_func: the function name for a conditional
            independency test.
        data_matrix: data (as a numpy array).
        alpha: the significance level.
        kwargs:
            'max_reach': maximum value of l (see the code).  The
                value depends on the underlying distribution.
            'method': if 'stable' given, use stable-PC algorithm
                (see [Colombo2014]).
            'init_graph': initial structure of skeleton graph
                (as a networkx.Graph). If not specified,
                a complete graph is used.
            other parameters may be passed depending on the
                indep_test_func()s.
    Returns:
        g: a skeleton graph (as a networkx.Graph).
        sep_set: a separation set (as an 2D-array of set()).

    [Colombo2014] Diego Colombo and Marloes H Maathuis. Order-independent
    constraint-based causal structure learning. In The Journal of Machine
    Learning Research, Vol. 15, pp. 3741-3782, 2014.
    """

    def method_stable(kwargs):
        return ('method' in kwargs) and kwargs['method'] == "stable"

    node_ids = range(data_matrix.shape[1])
    node_size = data_matrix.shape[1]
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]
    if 'init_graph' in kwargs:
        g = kwargs['init_graph']
        if not isinstance(g, nx.Graph):
            raise ValueError
        elif not g.number_of_nodes() == len(node_ids):
            raise ValueError('init_graph not matching data_matrix shape')
        for (i, j) in combinations(node_ids, 2):
            if (not g.has_edge(i, j)):
                sep_set[i][j] = None
                sep_set[j][i] = None
    else:
        g = _create_complete_graph(node_ids)

    l = 0
    while True:
        cont = False
        remove_edges = []
        for (i, j) in permutations(node_ids, 2):
            adj_i = list(g.neighbors(i))
            if j not in adj_i:
                continue
            else:
                adj_i.remove(j)
            #if g.has_edge(i,j):
            if len(adj_i) >= l:
                _logger.debug('testing %s and %s' % (i,j))
                _logger.debug('neighbors of %s are %s' % (i, str(adj_i)))
                if len(adj_i) < l:
                    continue
                for k in combinations(adj_i, l):
                    _logger.debug('indep prob of %s and %s with subset %s'
                                % (i, j, str(k)))
                    p_val = indep_test_func(data_matrix, i, j, set(k),
                                            **kwargs)
                    _logger.debug('p_val is %s' % str(p_val))
                    if p_val > alpha:
                        if g.has_edge(i, j):
                            _logger.debug('p: remove edge (%s, %s)' % (i, j))
                            if method_stable(kwargs):
                                remove_edges.append((i, j))
                            else:
                                g.remove_edge(i, j)
                        sep_set[i][j] |= set(k)
                        sep_set[j][i] |= set(k)
                        break
                cont = True
        l += 1
        if method_stable(kwargs):
            g.remove_edges_from(remove_edges)
        if cont is False:
            break
        if ('max_reach' in kwargs) and (l > kwargs['max_reach']):
            break

    return (g, sep_set)

def estimate_cpdag(skel_graph, sep_set):
    """Estimate a CPDAG from the skeleton graph and separation sets
    returned by the estimate_skeleton() function.

    Args:
        skel_graph: A skeleton graph (an undirected networkx.Graph).
        sep_set: An 2D-array of separation set.
            The contents look like something like below.
                sep_set[i][j] = set([k, l, m])

    Returns:
        An estimated DAG.
    """
    dag = skel_graph.to_directed()
    node_ids = skel_graph.nodes()
    for (i, j) in combinations(node_ids, 2):
        adj_i = set(dag.successors(i))
        if j in adj_i:
            continue
        adj_j = set(dag.successors(j))
        if i in adj_j:
            continue
        if sep_set[i][j] is None:
            continue
        common_k = adj_i & adj_j
        for k in common_k:
            if k not in sep_set[i][j]:
                if dag.has_edge(k, i):
                    _logger.debug('S: remove edge (%s, %s)' % (k, i))
                    dag.remove_edge(k, i)
                if dag.has_edge(k, j):
                    _logger.debug('S: remove edge (%s, %s)' % (k, j))
                    dag.remove_edge(k, j)

    def _has_both_edges(dag, i, j):
        return dag.has_edge(i, j) and dag.has_edge(j, i)

    def _has_any_edge(dag, i, j):
        return dag.has_edge(i, j) or dag.has_edge(j, i)

    def _has_one_edge(dag, i, j):
        return ((dag.has_edge(i, j) and (not dag.has_edge(j, i))) or
                (not dag.has_edge(i, j)) and dag.has_edge(j, i))

    def _has_no_edge(dag, i, j):
        return (not dag.has_edge(i, j)) and (not dag.has_edge(j, i))

    # For all the combination of nodes i and j, apply the following
    # rules.
    old_dag = dag.copy()
    while True:
        for (i, j) in combinations(node_ids, 2):
            # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
            # such that k and j are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Look all the predecessors of i.
                for k in dag.predecessors(i):
                    # Skip if there is an arrow i->k.
                    if dag.has_edge(i, k):
                        continue
                    # Skip if k and j are adjacent.
                    if _has_any_edge(dag, k, j):
                        continue
                    # Make i-j into i->j
                    _logger.debug('R1: remove edge (%s, %s)' % (j, i))
                    dag.remove_edge(j, i)
                    break

            # Rule 2: Orient i-j into i->j whenever there is a chain
            # i->k->j.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where k is i->k.
                succs_i = set()
                for k in dag.successors(i):
                    if not dag.has_edge(k, i):
                        succs_i.add(k)
                # Find nodes j where j is k->j.
                preds_j = set()
                for k in dag.predecessors(j):
                    if not dag.has_edge(j, k):
                        preds_j.add(k)
                # Check if there is any node k where i->k->j.
                if len(succs_i & preds_j) > 0:
                    # Make i-j into i->j
                    _logger.debug('R2: remove edge (%s, %s)' % (j, i))
                    dag.remove_edge(j, i)

            # Rule 3: Orient i-j into i->j whenever there are two chains
            # i-k->j and i-l->j such that k and l are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where i-k.
                adj_i = set()
                for k in dag.successors(i):
                    if dag.has_edge(k, i):
                        adj_i.add(k)
                # For all the pairs of nodes in adj_i,
                for (k, l) in combinations(adj_i, 2):
                    # Skip if k and l are adjacent.
                    if _has_any_edge(dag, k, l):
                        continue
                    # Skip if not k->j.
                    if dag.has_edge(j, k) or (not dag.has_edge(k, j)):
                        continue
                    # Skip if not l->j.
                    if dag.has_edge(j, l) or (not dag.has_edge(l, j)):
                        continue
                    # Make i-j into i->j.
                    _logger.debug('R3: remove edge (%s, %s)' % (j, i))
                    dag.remove_edge(j, i)
                    break

            # Rule 4: Orient i-j into i->j whenever there are two chains
            # i-k->l and k->l->j such that k and j are nonadjacent.
            #
            # However, this rule is not necessary when the PC-algorithm
            # is used to estimate a DAG.

        if nx.is_isomorphic(dag, old_dag):
            break
        old_dag = dag.copy()

    return dag
# def pre_whiten(data,S=None):
#     import numpy as np
#     from scipy import stats, linalg
#     import GPy
#     import GPyOpt
#     import seaborn as sns
#     S=np.tile(np.arange(data.shape[0]),reps=[data.shape[1],1]).T
#     sigma_f, l = 1.5, 2
#     kernel = GPy.kern.RBF(1, sigma_f, l)
#     p = data.shape[1]
#     #model = GPy.models.GPRegression(X,y,kernel) 
#     #r = ndw_corr(A,B,S,data)
#     #kernel = RBF(0.1, (10,10))
#     #gp = gpr(kernel=kernel, n_restarts_optimizer=100, alpha = 0.04)
#     #gp.fit(X,y)
#     r=np.zeros(data.shape)
#     for A in range(p):
#         print((A,p))
#         model_A = GPy.models.GPRegression(S[:,A].reshape((-1,1)),data[:,A].reshape((-1,1)),kernel)#
#         model_A.optimize()#_restarts(num_restarts=20)
#         r[:,A] = data[:,A] - model_A.predict(S[:,A].reshape((-1,1)))[0].reshape(data[:,A].shape)#model_A.predict(S[:,A].reshape(-1,1))
#     return r

# def ci_test_gp(data,A,B,C,**kwargs):
#     #use pre-whitened data in data.
#     import numpy as np
#     from scipy import stats, linalg
#     import GPy
#     import GPyOpt
#     import seaborn as sns
#     from HSIC import hsic_gam
#     S=np.tile(np.arange(data.shape[0]),reps=[data.shape[1],1]).T
#     sigma_f, l = 1.5, 2
#     kernel = GPy.kern.RBF(1, sigma_f, l)
#     p = data.shape[1]
#     #model = GPy.models.GPRegression(X,y,kernel) 
#     C2 = np.zeros(p, dtype=np.bool)
#     for i in range(p):
#         if i in C:
#             C2[i]=True
#     #r = ndw_corr(A,B,S,data)
    
#     # if whiten == True:
#     #     r = pre_whiten(data)
#     # else:
#     #     r=data
#     #print(r)
#     # if r==1:
#     #     pval = 0    
#     # else:
#     #     z = 0.5 * np.log((1+r)/(1-r))
#     #     T = np.sqrt(data.shape[0]-len(S)-3)*np.abs(z)
#     #     pval = 2*(1 - stats.norm.cdf(T))
#     if len(C) != 0:
#         if len(C)>1:
#             model_A = GPy.models.GPRegression(data[:,C2],data[:,A].reshape((-1,1)),kernel)#
#             model_B = GPy.models.GPRegression(data[:,C2],data[:,B].reshape((-1,1)),kernel)
#         if len(C)==1:
#             model_A = GPy.models.GPRegression(data[:,C2].reshape((-1,1)),data[:,A].reshape((-1,1)),kernel)
#             model_B = GPy.models.GPRegression(data[:,C2].reshape((-1,1)),data[:,B].reshape((-1,1)),kernel)
#         model_A.optimize()#_restarts(num_restarts=20,verbose=False);
#         model_B.optimize()
#         rA = data[:,A] - model_A.predict(data[:,C2])[0].reshape(data[:,A].shape)#model_A.predict(S[:,A].reshape(-1,1))
#         rB = data[:,B] - model_B.predict(data[:,C2])[0].reshape(data[:,B].shape)#model_A.predict(S[:,A].reshape(-1,1))
#     else:
#         rA = data[:,A]
#         rB = data[:,B]
#     pval = hsic_gam(rA.reshape((-1,1)),rB.reshape((-1,1)))
#     return pval
    
def ci_test_gauss(data,A,B,S,**kwargs):
    import numpy as np
    from scipy import stats, linalg
    r = partial_corr(A,B,S,data)
    #print(r)
    if r==1:
        pval = 0    
    else:
        z = 0.5 * np.log((1+r)/(1-r))
        T = np.sqrt(data.shape[0]-len(S)-3)*np.abs(z)
        pval = 2*(1 - stats.norm.cdf(T))
    return pval
def ci_test_gauss_btp(data,A,B,S,**kwargs):
    import numpy as np
    from scipy import stats, linalg
    from arch import bootstrap
    #import stationarybootstrap as SBB
    from numpy.random import RandomState
    r = partial_corr(A,B,S,data)
    #print(r)
    if r==1:
        pval = 0
    else:
        z = 0.5 * np.log((1+r)/(1-r))
        T = np.abs(z)
        band = bootstrap.optimal_block_length(data)
        n = data.shape[0]
        p = data.shape[1]
        nbtp = 50
        Tbtp = np.zeros(nbtp)
        idx=0
        bs = bootstrap.StationaryBootstrap(np.median(band.iloc[:,0]),data)
        #bs = bootstrap.StationaryBootstrap(50,data)
        #bs = bootstrap.CircularBlockBootstrap(50,data)
        #for data1 in bs.bootstrap(nbtp):
        #ystar, yindices, yindicedict = SBB.resample(data, 0.04)
        #for idx1 in range(50):
        #    data1, yindices, yindicedict = SBB.resample(data, 0.04)# = ystar[idx1,:,:]
        for data1 in bs.bootstrap(nbtp):
            rbtp = partial_corr(A,B,S,data1[0][0])
            zbtp = 0.5 * np.log((1+rbtp)/(1-rbtp))
            Tbtp[idx] = np.abs(zbtp)
            idx=idx+1
        # blower = np.quantile(Tbtp,alpha/2)-T
        # bupper = np.quantile(Tbtp,1-alpha/2)-T
        # T-bupper
        # T+blower
        # 2*T-np.quantile(Tbtp,1-alpha/2)
        pval = np.sum(Tbtp>2*T)/nbtp
        #print(pval)
    return pval
def hsic_condind(data,A,B,S,**kwargs):
    import pandas as pd
    from hsiccondTestIC import hsic_CI
    #from pcalg import estimate_skeleton, estimate_cpdag, causaleff_ida, ci_test_gauss
    if len(S) == 0:
        X=data[:,A]
        Y=data[:,B]
        pval=hsic_CI(X,Y)
        # kpc(**{'suffStat' : rlc.TaggedList((df,"hsic.perm"),tags=('data','ic.method')),
        # 'indepTest' : kpcalg.kernelCItest,
        # 'alpha' : alpha,
        # 'labels' : data_trans_pd.columns.astype(str),
        # 'u2pd' : "relaxed",
        # 'skel.method' : "stable",
        # #'fixedGaps' : fixedgaps_r,
        # 'verbose' : robjects.r('F')})
    else:
        p = data.shape[1]
        idx = np.zeros(p, dtype=bool)
        for i in range(p):
            if i in S:
                idx[i]=True
        X=data[:,A]
        Y=data[:,B]
        Z=data[:,idx]
        pval = hsic_CI(X,Y,Z)
    return pval
    # if len(S) == 0:
    #     sig,pval,T=hsiccondTestIC(data[:,A],data[:,B])
    # else:
    #     p = data.shape[1]
    #     idx = np.zeros(p, dtype=np.bool)
    #     for i in range(p):
    #         if i in S:
    #             idx[i]=True
    #     sig,pval,T=hsiccondTestIC(data[:,A],data[:,B],data[:,idx])

    
def partial_corr(A,B,S,data):
    import numpy as np
    from scipy import stats, linalg
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
if __name__ == '__main__':
    import networkx as nx
    import numpy as np

    from gsq.ci_tests import ci_test_bin, ci_test_dis
    from gsq.gsq_testdata import bin_data, dis_data

    # ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    # _logger.setLevel(logging.DEBUG)
    # _logger.addHandler(ch)

    dm = np.array(bin_data).reshape((5000, 5))
    (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_bin,
                                     data_matrix=dm,
                                     alpha=0.01)
    g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    g_answer = nx.DiGraph()
    g_answer.add_nodes_from([0, 1, 2, 3, 4])
    g_answer.add_edges_from([(0, 1), (2, 3), (3, 2), (3, 1),
                             (2, 4), (4, 2), (4, 1)])
    print('Edges are:', g.edges(), end='')
    if nx.is_isomorphic(g, g_answer):
        print(' => GOOD')
    else:
        print(' => WRONG')
        print('True edges should be:', g_answer.edges())

    dm = np.array(dis_data).reshape((10000, 5))
    (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_dis,
                                     data_matrix=dm,
                                     alpha=0.01,
                                     levels=[3,2,3,4,2])
    g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    g_answer = nx.DiGraph()
    g_answer.add_nodes_from([0, 1, 2, 3, 4])
    g_answer.add_edges_from([(0, 2), (1, 2), (1, 3), (4, 3)])
    print('Edges are:', g.edges(), end='')
    if nx.is_isomorphic(g, g_answer):
        print(' => GOOD')
    else:
        print(' => WRONG')
        print('True edges should be:', g_answer.edges())

    dm1 = np.random.normal(0,1,1000)
    dm2 = np.random.normal(0,1,1000)
    dm3 = dm1 + 0.5*dm2 + np.random.normal(0,1,1000)
    dm4 = dm3 + np.random.normal(0,1,1000)

    data=np.column_stack((dm1,dm2,dm3,dm4))
    data -= data.mean(axis=0)
    data /= data.std(axis=0)
    # (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
    #                                  data_matrix=data,
    #                                  alpha=0.01,
    #                                  method='stable')
    (g, sep_set) = estimate_skeleton(indep_test_func=hsic_condind,
                                     data_matrix=data,
                                     alpha=0.01,
                                     method='stable')
    g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    g_answer = nx.DiGraph()
    g_answer.add_nodes_from([0, 1, 2, 3])
    g_answer.add_edges_from([(0, 2), (1, 2), (2, 3)])
    print('Edges are:', g.edges(), end='')
    if nx.is_isomorphic(g, g_answer):
        print(' => GOOD')
    else:
        print(' => WRONG')
        print('True edges should be:', g_answer.edges())
#%%
# def cmiknn_indeptest(data,A,B,S,**kwargs):
#     from tigramite.independence_tests import CMIknn
#     cmi_knn = CMIknn(significance='shuffle_test', knn=0.1, shuffle_neighbors=5, transform='ranks')
#     if len(S)==0:
#         data_x = data[:,A]
#         data_y = data[:,B]
#         arr = np.row_stack((data_x,data_y))
#         xyz = np.array([0,1])
#         val = cmi_knn.get_dependence_measure(arr,xyz)
#         p = cmi_knn.get_shuffle_significance(arr,xyz,val)
#     else:
#         data_x = data[:,A]
#         data_y = data[:,B]
#         print(S)
#         data_z = data[:,list(S)].T
#         arr = np.row_stack((data_x,data_y,data_z))
#         xyz = np.array([0,1]+[2]*data_z.shape[0])
#         val = cmi_knn.get_dependence_measure(arr,xyz)
#         p = cmi_knn.get_shuffle_significance(arr,xyz,val)
#     return p
# #%%
# dm1 = np.random.normal(0,1,1000)
# dm2 = np.random.normal(0,1,1000)
# dm3 = dm1 + 0.5*dm2 + np.random.normal(0,1,1000)
# dm4 = dm3 + np.random.normal(0,1,1000)

# dm=np.column_stack((dm1,dm2,dm3,dm4))
# (g, sep_set) = estimate_skeleton(indep_test_func=cmiknn_indeptest,
#                                     data_matrix=dm,
#                                     alpha=0.01,
#                                     method='stable')
# g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
# g_answer = nx.DiGraph()
# g_answer.add_nodes_from([0, 1, 2, 3])
# g_answer.add_edges_from([(0, 2), (1, 2), (2, 3)])
# print('Edges are:', g.edges(), end='')
# if nx.is_isomorphic(g, g_answer):
#     print(' => GOOD')
# else:
#     print(' => WRONG')
#     print('True edges should be:', g_answer.edges())
#%%
def causaleff_parcorr(g,data):
    import numpy as np
    Edges = list(g.edges)
    Nodes = list(g.nodes)
    causaleff=np.zeros((len(Nodes),len(Nodes)))
    for x in Nodes:
        for y in Nodes:
            if x !=y:
                S= [elem for elem in Nodes if elem not in [x,y]]
                causaleff[x,y] = partial_corr(x,y,S,data)
    return causaleff
def causaleff_ida_single(g,data,x,y):
    from sklearn import linear_model
    import numpy as np
    Nodes = list(g.nodes)
    if x in Nodes and y in Nodes:
        if x!=y and x in list(nx.ancestors(g,y)):
            lm = linear_model.LinearRegression()
            pa_x = list(g.predecessors(x))
            pa_y = list(g.predecessors(y))
            if x not in pa_x:
                regressors = pa_x + [x]
            else:
                regressors = pa_x
            if y in pa_x:
                causaleff = 0
            else:
                #if len(regressors)>1:
                X=data[:,regressors]
                #else:
                #    X=data[:,regressors].reshape(-1,1)
                Y=data[:,y]
                lm_out = lm.fit(X,Y)
                causaleff = lm_out.coef_[regressors.index(x)]#lm_out.coef_[0]#lm_out.coef_[regressors.index(x)]#
    return causaleff
# def causaleff_ida(g,data):
#     from sklearn import linear_model
#     import numpy as np
#     #from gen_data_fns import sigmoid, relu
#     #Edges = list(g.edges)
#     Nodes = list(g.nodes)
#     causaleff=np.zeros((len(Nodes),len(Nodes)))
#     # if transformed == True and lag is None:
#     #     print("Please provide lag used in transformation")
#     # elif transformed == True and lag is not None:
#     #     h=np.repeat(np.arange(0,len(Nodes)),lag)
#     # if activation == 'centred-sigmoid':
#     #     activationfn = lambda x: sigmoid(x) - 0.5
#     # elif activation == 'tanh':
#     #     activationfn = lambda x: np.tanh(x)
#     # elif activation == 'linear':
#     #     activationfn = lambda x: x
#     # elif activation == 'relu':
#     #     activationfn = lambda x: relu(x)
#     for x in Nodes:
#         for y in Nodes:
#             if x!=y and x in list(nx.ancestors(g,y)):#list(g.predecessors(y)):#list(nx.ancestors(g,y)):
#             #if (x,y) in Edges:
#             #if y not in list(g.predecessors(x)):
#             #if x in list(nx.ancestors(g,y)):
#                 lm = linear_model.LinearRegression()
#                 pa_x = list(g.predecessors(x))
#                 pa_y = list(g.predecessors(y))
#                 if x not in pa_x:
#                     regressors = pa_x + [x]
#                 else:
#                     regressors = pa_x
#                 if y in pa_x:
#                     causaleff[x,y] = 0
#                 else:
#                 # if y in pa_y:
#                 #     pa_y = pa_y.pop(y)
#                 # if x not in pa_y:
#                 #     regressors = pa_y + [x]
#                 # else:
#                 #     regressors = pa_y
#                 #    regressors=[x] + pa_x
#                 #if x in pa_x:
#                 #    print("x in pa_x")
#                     X=data[:,regressors]#.reshape(-1,1)
#                     # if transformed == True and lag is not None:
#                     #     for iter in range(X.shape[1]):
#                     #         if h(regressors[iter]) != h(y):
#                     #             X[iter] = activationfn(X[iter])
#                     Y=data[:,y]
#                     lm_out = lm.fit(X,Y)
#                     causaleff[x,y] = lm_out.coef_[regressors.index(x)]#lm_out.coef_[0]#lm_out.coef_[regressors.index(x)]#
#                     # if causaleff[x,y]>0:
#                         #     causaleff[x,y] = np.log(causaleff[x,y]+1)
#                         # else:
#                         #     causaleff[x,y] = -np.log(-causaleff[x,y]+1)
#     return causaleff
def causaleff_ida(g,data):
    #from sklearn import linear_model
    #import statsmodels.api as sm
    import numpy as np
    #from gen_data_fns import sigmoid, relu
    #Edges = list(g.edges)
    Nodes = list(g.nodes)
    causaleff=np.zeros((len(Nodes),len(Nodes)))

    for x in Nodes:
        for y in Nodes:
            if x!=y:# and x in list(nx.ancestors(g,y)):#list(g.predecessors(y)):#list(nx.ancestors(g,y)):
            #if (x,y) in Edges:
            #if y not in list(g.predecessors(x)):
            #if x in list(nx.ancestors(g,y)):
                #lm = linear_model.LinearRegression()
                pa_x = list(g.predecessors(x))
                pa_y = list(g.predecessors(y))
                if x not in pa_x:
                    regressors = pa_x + [x]
                else:
                    regressors = pa_x
                if y in pa_x:
                    causaleff[x,y] = 0
                else:
                    X=np.asarray(data[:,regressors])
                    Y=np.asarray(data[:,y])
                    X0=np.hstack((np.ones((X.shape[0],1)),X))
                    lm_out = np.linalg.lstsq(X0,Y,rcond=None)[0]
                    causaleff[x,y] = lm_out[regressors.index(x)+1]
    return causaleff
def causaleff_ida_fin(g,data):
    #from sklearn import linear_model
    #import statsmodels.api as sm
    import numpy as np
    #from gen_data_fns import sigmoid, relu
    #Edges = list(g.edges)
    Nodes = list(g.nodes)
    causaleff=np.zeros((len(Nodes),len(Nodes)))

    for x in Nodes:
        for y in Nodes:
            if x!=y and x in list(nx.ancestors(g,y)):#list(g.predecessors(y)):#list(nx.ancestors(g,y)):
                pa_x = list(g.predecessors(x))
                pa_y = list(g.predecessors(y))
                if x not in pa_x:
                    regressors = [Nodes.index(i) for i in pa_x] + [Nodes.index(x)]
                else:
                    regressors = [Nodes.index(i) for i in pa_x]#pa_x
                if y in pa_x:
                    causaleff[Nodes.index(x),Nodes.index(y)] = 0
                else:
                    X=np.asarray(data[:,regressors])
                    Y=np.asarray(data[:,Nodes.index(y)])
                    X0=np.hstack((np.ones((X.shape[0],1)),X))
                    lm_out = np.linalg.lstsq(X0,Y,rcond=None)[0]#lm_out = np.linalg.inv(X0.T @ X0) @ X0.T @ Y#model.fit()
                    #causaleff[x,y] = lm_out.coef_[regressors.index(x)]
                    causaleff[Nodes.index(x),Nodes.index(y)] = lm_out[regressors.index(Nodes.index(x))+1]
    return causaleff
def causaleff_ida_pconly(g,data,transformed=True,lag=None):
    from sklearn import linear_model
    import numpy as np
    from gen_data_fns import sigmoid, relu
    Edges = list(g.edges)
    Nodes = list(g.nodes)
    causaleff=np.zeros((len(Nodes),len(Nodes)))
    # if transformed == True and lag is None:
    #     print("Please provide lag used in transformation")
    # elif transformed == True and lag is not None:
    #     h=np.repeat(np.arange(0,len(Nodes)),lag)
    # if activation == 'centred-sigmoid':
    #     activationfn = lambda x: sigmoid(x) - 0.5
    # elif activation == 'tanh':
    #     activationfn = lambda x: np.tanh(x)
    # elif activation == 'linear':
    #     activationfn = lambda x: x
    # elif activation == 'relu':
    #     activationfn = lambda x: relu(x)
    for x in Nodes:
        for y in Nodes:
            #if x<=y:
            if x!=y and x in list(g.predecessors(y)):
            #if (x,y) in Edges:
            #if y not in list(g.predecessors(x)):
            #if x in list(nx.ancestors(g,y)):
                lm = linear_model.LinearRegression()
                pa_x = list(g.predecessors(x))
                # pa_y = list(g.predecessors(y))
                # # if x not in pa_x:
                # #     regressors = pa_x + [x]
                # # else:
                # #     regressors = pa_x
                # if y in pa_y:
                #     pa_y = pa_y.pop(y)
                # if x not in pa_y:
                #     regressors = pa_y + [x]
                # else:
                #     regressors = pa_y
                regressors=[x] + pa_x
                X=data[:,regressors]#.reshape(-1,1)
                # if transformed == True and lag is not None:
                #     for iter in range(X.shape[1]):
                #         if h(regressors[iter]) != h(y):
                #             X[iter] = activationfn(X[iter])
                Y=data[:,y]
                lm_out = lm.fit(X,Y)
                causaleff[x,y] = lm_out.coef_[regressors.index(x)]#lm_out.coef_[0]#lm_out.coef_[regressors.index(x)]#
                # if causaleff[x,y]>0:
                #     causaleff[x,y] = np.log(causaleff[x,y]+1)
                # else:
                #     causaleff[x,y] = -np.log(-causaleff[x,y]+1)
    return causaleff
# %%

# def pc_plot_out(lag,alpha,motif,n_ctrnn,n_samp,m,w,tau):
#     import networkx as nx
#     import matplotlib.pyplot as plt
#     from gen_data_fns import create_dataset4, plot_matrix
#     import numpy as np
#     from pcalg import causaleff_ida
#     dataset = create_dataset4(n_ctrnn,n_ctrnn,m,w,tau).T
#     for i in range(n_samp - 1):
#         dataset = np.vstack((dataset, create_dataset4(n_ctrnn,n_ctrnn,m,w,tau).T))
#     dataset2=data_transformed(dataset, lag)
#     (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
#                                      data_matrix=dataset2,
#                                      alpha=alpha,method='stable')
#     g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
#     causaleff = causaleff_ida(g,dataset2)
#     g1, causaleff1 = return_finaledges(g,causaleff,lag,m)
#     causaleff2 = causaleff1/np.max(np.abs(causaleff1))
#     plot_matrix(causaleff2,motif,'pc_causaleff')
#     fig, ax = plt.subplots(1,1,figsize=(10,10))
#     nx.draw(g1, with_labels= True,ax=ax)
#     fig.savefig('dag_'+motif+'.png',format = 'png', dpi =600)
#     return g1,causaleff2
def pc_original_bootstrap_plot(alpha,motif,n_ctrnn,n_samp,m,w,tau,niter=50):
    import networkx as nx
    import matplotlib.pyplot as plt
    from gen_data_fns import create_dataset4, plot_matrix
    import numpy as np
    from pcalg import causaleff_ida
    import random
    dataset = create_dataset4(n_ctrnn,n_ctrnn,m,w,tau).T
    for i in range(n_samp - 1):
        dataset = np.vstack((dataset, create_dataset4(n_ctrnn,n_ctrnn,m,w,tau).T))
    #dataset2=data_transformed(dataset, lag)
    dataset2=dataset
    g1={}
    causaleff2={}
    for iter in range(niter):
        idx=random.randint(0,dataset2.shape[0]-10000)
        #dataset3 = dataset2[random.sample(range(dataset2.shape[0]),10000),:]
        dataset3 = dataset2[idx:(idx+10000),:]
        #dataset3 = dataset2[random.sample(range(dataset2.shape[0]),10000),:]
        (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                                     data_matrix=dataset3,
                                     alpha=alpha,method='stable')
        g1[iter] = estimate_cpdag(skel_graph=g, sep_set=sep_set)
        causaleff2[iter] = causaleff_ida(g,dataset3)
        #g1[iter], causaleff1 = return_finaledges(g,causaleff,lag,m)
        #causaleff2[iter] = causaleff1#/np.max(np.abs(causaleff1))
    edgemat=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            for iter in range(niter):
                if (i,j) in list(g1[iter].edges):
                    edgemat[i,j] = edgemat[i,j]+1
    edgefinal = set([])
    for i in range(m):
        for j in range(m):
            if edgemat[i,j]>= (75*niter/100.):
                edgefinal = edgefinal | {(i,j)}
    g2=nx.DiGraph()
    g2.add_nodes_from(range(m))
    g2.add_edges_from(edgefinal)
#nx.draw(g2,with_labels=True)
    causaleff3 = np.zeros((m,m))
    for iter in range(niter):
        causaleff3 = causaleff3+causaleff2[iter]
    causaleff3 = causaleff3/niter
    # for i in list(g2.nodes):
    #     for j in list(g2.nodes):
    #         if i not in nx.ancestors(g2,j):
    #             causaleff3[i,j] = 0
    plot_matrix(causaleff3/np.max(np.abs(causaleff3)),motif,'pc_causaleff_orig')
    plot_matrix(0.5*(causaleff3!=0),motif,'pc_causalconn_orig')
    #plot_matrix(0.5*(causaleff3>0),motif,'pc_causalconn')
    #fig, ax = plt.subplots(1,1,figsize=(10,10))
    #nx.draw(G, with_labels= True,ax=ax)
    #fig.savefig('dag_'+motif+'.png',format = 'png', dpi =600)
    return g2,causaleff3
def pc_bootstrap(dataset2,lag,alpha,m,niter=50):
    import networkx as nx
    import matplotlib.pyplot as plt
    #from gen_data_fns import create_dataset4, plot_matrix
    import numpy as np
    from pcalg import causaleff_ida
    from tqdm import tqdm
    import random
    g1={}
    causaleff2={}
    for iter in tqdm(range(niter)):
        print(0)
        idx=random.randint(0,dataset2.shape[0]-100)
        #dataset3 = dataset2[random.sample(range(dataset2.shape[0]),1000),:]
        dataset3 = dataset2[idx:(idx+500),:]
        #3:41 AM started
        #(g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gp,
        #                             data_matrix=dataset3,
        #                             alpha=alpha,method='stable')
        (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                                     data_matrix=dataset3,
                                     alpha=alpha,method='stable')
        g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
        causaleff = causaleff_ida(g,dataset3)
        g1[iter], causaleff1 = return_finaledges(g,causaleff,lag,m)
        causaleff2[iter] = causaleff1#np.max(np.abs(causaleff1))
        print(1)
    edgemat=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            for iter in range(niter):
                if (i,j) in list(g1[iter].edges):
                    edgemat[i,j] = edgemat[i,j]+1
    edgefinal = set([])
    for i in range(m):
        for j in range(m):
            if edgemat[i,j]>= (25*niter/100.):
                edgefinal = edgefinal | {(i,j)}
    g2=nx.DiGraph()
    g2.add_nodes_from(range(m))
    g2.add_edges_from(edgefinal)
#nx.draw(g2,with_labels=True)
    causaleff3 = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            s=0
            for iter in range(niter):
                if causaleff2[iter][i,j]!=0:
                    causaleff3[i,j] = causaleff3[i,j]+causaleff2[iter][i,j]
                    s=s+1
            if s>0:
                causaleff3[i,j] = causaleff3[i,j]/s
    # for i in list(g2.nodes):
    #     for j in list(g2.nodes):
    #         if i not in nx.ancestors(g2,j):
    #             causaleff3[i,j] = 0
    return g2,causaleff3
def pc_bootstrap_2(dataset,lag,alpha,m, n_ctrnn, n_samp,niter=50):
    import networkx as nx
    import matplotlib.pyplot as plt
    from gen_data_fns import create_dataset4, plot_matrix
    import numpy as np
    from pcalg import causaleff_ida
    from tqdm import tqdm
    import random
    g1={}
    causaleff2={}
    for iter in tqdm(range(niter)):
        dataset2 = data_transformed_2(dataset, lag, n_ctrnn, n_samp)
        #idx=random.randint(0,dataset2.shape[0]-100)
        #dataset3 = dataset2[random.sample(range(dataset2.shape[0]),1000),:]
        dataset3 = dataset2#[idx:(idx+500),:]
        (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                                     data_matrix=dataset3,
                                     alpha=alpha,method='stable')
        g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
        causaleff = causaleff_ida(g,dataset3)
        gr1, causaleff1 = return_finaledges(g,causaleff,lag,m)
        g1[iter]=gr1
        causaleff2[iter] = causaleff1#np.max(np.abs(causaleff1))
    edgemat=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            for iter in range(niter):
                if (i,j) in list(g1[iter].edges):
                    edgemat[i,j] = edgemat[i,j]+1
    edgefinal = set([])
    for i in range(m):
        for j in range(m):
            if edgemat[i,j]>= (25*niter/100.):
                edgefinal = edgefinal | {(i,j)}
    g2=nx.DiGraph()
    g2.add_nodes_from(range(m))
    g2.add_edges_from(edgefinal)
#nx.draw(g2,with_labels=True)
    causaleff3 = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            s=0
            for iter in range(niter):
                if causaleff2[iter][i,j]!=0:
                    causaleff3[i,j] = causaleff3[i,j]+causaleff2[iter][i,j]
                    s=s+1
            causaleff3[i,j] = causaleff3[i,j]/s
    # for i in list(g2.nodes):
    #     for j in list(g2.nodes):
    #         if i not in nx.ancestors(g2,j):
    #             causaleff3[i,j] = 0
    return g2,causaleff3
# def pc_bootstrap(dataset2,lag,alpha,m,niter=50):
#     import networkx as nx
#     import matplotlib.pyplot as plt
#     from gen_data_fns import create_dataset4, plot_matrix
#     import numpy as np
#     from pcalg import causaleff_ida
#     from tqdm import tqdm
#     import random
#     g1={}
#     causaleff2={}
#     for iter in tqdm(range(niter)):
#         idx=random.randint(0,dataset2.shape[0]-100)
#         #dataset3 = dataset2[random.sample(range(dataset2.shape[0]),1000),:]
#         dataset3 = dataset2[idx:(idx+500),:]
#         (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
#                                      data_matrix=dataset3,
#                                      alpha=alpha,method='stable')
#         g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
#         causaleff = causaleff_ida(g,dataset3)
#         g1[iter], causaleff1 = return_finaledges(g,causaleff,lag,m)
#         causaleff2[iter] = causaleff1#np.max(np.abs(causaleff1))
#     edgemat=np.zeros((m,m))
#     for i in range(m):
#         for j in range(m):
#             for iter in range(niter):
#                 if (i,j) in list(g1[iter].edges):
#                     edgemat[i,j] = edgemat[i,j]+1
#     edgefinal = set([])
#     for i in range(m):
#         for j in range(m):
#             if edgemat[i,j]>= (25*niter/100.):
#                 edgefinal = edgefinal | {(i,j)}
#     g2=nx.DiGraph()
#     g2.add_nodes_from(range(m))
#     g2.add_edges_from(edgefinal)
# #nx.draw(g2,with_labels=True)
#     causaleff3 = np.zeros((m,m))
#     for i in range(m):
#         for j in range(m):
#             s=0
#             for iter in range(niter):
#                 if causaleff2[iter][i,j]!=0:
#                     causaleff3[i,j] = causaleff3[i,j]+causaleff2[iter][i,j]
#                     s=s+1
#             causaleff3[i,j] = causaleff3[i,j]/s
#     # for i in list(g2.nodes):
#     #     for j in list(g2.nodes):
#     #         if i not in nx.ancestors(g2,j):
#     #             causaleff3[i,j] = 0
#     return g2,causaleff3
def pc_plot_out(lag,alpha,motif,n_ctrnn,n_samp,m,w,tau,niter=50,isPlot=True):
    import networkx as nx
    import matplotlib.pyplot as plt
    from gen_data_fns import create_dataset4, plot_matrix
    import numpy as np
    from pcalg import causaleff_ida, pc_bootstrap
    import random
    dataset = create_dataset4(n_ctrnn,n_ctrnn,m,w,tau).T
    for i in range(n_samp - 1):
        dataset = np.vstack((dataset, create_dataset4(n_ctrnn,n_ctrnn,m,w,tau).T))
    dataset2=data_transformed(dataset, lag)
    # (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
    #                                  data_matrix=dataset2, alpha=alpha,method='stable')
    # g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    # causaleff = causaleff_ida(g,dataset2)
    # g1, causaleff1 = return_finaledges(g,causaleff,lag,m)
    # causaleff1 = causaleff1/np.max(np.abs(causaleff1))
    # nx.draw(g1,with_labels=True)
    #%%
    #g1[iter], causaleff1 = return_finaledges(g,causaleff,lag,m)
    #causaleff2[iter] = causaleff1/np.max(np.abs(causaleff1))
    g2,causaleff2 = pc_bootstrap(dataset2,lag,alpha,m,niter=50)
    #%%
    if isPlot is True:
        plot_matrix(causaleff2,motif,'pc_causaleff')
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        nx.draw(g2, with_labels= True,ax=ax)
        fig.savefig('dag_'+motif+'.png',format = 'png', dpi =600)
    return g2,causaleff2
def pc_plot_out2(lag,alpha,motif,n_ctrnn,n_samp,m,w,tau,niter=50):
    import networkx as nx
    import matplotlib.pyplot as plt
    from gen_data_fns import create_dataset4, plot_matrix, create_dataset5
    import numpy as np
    from pcalg import causaleff_ida, pc_bootstrap
    import random
    from tqdm import tqdm
#%%
    dataset = create_dataset4(n_ctrnn,n_ctrnn,m,w,tau).T
    for i in range(n_samp - 1):
        dataset = np.vstack((dataset, create_dataset4(n_ctrnn,n_ctrnn,m,w,tau).T))
    #dataset2=data_transformed(dataset, lag)
    #dataset0=dataset#[random.sample(range(dataset.shape[0]),10000),:]
    dataset01=data_transformed(dataset, lag)
    # (g0, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
    #                                 data_matrix=dataset01,
    #                                 alpha=alpha,method='stable')
    # g0 = estimate_cpdag(skel_graph=g0, sep_set=sep_set)
    # causaleff0 = causaleff_ida(g0,data_transformed(dataset01, lag))
    # g0, causaleff0 = return_finaledges(g0,causaleff0,lag,m)
#%%    
    g0, causaleff0 = pc_bootstrap(dataset01,lag,alpha,m,niter=25)
    g01=g0.copy()
    g01.remove_edges_from(g01.selfloop_edges())
    G=g0
    g1={}
    causaleff2={}
    #niter=50
    #edgemat_btrsp={}
#%%
    #niter=50
    #for iter0 in tqdm(range(niter)):#tqdm(range(50)):
    for k in tqdm(range(m)):
        #if g01.out_degree(k)==0 and g01.in_degree(k)==1:
        iter = k
        # dataset3 = data_transformed(np.delete(dataset2,[k],axis=1),lag)
        # for node in range(m):
        #     if len(g0.predecessors(node))>0:
        #         lm = linear_model.LinearRegression()
        #         pa_x = list(g0.predecessors(node))
        #         regressors = pa_x
        #         X=data[:,regressors]
        #         Y=data[:,y]
        #         lm_out = lm.fit(X,Y)
        #         causaleff[x,y] = lm_out.coef_[regressors.index(x)]
        dataset2 = data_transformed(np.delete(dataset, [k],axis=1),lag) 
        idx=random.randint(0,dataset2.shape[0]-100)
        #dataset2[:,random.sample(range(m),1)[0]]=np.random.normal(10,10,dataset2.shape[0])
        #k=random.sample(range(m),1)[0]
        dataset3 = dataset2[idx:(idx+500),:]
        labels = list(range(m))
        labels.remove(k)
        relabels = dict(zip(range(m-1),labels))
        (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                                    data_matrix=dataset3,
                                    alpha=alpha,method='stable')
        g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
        causaleff = causaleff_ida(g,dataset3)
        #g1, causaleff1 = return_finaledges(g,causaleff,lag,m-1)
        #g1 = nx.relabel_nodes(g1,relabels)
        #nx.draw(g1,with_labels=True)
        g1[iter], causaleff1 = return_finaledges(g,causaleff,lag,m-1)
        g1[iter] = nx.relabel_nodes(g1[iter],relabels)
        causaleff2[iter] = causaleff1/np.max(np.abs(causaleff1))

        # for (node1,node2) in list(g1[iter].edges):
        #     if (node1,node2) not in list(g01.edges):
        #         G.add_edge(node1,node2)
        for (node1,node2) in list(g1[iter].edges):
                if (k,node2) not in list(G.edges):
                    if node1 != node2 and node1 in nx.ancestors(G,k):
                        ch=0
                        for z in [elem for elem in range(m) if elem not in [k,node2]]:
                            if (z,node2) in list(G.edges):
                                ch=1#(node1,node2) not in list(g01.edges) and node1 in nx.ancestors(g01,k) and (k,node2) not in list(g01.edges):#(node1, k) in list(g01.edges) and (k,node2) not in list(g01.edges):
                        if ch==0:
                            G.add_edge(k,node2)
#%%
                    # for nodeorig in list(g0.nodes):
        #             #     if (node1,node2) not in list(g0.edges) and (node1,k) not in list 
        # edgemat=np.zeros((m,m))
        # for i in range(m):
        #     for j in range(m):
        #         for iter1 in range(niter):
        #             if (i,j) in list(g1[iter1].edges):
        #                 edgemat[i,j] = edgemat[i,j]+1
        #edgemat_btrsp[iter0]=edgemat
#%%
#    out=sum([i for i in list(edgemat_btrsp.values())])/len(edgemat_btrsp)

#%%
    # g2=g0.to_undirected()
    # edgefinal = set([])
    # for i in range(m):
    #     temp = set([])
    #     for j in range(m):
    #         if nx.has_path(g2,j,i):
    #             continue
    #         else:
    #             if out[j,i]>=0.5:#edgemat[i,j]>=20*niter/100.:
    #                 temp = temp | {(j,i)}
    #     if len(temp)>1:
    #         temp2 = set([])
    #         for (j1,i1) in temp:
    #             test_temp = sum([(l[0] in nx.ancestors(g0,i1)) for l in temp-{(j1,i1)}])
    #             if test_temp == len(temp-{(j1,i1)}):
    #                 temp2 = {(j1,i1)}
    #     else:
    #         temp2 = temp
    # g2=nx.DiGraph()
    # g2.add_nodes_from(range(m))
    # g2.add_edges_from(edgefinal)
#%%
    nx.draw(G,with_labels=True)
    #G1=G.copy()
    #G1.remove_edges_from(G1.selfloop_edges())
    # for (node1,node2) in list(G.edges):
    #     if causaleff0[node1,node2]/np.max(np.abs(causaleff0))<0.05:
    #         G.remove_edge(node1,node2)
    causaleff3 = np.zeros((m,m))
    for i in list(G.nodes):
        for j in list(G.nodes):
    #    for iter in range(niter):
            if i in list(nx.ancestors(G,j)):
                causaleff3[i,j] = causaleff0[i,j]#causaleff3[i,j]+causaleff2[iter][i,j]
    #    causaleff3[i,j] = causaleff3[i,j]/niter



    plot_matrix(causaleff3,motif,'pc_causaleff')
    plot_matrix(0.5*(causaleff3!=0),motif,'pc_causalconn')
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    nx.draw(G, with_labels= True,ax=ax)
    fig.savefig('dag_'+motif+'.png',format = 'png', dpi =600)
    return G,causaleff3
#%%
def pc_plot_out3(lag,alpha,motif,n_ctrnn,n_samp,m,w,tau,niter=50):
    import networkx as nx
    import matplotlib.pyplot as plt
    from gen_data_fns import create_dataset4, plot_matrix, create_dataset5
    import numpy as np
    from pcalg import causaleff_ida, pc_bootstrap
    import random
    from tqdm import tqdm
#%%
    dataset = create_dataset4(n_ctrnn,n_ctrnn,m,w,tau).T
    for i in range(n_samp - 1):
        dataset = np.vstack((dataset, create_dataset4(n_ctrnn,n_ctrnn,m,w,tau).T))
#%%
    g0, causaleff0 = pc_bootstrap_2(dataset,lag,alpha,m,n_ctrnn, n_samp,niter)
    g01=g0.copy()
    g01.remove_edges_from(g01.selfloop_edges())
    G=g0
    g1={}
    causaleff2={}
    #niter=50
    #edgemat_btrsp={}
#%%
    #niter=50
    #for iter0 in tqdm(range(niter)):#tqdm(range(50)):
    # for k in tqdm(range(m)):
    #     iter = k
    #     dataset2 = data_transformed(np.delete(dataset, [k],axis=1),lag) 
    #     idx=random.randint(0,dataset2.shape[0]-100)
    #     #dataset2[:,random.sample(range(m),1)[0]]=np.random.normal(10,10,dataset2.shape[0])
    #     #k=random.sample(range(m),1)[0]
    #     dataset3 = dataset2[idx:(idx+500),:]
    #     labels = list(range(m))
    #     labels.remove(k)
    #     relabels = dict(zip(range(m-1),labels))
    #     (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
    #                                 data_matrix=dataset3,
    #                                 alpha=alpha,method='stable')
    #     g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    #     causaleff = causaleff_ida(g,dataset3)
    #     g1[iter], causaleff1 = return_finaledges(g,causaleff,lag,m-1)
    #     g1[iter] = nx.relabel_nodes(g1[iter],relabels)
    #     causaleff2[iter] = causaleff1/np.max(np.abs(causaleff1))

    #     for (node1,node2) in list(g1[iter].edges):
    #             if node1 !=node2 and (node1,node2) not in list(g01.edges) and node1 in nx.ancestors(g01,k) and (k,node2) not in list(g01.edges):#(node1, k) in list(g01.edges) and (k,node2) not in list(g01.edges):
    #                 G.add_edge(k,node2)

    causaleff3 = np.zeros((m,m))
    for i in list(G.nodes):
        for j in list(G.nodes):
            if i in list(nx.ancestors(G,j)):
                causaleff3[i,j] = causaleff0[i,j]#causaleff3[i,j]+causaleff2[iter][i,j]

    plot_matrix(causaleff3/np.max(np.abs(causaleff3)),motif,'pc_causaleff')
    plot_matrix(0.5*(causaleff3!=0),motif,'pc_causalconn')
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    nx.draw(G, with_labels= True,ax=ax)
    fig.savefig('dag_'+motif+'.png',format = 'png', dpi =600)
    return G,causaleff3
#%%
def ablation(lag,dataset,g01,G,k):
  m=dataset.shape[1]
  dataset3 = data_transformed(np.delete(dataset, [k],axis=1),lag) 
  g2=G.copy()
  labels = list(range(m))
  labels.remove(k)
  relabels = dict(zip(range(m-1),labels))
  (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                              data_matrix=dataset3,
                              alpha=alpha,method='stable')
  g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
  causaleff = causaleff_ida(g,dataset3)

  g1, causaleff1 = return_finaledges(g,causaleff,lag,m-1)
  g1 = nx.relabel_nodes(g1,relabels)
  #causaleff2 = causaleff1/np.max(np.abs(causaleff1))

  for (node1,node2) in list(g1.edges):
          if node1 !=node2 and (node1,node2) not in list(g01.edges) and node1 in nx.ancestors(g01,k) and (k,node2) not in list(g01.edges):#(node1, k) in list(g01.edges) and (k,node2) not in list(g01.edges):
              g2.add_edge(k,node2)
  return g2

def pc_plot_realdata(dataset,lag,alpha,motif,niter=10):
    import networkx as nx
    import matplotlib.pyplot as plt
    from gen_data_fns import create_dataset4, plot_matrix
    import numpy as np
    from pcalg import causaleff_ida, pc_bootstrap
    import random
    from tqdm import tqdm
    import multiprocessing as mp
    from functools import partial
    import time
    m=dataset.shape[1]
    dataset01=data_transformed(dataset, lag)
    g0, causaleff0 = pc_bootstrap_realdata(dataset01,lag,alpha,m,niter)
    g01=g0.copy()
    g01.remove_edges_from(list(nx.selfloop_edges(g01)))
    G=g0
    #nx.draw(G, with_labels= True,ax=ax)
    #g1=[]
    causaleff2={}

    pool = mp.Pool(4)
    func= partial(ablation,lag,dataset,g01,G)
    t1=time.time()
    g1 = pool.map(func,range(m))
    print(time.time()-t1)
    #for k in tqdm(range(m)):
        #idx=random.randint(0,dataset.shape[0]-1000)
        #k=random.sample(range(m),1)[0]
        #dataset2 = dataset[idx:(idx+5000),:]
        #g1.append(ablation(k,lag,dataset,g01,G))
    G=nx.compose_all(g1)
    nx.draw(G,with_labels=True)
    causaleff3 = np.zeros((m,m))
    for i in list(G.nodes):
        for j in list(G.nodes):
            if i in nx.ancestors(G,j):
                causaleff3[i,j] = causaleff0[i,j]#causaleff3[i,j]+causaleff2[iter][i,j]
    causaleff4 = np.zeros((m,m))
    for i in list(G.nodes):
        for j in list(G.nodes):
            if i in G.predecessors(j):
                causaleff4[i,j] = causaleff0[i,j]
    plot_matrix(causaleff3/np.max(np.abs(causaleff3)),motif,'pc_causaleff')
    plot_matrix(0.5*(causaleff3!=0),motif,'pc_causalconn')
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    nx.draw(G, with_labels= True,ax=ax)
    fig.savefig('dag_'+motif+'.png',format = 'png', dpi =600)
    return G,causaleff3,causaleff4
# #%%
# def pc_plot_realdata(dataset,lag,alpha,motif,niter=10):
#     import networkx as nx
#     import matplotlib.pyplot as plt
#     from gen_data_fns import create_dataset4, plot_matrix
#     import numpy as np
#     from pcalg import causaleff_ida, pc_bootstrap
#     import random
#     from tqdm import tqdm
#     m=dataset.shape[1]
#     dataset01=data_transformed(dataset, lag)
#     g0, causaleff0 = pc_bootstrap_realdata(dataset01,lag,alpha,m,niter)
#     g01=g0.copy()
#     g01.remove_edges_from(g01.selfloop_edges())
#     G=g0
#     #nx.draw(G, with_labels= True,ax=ax)
#     g1={}
#     causaleff2={}
#     g2=[]
#     for k in tqdm(range(m)):
#         #idx=random.randint(0,dataset.shape[0]-1000)
#         #k=random.sample(range(m),1)[0]
#         #dataset2 = dataset[idx:(idx+5000),:]
#         dataset3 = data_transformed(np.delete(dataset, [k],axis=1),lag) 
#         g_iter=G.copy()
#         labels = list(range(m))
#         labels.remove(k)
#         relabels = dict(zip(range(m-1),labels))
#         (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
#                                     data_matrix=dataset3,
#                                     alpha=alpha,method='stable')
#         g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
#         causaleff = causaleff_ida(g,dataset3)

#         g1[iter], causaleff1 = return_finaledges(g,causaleff,lag,m-1)
#         g1[iter] = nx.relabel_nodes(g1[iter],relabels)
#         causaleff2[iter] = causaleff1/np.max(np.abs(causaleff1))

#         for (node1,node2) in list(g1[iter].edges):
#                 if node1 !=node2 and (node1,node2) not in list(g01.edges) and node1 in nx.ancestors(g01,k) and (k,node2) not in list(g01.edges):#(node1, k) in list(g01.edges) and (k,node2) not in list(g01.edges):
#                     g_iter.add_edge(k,node2)
#     nx.draw(G,with_labels=True)
#     causaleff3 = np.zeros((m,m))
#     for i in list(G.nodes):
#         for j in list(G.nodes):
#             if i in nx.ancestors(G,j):
#                 causaleff3[i,j] = causaleff0[i,j]#causaleff3[i,j]+causaleff2[iter][i,j]
#     causaleff4 = np.zeros((m,m))
#     for i in list(G.nodes):
#         for j in list(G.nodes):
#             if i in G.predecessors(j):
#                 causaleff4[i,j] = causaleff0[i,j]
#     plot_matrix(causaleff3/np.max(np.abs(causaleff3)),motif,'pc_causaleff')
#     plot_matrix(0.5*(causaleff3!=0),motif,'pc_causalconn')
#     fig, ax = plt.subplots(1,1,figsize=(10,10))
#     nx.draw(G, with_labels= True,ax=ax)
#     fig.savefig('dag_'+motif+'.png',format = 'png', dpi =600)
#     return G,causaleff3,causaleff4
#%%
#lag=2
def pc_bootstrap_realdata(dataset2,lag,alpha,m,niter=10):
    import networkx as nx
    import matplotlib.pyplot as plt
    from gen_data_fns import create_dataset4, plot_matrix
    import numpy as np
    from pcalg import causaleff_ida
    import random
    from tqdm import tqdm
    g1={}
    causaleff2={}
    for iter in tqdm(range(niter)):
        idx=iter#random.randint(0,dataset2.shape[0]-10)
        #dataset3 = dataset2[random.sample(range(dataset2.shape[0]),1000),:]
        dataset3 = dataset2[idx:(dataset2.shape[0]-niter+idx+1),:]#(idx+dataset2.shape[0]-10),:]
        (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                                     data_matrix=dataset3,
                                     alpha=alpha,method='stable')
        g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
        causaleff = causaleff_ida(g,dataset3)
        g1[iter], causaleff1 = return_finaledges(g,causaleff,lag,m)
        causaleff2[iter] = causaleff1#/np.max(np.abs(causaleff1))
    edgemat=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            for iter in range(niter):
                if (i,j) in list(g1[iter].edges):
                    edgemat[i,j] = edgemat[i,j]+1
    edgefinal = set([])
    for i in range(m):
        for j in range(m):
            if edgemat[i,j]>= (25*niter/100.):
                edgefinal = edgefinal | {(i,j)}
    g2=nx.DiGraph()
    g2.add_nodes_from(range(m))
    g2.add_edges_from(edgefinal)
#nx.draw(g2,with_labels=True)
    causaleff3 = np.zeros((m,m))
    for iter in range(niter):
        causaleff3 = causaleff3+causaleff2[iter]
    causaleff3 = causaleff3/niter
    # for i in list(g2.nodes):
    #     for j in list(g2.nodes):
    #         if i not in nx.ancestors(g2,j):
    #             causaleff3[i,j] = 0
    return g2,causaleff3
def data_transformed(data, lag):
    import numpy as np
    n = data.shape[0]
    p = data.shape[1]
    lag1=lag+1
    new_n = int(np.floor((n-lag)/(2*lag1))*(2*lag1))
    data=data[:new_n,:]
    data2=np.zeros((int(new_n/(2*lag1)),p*lag1))
    for i in range(p):
        for j in range(lag1):
            data2[:,lag1*i+j]=data[j::(2*lag1),i]
    return data2
def data_transformed_mod(data, lag, node):
    import numpy as np
    n = data.shape[0]
    p = data.shape[1]
    lag1=lag+1
    new_n = int(np.floor((n-lag)/(2*lag1))*(2*lag1))
    data=data[:new_n,:]
    data2=np.zeros((int(new_n/(2*lag1)),p*(lag1-1)+1))
    #data2[:,lag1*i]=data[j::(2*lag1),i]
    for i in range(p):
        for j in range(1,lag1):
            data2[:,(lag1-1)*i+j]=data[j::(2*lag1),i]
    data2[:,0] = data[0::(2*lag1),node]
    return data2

def data_transformed_overlapping(data, tau):
    import numpy as np
    n = data.shape[0]
    p = data.shape[1]
    #lag1=lag+1
    #lag = lag+1
    new_n = n-tau+1#int(np.floor((n-lag)/(2*lag1))*(2*lag1))
    p_new = p*tau
    if tau <1 :
        return('lag should be >= 1')
    elif tau == 1:
        data2 = data
    else:
        data2=np.zeros((new_n,p_new))
        for t in range(0,n-tau+1):
                data2[t,:]=np.hstack(data[t:t+tau,])
    return data2

def data_transformed_btstrp(data):
    import numpy as np
    from arch import bootstrap
    from numpy.random import RandomState
    band = bootstrap.optimal_block_length(data)
    n = data.shape[0]
    p = data.shape[1]
    data2=np.zeros((2*n*p,n*p))
    bs = bootstrap.StationaryBootstrap(np.median(band.iloc[:,0]),data,random_state=RandomState(111))
    t=0
    for data1 in bs.bootstrap(2*n*p):
        data2[t,:]=np.hstack(data1[0][0])
        t=t+1
    return data2
# def data_transformed_v2(data, lag, window):
#     import numpy as np
#     n = data.shape[0]
#     m = data.shape[1]
#     lag1=lag+1
#     data1 = np.zeros((int(n/(2*window)),m*lag1))
#     for t1 in range(int(n/(2*window))):
#         t=2*window*t1
#         for p1 in range(m):
#             for l in range(lag1):
#                 data1[t1,lag1*p1+l]=np.mean(data[(t+l):(t+l+window),p1])
#     return data1

def data_transformed1(data, lag):
    import numpy as np
    n = data.shape[0]
    p = data.shape[1]
    lag1=2
    lag2={}
    lag2[0]=0
    lag2[1]=lag
    lag2[2]=lag+1
    new_n = int(np.floor((n-lag)/(2*lag2[2]))*(2*lag2[2]))
    data=data[:new_n,:]
    data2=np.zeros((int(np.floor((n-lag)/(2*lag2[2]))),p*lag1))#np.zeros((int(new_n/(2*lag2[2])),p*lag1))
    for i in range(p):
        for j in range(lag1):
            data2[:,lag1*i+j]=data[lag2[j]::(2*lag2[2]),i]
    return data2
def data_transformed_nogaps(data, lag):
    import numpy as np
    n = data.shape[0]
    p = data.shape[1]
    lag1=lag+1
    new_n = int(np.floor((n-lag)/(lag1))*(lag1))
    data=data[:new_n,:]
    data2=np.zeros((int(new_n/(lag1)),p*lag1))
    for i in range(p):
        for j in range(lag1):
            data2[:,lag1*i+j]=data[j::(lag1),i]
    return data2
# def data_transformed(data, lag):
#     import numpy as np
#     n = data.shape[0]
#     p = data.shape[1]
#     lag1=lag+1
#     new_n = int(np.floor(n/(2*lag1))*(2*lag1))
#     data=data[:new_n,:]
#     data2=np.zeros((int(new_n/(2*lag1)),p*lag1))
#     for i in range(p):
#         for j in range(lag1):
#             data2[:,lag1*i+j]=data[j::(2*lag1),i]
#     return data2

# def data_transformed_old(data, lag):
#     import numpy as np
#     n = data.shape[0]
#     p = data.shape[1]
#     lag1=lag+1
#     new_n = int(np.floor(n/lag1)*lag1)
#     data=data[:new_n,:]
#     data2=np.zeros((int(new_n/lag1),p*lag1))
#     for i in range(p):
#         for j in range(lag1):
#             data2[:,lag1*i+j]=data[j::lag1,i]
#     return data2
def data_transformed_fin(data,lag,maxdeg=1):
    n = data.shape[0]
    p = data.shape[1]
    #data2=np.zeros(data.shape)
    data1 = data
    if maxdeg>1:
        for diter in range(2,maxdeg+1):
           data1=data1+ (data**diter)/np.math.factorial(diter)
    data2=np.cumsum(data1,axis=0)[lag:,] - np.cumsum(data1,axis=0)[:-lag,]
    data3=np.zeros((int(n/(2*lag)),p))#np.zeros((int((n-lag-1)/(2*lag)),2*p))
    k=0
    for i in range(lag+1,n,2*lag):
        data3[k,:] = data2[i-1-lag,:]#np.hstack((data[i,:],data2[i-1-lag,:]))
        k=k+1
    return data3
def data_transformed_2(data, lag,n_ctrnn,n_samp):
    import numpy as np
    import random as random
    r = random.randint(0,n_ctrnn-lag-1)
    lag1=lag+1
    n = data.shape[0]
    p = data.shape[1]
    n_new = n_samp
    p_new = (lag+1)*p
    y=np.zeros((n_new,p_new))
    for i in range(r,r+lag+1):
        for j in range(p):
            y[:,lag*(j-1)+i-r] = data[i::n_ctrnn,j]
    return y
#causaleff = causaleff_ida(g,dataset2)
#g1, causaleff1 = return_finaledges(g,causaleff,lag1,p)
    #nx.draw(g1,with_labels=True)
#%%plot graph
def return_relabels(lag,p):
    lag1=lag+1
    labels={}#strs = ["" for x in range(lag1*p)]#np.empty(lag1*p,dtype=str)
    for i in range(p):
        for j in range(lag1):
            labels[lag1*i+j]=str(i)+'_'+str(j)
    return labels

def orient(g,lag,m):
    import networkx as nx
    import numpy as np
    labels=np.arange(0,(lag+1)*m)

    edge=set([])
    labelmat=labels.reshape((m,lag+1))
    for i in range(m):
        for k in range(m):
            for j in range(lag+1):
                for l in range(lag+1):
                    if j<=l:# and j>=l-lag:
                    #if j==l-1:
                        if (labelmat[i,j],labelmat[k,l]) in g.edges or (labelmat[k,l],labelmat[i,j]) in g.edges:
                            edge = edge | {(labelmat[i,j],labelmat[k,l])}
    g1 = nx.DiGraph()
    g1.add_nodes_from(g.nodes)
    g1.add_edges_from(edge)
    return g1
#%%
def return_finaledges(g,causaleff,lag,m):
    import networkx as nx
    import numpy as np
    labels=np.arange(0,(lag+1)*m)

    # edge=set([])
    labelmat=labels.reshape((m,lag+1))
    causaleff1 = np.zeros((m,m))
    causaleff2 = np.zeros((m,m))
    #pval_fin = np.zeros((m,m))
    edge_mat = np.zeros((m,m))==1
    # for i in range(m):
    #     for k in range(m):
    #         for j in range(lag+1):
    #             for l in range(lag+1):
    #                 if j<=l:# and j>=l-lag:
    #                 #if j==l-1:
    #                     if (labelmat[i,j],labelmat[k,l]) in g.edges or (labelmat[k,l],labelmat[i,j]) in g.edges:
    #                         edge = edge | {(labelmat[i,j],labelmat[k,l])}
    # g1 = nx.DiGraph()
    # g1.add_nodes_from(g.nodes)
    # g1.add_edges_from(edge)
    g1=g
    for i in range(m):
        for k in range(m):
            acc=[]
            ansacc=[]
            for j in range(lag+1):
                for l in range(lag+1):
                    if j<=l:
                        if (labelmat[i,j],labelmat[k,l]) in g1.edges:
                            #print((labelmat[i,j],labelmat[k,l]))
                           # edge = edge | {(i,k)}
                            edge_mat[i,k]=True
                            #acc.append(causaleff[labelmat[i,j],labelmat[k,l]])
                        #for lagit in range(1,lag+1):
                        #if labelmat[i,j] in nx.ancestors(g,labelmat[k,l]):####CF at delay=lag
                            acc.append(causaleff[labelmat[i,j],labelmat[k,l]])
                        if labelmat[i,j] in nx.ancestors(g1,labelmat[k,l]):
                            ansacc.append(causaleff[labelmat[i,j],labelmat[k,l]])
            if len(acc)>0:
                causaleff1[i,k] = np.mean(acc)#acc[iacc]#np.mean(acc)
            if len(ansacc)>0:
                causaleff2[i,k] = np.mean(ansacc)
    return edge_mat.astype(int), causaleff1,causaleff2#, pval_fin

def return_finaledges_v2(g,causaleff,lag,m):
    import networkx as nx
    import numpy as np
    labels=np.arange(0,lag*m)

    # edge=set([])
    labelmat=labels.reshape((m,lag))
    causaleff1 = np.zeros((m,m))
    causaleff2 = np.zeros((m,m))
    #pval_fin = np.zeros((m,m))
    edge_mat = np.zeros((m,m))==1
    # for i in range(m):
    #     for k in range(m):
    #         for j in range(lag+1):
    #             for l in range(lag+1):
    #                 if j<=l:# and j>=l-lag:
    #                 #if j==l-1:
    #                     if (labelmat[i,j],labelmat[k,l]) in g.edges or (labelmat[k,l],labelmat[i,j]) in g.edges:
    #                         edge = edge | {(labelmat[i,j],labelmat[k,l])}
    # g1 = nx.DiGraph()
    # g1.add_nodes_from(g.nodes)
    # g1.add_edges_from(edge)
    g1=g
    for i in range(m):
        for k in range(m):
            acc=[]
            ansacc=[]
            for j in range(lag):
                for l in range(lag):
                    if j<=l:
                        if (labelmat[i,j],labelmat[k,l]) in g1.edges:
                            #print((labelmat[i,j],labelmat[k,l]))
                           # edge = edge | {(i,k)}
                            edge_mat[i,k]=True
                            #acc.append(causaleff[labelmat[i,j],labelmat[k,l]])
                        #for lagit in range(1,lag+1):
                        #if labelmat[i,j] in nx.ancestors(g,labelmat[k,l]):####CF at delay=lag
                            acc.append(causaleff[labelmat[i,j],labelmat[k,l]])
                        if labelmat[i,j] in nx.ancestors(g1,labelmat[k,l]):
                            ansacc.append(causaleff[labelmat[i,j],labelmat[k,l]])
            if len(acc)>0:
                causaleff1[i,k] = np.mean(acc)#acc[iacc]#np.mean(acc)
            if len(ansacc)>0:
                causaleff2[i,k] = np.mean(ansacc)
    return edge_mat.astype(int), causaleff1,causaleff2#, pval_fin

# def return_finaledges(g,causaleff,lag,m):
#     import networkx as nx
#     import numpy as np
#     labels=np.arange(0,(lag+1)*m)

#     edge=set([])
#     labelmat=labels.reshape((m,lag+1))
#     causaleff1 = np.zeros((m,m))
#     edge_mat = np.zeros((m,m))==1
#     for i in range(m):
#         for k in range(m):
#             acc=[]
#             for j in range(lag+1):
#                 for l in range(lag+1):
#                     if j<=l:# and j>=l-lag:
#                     #if j==l-1:
#                         if (labelmat[i,j],labelmat[k,l]) in g.edges or (labelmat[k,l],labelmat[i,j]) in g.edges:
#                             #print((labelmat[i,j],labelmat[k,l]))
#                             # edge = edge | {(i,k)}
#                             edge_mat[i,k]=True
#                             #acc.append(causaleff[labelmat[i,j],labelmat[k,l]])
#                         #for lagit in range(1,lag+1):
#                         #if labelmat[i,j] in nx.ancestors(g,labelmat[k,l]):####CF at delay=lag
#                             acc.append(causaleff[labelmat[i,j],labelmat[k,l]])
#             if len(acc)>0:
#                 #iacc=np.argmax(np.abs(acc))
#                 causaleff1[i,k] = np.mean(acc)#acc[iacc]#np.mean(acc)
#     return edge_mat.astype(int), causaleff1

# def return_finaledges_v2(g,causaleff,lag,m):
#     import networkx as nx
#     import numpy as np
#     labels=np.arange(0,(lag+1)*m)

#     edge=set([])
#     labelmat=labels.reshape((m,lag+1))
#     causaleff1 = np.zeros((m,m))
#     edge_mat = np.zeros((m,m))==1
#     for i in range(m):
#         for k in range(m):
#             acc=[]
#             for j in range(lag+1):
#                 for l in range(lag+1):
#                     if j<=l:# and j>=l-lag:
#                     #if j==l-1:
#                         if (labelmat[i,j],labelmat[k,l]) in g.edges or (labelmat[k,l],labelmat[i,j]) in g.edges:
#                             #print((labelmat[i,j],labelmat[k,l]))
#                            # edge = edge | {(i,k)}
#                             edge_mat[i,k]=True
#                             #acc.append(causaleff[labelmat[i,j],labelmat[k,l]])
#                         #for lagit in range(1,lag+1):
#                         #if labelmat[i,j] in nx.ancestors(g,labelmat[k,l]):####CF at delay=lag
#                             acc.append(causaleff[labelmat[i,j],labelmat[k,l]])
#             if len(acc)>0:
#                 #iacc=np.argmax(np.abs(acc))
#                 causaleff1[i,k] = np.mean(acc)#acc[iacc]#np.mean(acc)
#     #g1=nx.DiGraph()
#     #g1.add_nodes_from(range(m))
#     #g1.add_edges_from(edge)
#     #g1.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
#     #g1.graph['graph'] = {'scale': '3'}

#     return edge_mat.astype(int), causaleff1

def return_finaledges_fin(g,causaleff,p):
    import networkx as nx
    import numpy as np
    causaleff1 = np.zeros(p)
    edge_mat = np.zeros(p)==1
    for k in range(p):
        #for k in range(p):
        if (k+1,0) in g.edges:#(k,i) in g.edges or 
            edge_mat[k]=True
            causaleff1[k]=causaleff[k+1,0]#causaleff[k,i]*int((k,i) in g.edges) + 
    return edge_mat.astype(int), causaleff1

# def return_finaledges(g,causaleff,lag,m):
#     import networkx as nx
#     import numpy as np
#     labels=np.arange(0,(lag+1)*m)

#     edge=set([])
#     labelmat=labels.reshape((m,lag+1))
#     causaleff1 = np.zeros((m,m))
#     for i in range(m):
#         for k in range(m):
#             acc=[]
#             for j in range(lag+1):
#                 for l in range(lag+1):
#                     if j<=l:
#                     #if j==l-1:
#                         if (labelmat[i,j],labelmat[k,l]) in g.edges:
#                             #print((labelmat[i,j],labelmat[k,l]))
#                             edge = edge | {(i,k)}
#                             #acc.append(causaleff[labelmat[i,j],labelmat[k,l]])
#                         if j==l-lag and labelmat[i,j] in nx.ancestors(g,labelmat[k,l]):####CF at delay=lag
#                             acc.append(causaleff[labelmat[i,j],labelmat[k,l]])
#             if len(acc)>0:
#                 causaleff1[i,k] = np.mean(acc)
    
#     g1=nx.DiGraph()
#     g1.add_nodes_from(range(m))
#     g1.add_edges_from(edge)
#     #g1.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved'}
#     #g1.graph['graph'] = {'scale': '3'}
#     return g1, causaleff1

def find_lag(data):
    p=data.shape[1]
    n=data.shape[0]
    valuelag= np.zeros(int(n/2))
    for lag in range(1,int(n/2)):
        vecnorm = np.zeros(n)
        for i in range(n-lag):
            vecnorm[i]=np.linalg.norm(data[i,:]-data[(i+lag),:])
        valuelag[lag]=np.mean(vecnorm)
        #np.where(valuelag>=np.quantile(valuelag,0.1))[0][0]
        return valuelag
# %%
def causaleffin(G,data_trans,lag):
    import numpy as np
    Nodes = list(G.nodes)
    m = len(Nodes)
    causaleff=np.zeros(m,m)
    causaleff2=np.zeros(m,m)
    labels=np.arange(0,(lag+1)*m)
    labelmat=labels.reshape((m,lag+1))

    for x1 in Nodes:
        for y1 in Nodes:
            if x1!=y1 and (x1,y1) in G.edges:#list(nx.ancestors(g,y)):#list(g.predecessors(y)):#list(nx.ancestors(g,y)):
                for i in range(lag+1):
                    for j in range(i,lag+1):
                    #if (x,y) in Edges:
                    #if y not in list(g.predecessors(x)):
                    #if x in list(nx.ancestors(g,y)):
                        #lm = linear_model.LinearRegression()
                        x = labelmat[x1,i]
                        y = labelmat[y1,j]
                        pa_x = list(g.predecessors(x))
                        pa_y = list(g.predecessors(y))
                        if x not in pa_x:
                            regressors = pa_x + [x]
                        else:
                            regressors = pa_x
                        if y in pa_x:
                            causaleff[x,y] = 0
                        else:
                            X=np.asarray(data[:,regressors])
                            Y=np.asarray(data[:,y])
                            X0=np.hstack((np.ones((X.shape[0],1)),X))
                            lm_out = np.linalg.lstsq(X0,Y,rcond=None)[0]
                            causaleff[x,y] = lm_out[regressors.index(x)+1]
    return causaleff
