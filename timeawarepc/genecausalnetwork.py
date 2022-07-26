def val_pc_hsic1(data_trans,alpha):
    from timeawarepc.pcalg_ray import estimate_skeleton, hsic_condind, estimate_cpdag, causaleff_ida, ci_test_gauss
    import networkx as nx
    (g, sep_set) = estimate_skeleton(indep_test_func=hsic_condind,
                                        data_matrix=data_trans,
                                        alpha=alpha,method='stable')
    g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    A=nx.adjacency_matrix(g).toarray()
    val_matrix = A
    causaleff = causaleff_ida(g,data_trans)
    return val_matrix, causaleff
def val_pc_hsic(data_trans,alpha):
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    import rpy2.rlike.container as rlc
    from rpy2.robjects import pandas2ri   
    import pandas as pd
    import numpy as np
    import networkx as nx
    d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
    kpcalg = importr('kpcalg', robject_translations = d)
    data_trans_pd=pd.DataFrame(data_trans)
    pandas2ri.activate()
    df = robjects.conversion.py2rpy(data_trans_pd)
    base=importr("base")
    out=kpcalg.kpc(**{'suffStat' : rlc.TaggedList((df,"hsic.perm"),tags=('data','ic.method')),
    'indepTest' : kpcalg.kernelCItest,
    'alpha' : alpha,
    'labels' : data_trans_pd.columns.astype(str),
    'u2pd' : "relaxed",
    'skel.method' : "stable",
    #'fixedGaps' : fixedgaps_r,
    'verbose' : robjects.r('F')})
    dollar = base.__dict__["@"]
    graphobj=dollar(out, "graph")
    graph=importr("graph")
    graphedges=graph.edges(graphobj)#, "matrix")
    import re
    graphedgespy={int(key): np.array(re.findall(r'-?\d+\.?\d*', str(graphedges.rx2(key)))[1:]).astype(int) for key in graphedges.names}
    g=nx.DiGraph(graphedgespy)
    A=nx.adjacency_matrix(g).toarray()
    val_matrix = A
    causaleff = causaleff_ida(g,data_trans)
    return val_matrix, causaleff
def val_pc_gauss(data_trans,alpha):
    from timeawarepc.pcalg_ray import estimate_skeleton, estimate_cpdag, causaleff_ida, ci_test_gauss
    import networkx as nx
    (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                                        data_matrix=data_trans,
                                        alpha=alpha,method='stable')
    g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    A=nx.adjacency_matrix(g).toarray()
    val_matrix = A
    causaleff = causaleff_ida(g,data_trans)
    return val_matrix, causaleff
def causaleff_ida(g,data):
    import numpy as np
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
                    causaleff[x,y] = 0
                else:
                    X=np.asarray(data[:,regressors])
                    Y=np.asarray(data[:,y])
                    X0=np.hstack((np.ones((X.shape[0],1)),X))
                    lm_out = np.linalg.lstsq(X0,Y,rcond=None)[0]
                    causaleff[x,y] = lm_out[regressors.index(x)+1]
    return causaleff
