import ray
def val_pc_gauss(data_trans,alpha):
    from timeawarepc.pcalg_ray_1 import estimate_skeleton, estimate_cpdag, causaleff_ida, ci_test_gauss
    import networkx as nx
    (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                                        data_matrix=data_trans,
                                        alpha=alpha,method='stable')
    g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    A=nx.adjacency_matrix(g).toarray()
    val_matrix = A
    causaleff = causaleff_ida(g,data_trans)
    return val_matrix, causaleff
def val_pc_hsic_btp(data_trans,alpha):
    from timeawarepc.pcalg_ray_1 import estimate_skeleton, estimate_cpdag, hsic_condind, causaleff_ida, ci_test_gauss
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
    from timeawarepc.pcalg_ray_1 import estimate_skeleton, estimate_cpdag, causaleff_ida, ci_test_gauss
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

def val_pc_gauss_btp(data_trans,alpha):
    from timeawarepc.pcalg_ray_1 import estimate_skeleton, estimate_cpdag, causaleff_ida, ci_test_gauss, ci_test_gauss_btp
    import networkx as nx
    (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss_btp,
                                        data_matrix=data_trans,
                                        alpha=alpha,method='stable')
    g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    A=nx.adjacency_matrix(g).toarray()
    val_matrix = A
    causaleff = causaleff_ida(g,data_trans)
    return val_matrix, causaleff
def val_pc(data_trans):
    alpha=0.1
    d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
    kpcalg = importr('kpcalg', robject_translations = d)
    data_trans_pd = pd.DataFrame(data_trans)
    #dat=robjects.r.data('data_trans_pd')
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
    causaleff = causaleff_ida(g,data_trans)
    #G,causaleffin=return_finaledges(g,causaleff,lag,smspikes.shape[0])
    A=nx.adjacency_matrix(g).toarray()
    val_matrix = A
    return val_matrix, causaleff
# %%
def val_neuropc_lin(data,lag=1,subsampsize=50,n_iter=1,alpha=0.3,thresh=0.25,v2=False):
    #random.seed(111)
    import time
    import numpy as np
    import pandas as pd
    from timeawarepc.pcalg_ray_1 import data_transformed, data_transformed_overlapping, causaleff_ida, return_finaledges, estimate_cpdag, estimate_skeleton,ci_test_gauss, orient
    #import rpy2.robjects as robjects
    #from rpy2.robjects.packages import importr
    #import rpy2.rlike.container as rlc
    #from rpy2.robjects import pandas2ri
    import random
    import networkx as nx
    C_iter=[]
    C_cf_iter=[]
    C_cf2_iter=[]
    #data_trans = data_transformed(smspikes[:,(inneriter*500):((inneriter+1)*500)].T, lag)
    #window=20
    start_time = time.time()
    #data_trans = data_transformed_v2(data, lag,window)
    if v2 is False:
        data_trans = data_transformed(data, lag)
    else:
        data_trans = data_transformed_overlapping(data, lag)
    print("Data transformed in "+str(time.time()-start_time))
    #d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
    #kpcalg = importr('kpcalg', robject_translations = d)
    for inneriter in range(n_iter):
        start_btrstrp = time.time()
        print("Starting bootstrap "+str(inneriter))
        #data_trans_pd=pd.DataFrame(data_trans[random.sample(range(data_trans.shape[0]),k=subsampsize),:])
        n=data_trans.shape[0]
        r_idx = random.sample(range(n-subsampsize),1)[0]
        data_trans_pd=data_trans[r_idx:(r_idx+subsampsize),:]
        #data_trans_pd=pd.DataFrame(data_trans)
        #dat=robjects.r.data('data_trans_pd')
        p=data_trans_pd.shape[1]
        m=data.shape[1]
        (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                                            data_matrix=data_trans_pd,
                                            alpha=alpha,method='stable')
        g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
        g=orient(g,lag,data.shape[1])
        causaleff = causaleff_ida(g,data_trans)
        G,causaleffin, causaleffin2=return_finaledges(g,causaleff,lag,data.shape[1])
        #A=nx.adjacency_matrix(G)
        A_rr=G#A.toarray()
        C_iter.append(A_rr)
        C_cf_iter.append(causaleffin)
        C_cf2_iter.append(causaleffin2)
        print("Done in "+str(time.time()-start_btrstrp))
    val_out=(np.mean(np.asarray(C_iter),axis=0)>=thresh).astype(int)
    ce_out=np.nanmean(np.where(np.asarray(C_cf_iter)!=0,np.asarray(C_cf_iter),np.nan),axis=0)#np.apply_along_axis(avg,0,np.asarray(C_cf_iter))
    ce_out2=np.nanmean(np.where(np.asarray(C_cf2_iter)!=0,np.asarray(C_cf2_iter),np.nan),axis=0)#np.apply_along_axis(avg,0,np.asarray(C_cf_iter2))
    print("CE shape "+str(ce_out.shape)+" "+str(ce_out2.shape))
    #val_out2=val_out
    val_out[np.abs(ce_out) <= np.nanmax(np.abs(ce_out))/10]=0
    return val_out,ce_out, ce_out2
def val_tpc_sno(data,lag=1,alpha=0.1):
    #random.seed(111)
    import time
    import numpy as np
    import pandas as pd
    from timeawarepc.pcalg_ray_1 import data_transformed_overlapping,data_transformed,return_finaledges_v2 ,estimate_cpdag, estimate_skeleton, ci_test_gauss,causaleff_ida, return_finaledges
    import random
    import networkx as nx
    start_time = time.time()
    data_trans = data_transformed(data, lag-1)
    (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                                        data_matrix=data_trans,
                                        alpha=alpha,method='stable')
    g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    causaleff = causaleff_ida(g,data_trans)
    G,causaleffin, causaleffin2=return_finaledges_v2(g,causaleff,lag,data.shape[1])
#    g=orient(g,lag,data.shape[1])
    val_out=G#nx.adjacency_matrix(G).toarray()
    print(time.time()-start_time)
    return val_out, causaleffin
def val_tpc_sno_hsic_btp(data,lag=1,alpha=0.1):
    #random.seed(111)
    import time
    import numpy as np
    import pandas as pd
    from timeawarepc.pcalg_ray_1 import data_transformed_overlapping,hsic_condind,data_transformed,return_finaledges_v2 ,estimate_cpdag, estimate_skeleton, ci_test_gauss,causaleff_ida, return_finaledges
    import random
    import networkx as nx
    start_time = time.time()
    data_trans = data_transformed(data, lag-1)
    (g, sep_set) = estimate_skeleton(indep_test_func=hsic_condind,
                                        data_matrix=data_trans,
                                        alpha=alpha,method='stable')
    g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    causaleff = causaleff_ida(g,data_trans)
    G,causaleffin, causaleffin2=return_finaledges_v2(g,causaleff,lag,data.shape[1])
#    g=orient(g,lag,data.shape[1])
    val_out=G#nx.adjacency_matrix(G).toarray()
    print(time.time()-start_time)
    return val_out, causaleffin
def val_tpc_sno_hsic(data,lag=1,alpha=0.1):
    #random.seed(111)
    import time
    import numpy as np
    import pandas as pd
    from timeawarepc.pcalg_ray_1 import data_transformed_overlapping,data_transformed,return_finaledges_v2 ,estimate_cpdag, estimate_skeleton, ci_test_gauss,causaleff_ida, return_finaledges
    import random
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    import rpy2.rlike.container as rlc
    from rpy2.robjects import pandas2ri
    import networkx as nx
    start_time = time.time()
    data_trans = data_transformed(data, lag-1)
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
    causaleff = causaleff_ida(g,data_trans)
    G,causaleffin, causaleffin2=return_finaledges_v2(g,causaleff,lag,data.shape[1])
#    g=orient(g,lag,data.shape[1])
    val_out=G#nx.adjacency_matrix(G).toarray()
    print(time.time()-start_time)
    return val_out, causaleffin
def val_tpc_sno_btp(data,lag=1,alpha=0.1):
    #random.seed(111)
    import time
    import numpy as np
    import pandas as pd
    from timeawarepc.pcalg_ray_1 import data_transformed_overlapping,data_transformed,return_finaledges_v2 ,estimate_cpdag, estimate_skeleton, ci_test_gauss,causaleff_ida, return_finaledges, ci_test_gauss_btp
    import random
    import networkx as nx
    start_time = time.time()
    data_trans = data_transformed(data, lag-1)
    (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss_btp,
                                        data_matrix=data_trans,
                                        alpha=alpha,method='stable')
    g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    causaleff = causaleff_ida(g,data_trans)
    G,causaleffin, causaleffin2=return_finaledges_v2(g,causaleff,lag,data.shape[1])
#    g=orient(g,lag,data.shape[1])
    val_out=G#nx.adjacency_matrix(G).toarray()
    print(time.time()-start_time)
    return val_out, causaleffin
def val_tpc_sno_btstrp(data,lag=1,alpha=0.1):
    #random.seed(111)
    import time
    import numpy as np
    import pandas as pd
    from timeawarepc.pcalg_ray_1 import data_transformed_btstrp,data_transformed,return_finaledges_v2 ,estimate_cpdag, estimate_skeleton, ci_test_gauss,causaleff_ida, return_finaledges
    import random
    import networkx as nx
    start_time = time.time()
    data_trans = data_transformed_btstrp(data)

    (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                                        data_matrix=data_trans,
                                        alpha=alpha,method='stable')
    g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    causaleff = causaleff_ida(g,data_trans)
    G,causaleffin, causaleffin2=return_finaledges_v2(g,causaleff,lag,data.shape[1])
#    g=orient(g,lag,data.shape[1])
    val_out=G#nx.adjacency_matrix(G).toarray()
    print(time.time()-start_time)
    return val_out, causaleffin
def val_tpc_s(data,lag=1,alpha=0.1):
    #random.seed(111)
    import time
    import numpy as np
    import pandas as pd
    from timeawarepc.pcalg_ray_1 import data_transformed_overlapping,data_transformed,return_finaledges_v2 ,estimate_cpdag, estimate_skeleton, ci_test_gauss,causaleff_ida, return_finaledges
    import random
    import networkx as nx
    start_time = time.time()
    data_trans = data_transformed_overlapping(data, lag)
    (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                                        data_matrix=data_trans,
                                        alpha=alpha,method='stable')
    g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    causaleff = causaleff_ida(g,data_trans)
    G,causaleffin, causaleffin2=return_finaledges_v2(g,causaleff,lag,data.shape[1])
#    g=orient(g,lag,data.shape[1])
    val_out=G#nx.adjacency_matrix(G).toarray()
    print(time.time()-start_time)
    return val_out, causaleffin
def val_tpc_ns(data,lag=1,subsampsize=50,n_iter=50,alpha=0.3,thresh=0.25):
    import time
    import numpy as np
    import pandas as pd
    from timeawarepc.pcalg_ray_1 import data_transformed,estimate_skeleton,return_finaledges_v2,estimate_cpdag,ci_test_gauss,data_transformed_overlapping, causaleff_ida, return_finaledges, orient
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    import rpy2.rlike.container as rlc
    from rpy2.robjects import pandas2ri
    import random
    import networkx as nx
    C_iter=[]
    C_cf_iter=[]
    C_cf2_iter=[]
    #data_trans = data_transformed(smspikes[:,(inneriter*500):((inneriter+1)*500)].T, lag)
    #window=20
    start_time = time.time()
    #data_trans = data_transformed_v2(data, lag,window)
    data_trans = data_transformed(data, lag-1)
    print("Data transformed in "+str(time.time()-start_time))
    d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
    for inneriter in range(n_iter):
        start_btrstrp = time.time()
        print("Starting bootstrap "+str(inneriter))
        #data_trans_pd=pd.DataFrame(data_trans[random.sample(range(data_trans.shape[0]),k=subsampsize),:])
        n=data_trans.shape[0]
        r_idx = random.sample(range(n-subsampsize),1)[0]
        data_trans_pd=data_trans[r_idx:(r_idx+subsampsize),:]
        #data_trans_pd=pd.DataFrame(data_trans)
        #dat=robjects.r.data('data_trans_pd')
        p=data_trans_pd.shape[1]
        m=data.shape[1]
        (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                                            data_matrix=data_trans_pd,
                                            alpha=alpha,method='stable')
        g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
        #g=orient(g,lag,data.shape[1])
        causaleff = causaleff_ida(g,data_trans)
        G,causaleffin, causaleffin2=return_finaledges_v2(g,causaleff,lag,data.shape[1])
        #A=nx.adjacency_matrix(G)
        A_rr=G#A.toarray()
        C_iter.append(A_rr)
        C_cf_iter.append(causaleffin)
        C_cf2_iter.append(causaleffin2)
        print("Done in "+str(time.time()-start_btrstrp))
    val_out=(np.mean(np.asarray(C_iter),axis=0)>=thresh).astype(int)
    ce_out=np.nanmean(np.where(np.asarray(C_cf_iter)!=0,np.asarray(C_cf_iter),np.nan),axis=0)#np.apply_along_axis(avg,0,np.asarray(C_cf_iter))
    ce_out2=np.nanmean(np.where(np.asarray(C_cf2_iter)!=0,np.asarray(C_cf2_iter),np.nan),axis=0)#np.apply_along_axis(avg,0,np.asarray(C_cf_iter2))
    print("CE shape "+str(ce_out.shape)+" "+str(ce_out2.shape))
    #val_out2=val_out
    val_out[np.abs(ce_out) <= np.nanmax(np.abs(ce_out))/10]=0
    return val_out,ce_out,ce_out2

def iter_tpc_ns_hsic_btp(data_trans,subsampsize,alpha,lag,m):
    import random
    from timeawarepc.pcalg_ray_1 import hsic_condind,estimate_skeleton,return_finaledges_v2,estimate_cpdag, causaleff_ida
    #data_trans_pd=pd.DataFrame(data_trans[random.sample(range(data_trans.shape[0]),k=subsampsize),:])
    n=data_trans.shape[0]
    r_idx = random.sample(range(n-subsampsize),1)[0]
    data_trans_pd=data_trans[r_idx:(r_idx+subsampsize),:]
    #data_trans_pd=pd.DataFrame(data_trans)
    #dat=robjects.r.data('data_trans_pd')
    p=data_trans_pd.shape[1]
    (g, sep_set) = estimate_skeleton(indep_test_func=hsic_condind,
                                        data_matrix=data_trans_pd,
                                        alpha=alpha,method='stable')
    g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    #g=orient(g,lag,data.shape[1])
    causaleff = causaleff_ida(g,data_trans)
    G,causaleffin, causaleffin2=return_finaledges_v2(g,causaleff,lag,m)
    return (G,causaleffin,causaleffin2)
def val_tpc_ns_hsic_btp(data,lag=1,subsampsize=50,n_iter=50,alpha=0.3,thresh=0.25):
    from timeawarepc.pcalg_ray_1 import data_transformed
    import networkx as nx
    import numpy as np
    C_iter=[]
    C_cf_iter=[]
    C_cf2_iter=[]
    data_trans = data_transformed(data, lag-1)
    #d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
    data_trans_id = ray.put(data_trans)
    out = ray.get([iter_tpc_ns_hsic_btp.remote(data_trans_id,subsampsize,alpha,lag,data.shape[1]) for _ in range(n_iter)])
    C_iter = list(zip(*out))[0]
    C_cf_iter = list(zip(*out))[1]
    C_cf2_iter = list(zip(*out))[2]
    val_out=(np.mean(np.asarray(C_iter),axis=0)>=thresh).astype(int)
    ce_out=np.nanmean(np.where(np.asarray(C_cf_iter)!=0,np.asarray(C_cf_iter),np.nan),axis=0)#np.apply_along_axis(avg,0,np.asarray(C_cf_iter))
    ce_out2=np.nanmean(np.where(np.asarray(C_cf2_iter)!=0,np.asarray(C_cf2_iter),np.nan),axis=0)#np.apply_along_axis(avg,0,np.asarray(C_cf_iter2))
    #print("CE shape "+str(ce_out.shape)+" "+str(ce_out2.shape))
    #val_out2=val_out
    val_out[np.abs(ce_out) <= np.nanmax(np.abs(ce_out))/10]=0
    return val_out,ce_out,ce_out2
def val_tpc_ns_hsic(data,lag=1,subsampsize=50,n_iter=50,alpha=0.3,thresh=0.25):
    #random.seed(111)
    import time
    import numpy as np
    import pandas as pd
    from timeawarepc.pcalg_ray_1 import data_transformed,estimate_skeleton,return_finaledges_v2,estimate_cpdag,ci_test_gauss,data_transformed_overlapping, causaleff_ida, return_finaledges, orient
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    import rpy2.rlike.container as rlc
    from rpy2.robjects import pandas2ri
    import random
    import networkx as nx
    C_iter=[]
    C_cf_iter=[]
    C_cf2_iter=[]
    #data_trans = data_transformed(smspikes[:,(inneriter*500):((inneriter+1)*500)].T, lag)
    #window=20
    start_time = time.time()
    #data_trans = data_transformed_v2(data, lag,window)
    data_trans = data_transformed(data, lag-1)
    print("Data transformed in "+str(time.time()-start_time))
    d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
    for inneriter in range(n_iter):
        start_btrstrp = time.time()
        print("Starting bootstrap "+str(inneriter))
        #data_trans_pd=pd.DataFrame(data_trans[random.sample(range(data_trans.shape[0]),k=subsampsize),:])
        n=data_trans.shape[0]
        r_idx = random.sample(range(n-subsampsize),1)[0]
        data_trans_pd=data_trans[r_idx:(r_idx+subsampsize),:]
        #data_trans_pd=pd.DataFrame(data_trans)
        #dat=robjects.r.data('data_trans_pd')
        p=data_trans_pd.shape[1]
        m=data.shape[1]
        (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss,
                                            data_matrix=data_trans_pd,
                                            alpha=alpha,method='stable')
        g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
        #g=orient(g,lag,data.shape[1])
        causaleff = causaleff_ida(g,data_trans)
        G,causaleffin, causaleffin2=return_finaledges_v2(g,causaleff,lag,data.shape[1])
        #A=nx.adjacency_matrix(G)
        A_rr=G#A.toarray()
        C_iter.append(A_rr)
        C_cf_iter.append(causaleffin)
        C_cf2_iter.append(causaleffin2)
        print("Done in "+str(time.time()-start_btrstrp))
    val_out=(np.mean(np.asarray(C_iter),axis=0)>=thresh).astype(int)
    ce_out=np.nanmean(np.where(np.asarray(C_cf_iter)!=0,np.asarray(C_cf_iter),np.nan),axis=0)#np.apply_along_axis(avg,0,np.asarray(C_cf_iter))
    ce_out2=np.nanmean(np.where(np.asarray(C_cf2_iter)!=0,np.asarray(C_cf2_iter),np.nan),axis=0)#np.apply_along_axis(avg,0,np.asarray(C_cf_iter2))
    print("CE shape "+str(ce_out.shape)+" "+str(ce_out2.shape))
    #val_out2=val_out
    val_out[np.abs(ce_out) <= np.nanmax(np.abs(ce_out))/10]=0
    return val_out,ce_out,ce_out2

@ray.remote
def iter_tpc_ns_btp(data_trans,subsampsize,alpha,lag,m):
    import random
    import time
    from timeawarepc.pcalg_ray_1 import data_transformed,estimate_skeleton,return_finaledges_v2,estimate_cpdag,ci_test_gauss,data_transformed_overlapping, causaleff_ida, return_finaledges, orient,ci_test_gauss_btp

    start_btrstrp = time.time()
    #print("Starting bootstrap "+str(inneriter))
    #data_trans_pd=pd.DataFrame(data_trans[random.sample(range(data_trans.shape[0]),k=subsampsize),:])
    n=data_trans.shape[0]
    r_idx = random.sample(range(n-subsampsize),1)[0]
    data_trans_pd=data_trans[r_idx:(r_idx+subsampsize),:]
    #data_trans_pd=pd.DataFrame(data_trans)
    #dat=robjects.r.data('data_trans_pd')
    (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_gauss_btp,
                                        data_matrix=data_trans_pd,
                                        alpha=alpha,method='stable')
    g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
    #g=orient(g,lag,data.shape[1])
    causaleff = causaleff_ida(g,data_trans)
    G,causaleffin, causaleffin2=return_finaledges_v2(g,causaleff,lag,m)
    #print("Done in "+str(time.time()-start_btrstrp))
    return (G,causaleffin,causaleffin2)


def val_tpc_ns_btp(data,lag=1,subsampsize=50,n_iter=50,alpha=0.3,thresh=0.25):
    #random.seed(111)
    import numpy as np
    import time
    from timeawarepc.pcalg_ray_1 import data_transformed
    C_iter=[]
    C_cf_iter=[]
    C_cf2_iter=[]
    #data_trans = data_transformed(smspikes[:,(inneriter*500):((inneriter+1)*500)].T, lag)
    #window=20
    start_time = time.time()
    #data_trans = data_transformed_v2(data, lag,window)
    data_trans = data_transformed(data, lag-1)
    #print("Data transformed in "+str(time.time()-start_time))
    d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
    data_trans_id = ray.put(data_trans)
    out = ray.get([iter_tpc_ns_btp.remote(data_trans_id,subsampsize,alpha,lag,data.shape[1]) for _ in range(n_iter)])
    C_iter = list(zip(*out))[0]
    C_cf_iter = list(zip(*out))[1]
    C_cf2_iter = list(zip(*out))[2]
    val_out=(np.mean(np.asarray(C_iter),axis=0)>=thresh).astype(int)
    ce_out=np.nanmean(np.where(np.asarray(C_cf_iter)!=0,np.asarray(C_cf_iter),np.nan),axis=0)#np.apply_along_axis(avg,0,np.asarray(C_cf_iter))
    ce_out2=np.nanmean(np.where(np.asarray(C_cf2_iter)!=0,np.asarray(C_cf2_iter),np.nan),axis=0)#np.apply_along_axis(avg,0,np.asarray(C_cf_iter2))
    print("CE shape "+str(ce_out.shape)+" "+str(ce_out2.shape))
    #val_out2=val_out
    val_out[np.abs(ce_out) <= np.nanmax(np.abs(ce_out))/10]=0
    return val_out,ce_out,ce_out2

def val_neuropc(data,lag=1,subsampsize=50,n_iter=1,alpha=0.3,thresh=0.25,v2=False):
    #random.seed(111)
    import time
    import numpy as np
    import pandas as pd
    from timeawarepc.pcalg_ray_1 import data_transformed,data_transformed_overlapping, causaleff_ida, return_finaledges, orient
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    import rpy2.rlike.container as rlc
    from rpy2.robjects import pandas2ri
    import random
    import networkx as nx
    C_iter=[]
    C_cf_iter=[]
    C_cf2_iter=[]
    #data_trans = data_transformed(smspikes[:,(inneriter*500):((inneriter+1)*500)].T, lag)
    #window=20
    start_time = time.time()
    #data_trans = data_transformed_v2(data, lag,window)
    if v2 is False:
        data_trans = data_transformed(data, lag)
    else:
        data_trans = data_transformed_overlapping(data, lag)
    print("Data transformed in "+str(time.time()-start_time))
    d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
    kpcalg = importr('kpcalg', robject_translations = d)
    for inneriter in range(n_iter):
        start_btrstrp = time.time()
        print("Starting bootstrap "+str(inneriter))
        #data_trans_pd=pd.DataFrame(data_trans[random.sample(range(data_trans.shape[0]),k=subsampsize),:])
        n=data_trans.shape[0]
        r_idx = random.sample(range(n-subsampsize),1)[0]
        data_trans_pd=pd.DataFrame(data_trans[r_idx:(r_idx+subsampsize),:])
        #data_trans_pd=pd.DataFrame(data_trans)
        #dat=robjects.r.data('data_trans_pd')
        p=data_trans_pd.shape[1]
        m=data.shape[1]
        fixedgaps = np.zeros((p,p))>=1#robjects.r.matrix(robjects.r('T'),nrow=p,ncol=p)
        labels=np.arange(0,(lag+1)*m)
        #labelmat=labels.reshape((m,lag+1))
        # for i in range(m):
        #     for j in range(m):
        #         for l in range(lag):
        #             for ld in range(lag):
        #                 if l is not ld:
        #                     fixedgaps[int(labelmat[i,l]),int(labelmat[j,ld])]=True
        pandas2ri.activate()
        df = robjects.conversion.py2rpy(data_trans_pd)
        base=importr("base")
        #fixedgaps_r = base.as_matrix(robjects.conversion.py2rpy(pd.DataFrame(fixedgaps)))
        out=kpcalg.kpc(**{'suffStat' : rlc.TaggedList((df,"hsic.perm"),tags=('data','ic.method')),#robjects.r('list(data=data_trans, ic.method="hsic.perm")'),#list(data=data_trans, ic.method="hsic.perm"),
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
        g=orient(g,lag,data.shape[1])
        causaleff = causaleff_ida(g,data_trans)
        G,causaleffin, causaleffin2=return_finaledges(g,causaleff,lag,data.shape[1])
        #A=nx.adjacency_matrix(G)
        A_rr=G#A.toarray()
        C_iter.append(A_rr)
        C_cf_iter.append(causaleffin)
        C_cf2_iter.append(causaleffin2)
        print("Done in "+str(time.time()-start_btrstrp))
    val_out=(np.mean(np.asarray(C_iter),axis=0)>=thresh).astype(int)
    ce_out=np.nanmean(np.where(np.asarray(C_cf_iter)!=0,np.asarray(C_cf_iter),np.nan),axis=0)#np.apply_along_axis(avg,0,np.asarray(C_cf_iter))
    ce_out2=np.nanmean(np.where(np.asarray(C_cf2_iter)!=0,np.asarray(C_cf2_iter),np.nan),axis=0)#np.apply_along_axis(avg,0,np.asarray(C_cf_iter2))
    print("CE shape "+str(ce_out.shape)+" "+str(ce_out2.shape))
    #val_out2=val_out
    val_out[np.abs(ce_out) <= np.nanmax(np.abs(ce_out))/10]=0
    return val_out,ce_out,ce_out2

    # ce_out=np.mean(np.asarray(C_cf_iter),axis=0)
    # return val_out,ce_out
#%%
def val_neuropc_fin(data,lag=1,maxdeg=1):
    C_iter=[]
    C_cf_iter=[]
    #data_trans = data_transformed(smspikes[:,(inneriter*500):((inneriter+1)*500)].T, lag)
    data_trans = data_transformed_fin(data, lag, maxdeg)
    alpha=0.1
    d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
    kpcalg = importr('kpcalg', robject_translations = d)
    subsampsize=100
    p=data.shape[1]
    for inneriter in tqdm(range(25)):
        data_trans_pd=pd.DataFrame(data_trans[random.sample(range(data_trans.shape[0]),k=subsampsize),:])
        B_rr = np.zeros((p,p))
        causaleff_rr = np.zeros((p,p))
        for idx in range(p):
            data_trans_pd1=data_trans_pd.iloc[:,np.hstack((idx,range(p,2*p)))]
            pandas2ri.activate()
            df = robjects.conversion.py2rpy(data_trans_pd1)
            out=kpcalg.kpc(**{'suffStat' : rlc.TaggedList((df,"hsic.perm"),tags=('data','ic.method')),#robjects.r('list(data=data_trans, ic.method="hsic.perm")'),#list(data=data_trans, ic.method="hsic.perm"),
            'indepTest' : kpcalg.kernelCItest,
            'alpha' : alpha,
            'labels' : data_trans_pd1.columns.astype(str),
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
            causaleff = causaleff_ida_fin(g,data_trans)
            G,causaleffin=return_finaledges_fin(g,causaleff,data.shape[1])
            B_rr[:,idx]=G
            causaleff_rr[:,idx]=causaleffin
        C_iter.append(B_rr)
        C_cf_iter.append(causaleff_rr)
    val_out=(np.mean(np.asarray(C_iter),axis=0)>=0.25).astype(int)
    ce_out=np.mean(np.asarray(C_cf_iter),axis=0)
    return val_out,ce_out
# %%
def val_gc(data,lag,alpha):
    #from scipy.stats import chi2
    import nitime.analysis as nta
    import nitime.timeseries as ts
    import numpy as np
    TR = 1
    thresh = 0
    time_series = ts.TimeSeries(data.T, sampling_interval=TR)
    order=lag
    G = nta.GrangerAnalyzer(time_series, order=order)
    adj_mat = np.zeros((data.shape[1],data.shape[1]))

    adj_mat=np.mean(np.nan_to_num(G.causality_xy[:, :]),-1)+np.mean(np.nan_to_num(G.causality_yx[:, :]),-1).T
    adjmat1=np.mean(np.nan_to_num(G.causality_xy[:, :]),-1)
    adjmat2=np.mean(np.nan_to_num(G.causality_yx[:, :]),-1)
    adj_mat=adjmat1+adjmat2.T
    # G = tsdata_to_autocov(data, 1)
    # AF, SIG = autocov_to_var(G)
    # for i in tqdm(range(data.shape[1])):
    #     for j in range(data.shape[1]):
    #         adj_mat[i,j] = autocov_to_mvgc(G, np.array([j]), np.array([i]))
    # adj_mat = adj_mat.nan_to_num(adj_mat)
    ce_out = adj_mat
    thresh = np.percentile(adj_mat,(1-alpha)*100)
    val_out=(adj_mat > thresh).astype(int)
    
    # val_out=adj_mat #(adj_mat/np.max(np.abs(adj_mat)))#.astype(int)
    #     #adj_mat = np.zeros((data.shape[1],data.shape[1]))
    #     #adj_mat=np.mean(np.nan_to_num(G.causality_xy[:, :]),-1)+np.mean(np.nan_to_num(G.causality_yx[:, :]),-1).T
    # ce_out = adj_mat#is this conditional granger of xy|z?
    #     #val_out = (val_out > thresh).astype(int) + (val_out < - thresh).astype(int)
    # stat=(data.shape[0]-order)*adj_mat
    # d = order*1*1
    # cutoff = chi2.isf(alpha,df=d)
    # val_out = (stat>=cutoff).astype(int)
    return val_out, ce_out
#%%
# def val_gc2(data,lag,alpha):
#     #from scipy.stats import chi2
#     import nitime.analysis as nta
#     import nitime.timeseries as ts
#     import numpy as np
#     from mvgc_py import tsdata_to_autocov, autocov_to_var, autocov_to_mvgc
#     from tqdm import tqdm
#     TR = 1
#     thresh = 0
#     time_series = ts.TimeSeries(data.T, sampling_interval=TR)
#     order=lag
#     G = nta.GrangerAnalyzer(time_series, order=order)
#     adj_mat = np.zeros((data.shape[1],data.shape[1]))

#     #adj_mat=np.mean(np.nan_to_num(G.causality_xy[:, :]),-1)+np.mean(np.nan_to_num(G.causality_yx[:, :]),-1).T
#     #adjmat1=np.mean(np.nan_to_num(G.causality_xy[:, :]),-1)
#     #adjmat2=np.mean(np.nan_to_num(G.causality_yx[:, :]),-1)
#     #adj_mat=adjmat1+adjmat2.T
#     G = tsdata_to_autocov(data, 1)
#     AF, SIG = autocov_to_var(G)
#     for i in tqdm(range(data.shape[1])):
#         for j in range(data.shape[1]):
#             adj_mat[i,j] = autocov_to_mvgc(G, np.array([j]), np.array([i]))
#     adj_mat = np.nan_to_num(adj_mat)
#     ce_out = adj_mat
#     thresh = np.percentile(adj_mat,(1-alpha)*100)
#     val_out=(adj_mat > thresh).astype(int)
    
#     # val_out=adj_mat #(adj_mat/np.max(np.abs(adj_mat)))#.astype(int)
#     #     #adj_mat = np.zeros((data.shape[1],data.shape[1]))
#     #     #adj_mat=np.mean(np.nan_to_num(G.causality_xy[:, :]),-1)+np.mean(np.nan_to_num(G.causality_yx[:, :]),-1).T
#     # ce_out = adj_mat#is this conditional granger of xy|z?
#     #     #val_out = (val_out > thresh).astype(int) + (val_out < - thresh).astype(int)
#     # stat=(data.shape[0]-order)*adj_mat
#     # d = order*1*1
#     # cutoff = chi2.isf(alpha,df=d)
#     # val_out = (stat>=cutoff).astype(int)
#     return val_out, ce_out
# %%
def val_gc3(data,lag,alpha):
    import matlab.engine
    import numpy as np
    eng = matlab.engine.start_matlab()
    filepath = 'D:\\MATLAB\\toolbox\\mvgc\\'
    eng.addpath(eng.genpath(filepath))
    TR = 1
    #thresh = 0
    #time_series = ts.TimeSeries(data.T, sampling_interval=TR)
    #order=lag
    #nobs = data.shape[0]
    #nvars = data.shape[1]
    #tstat     = 'F'
    regmode   = 'LWR';  # VAR model estimation regression mode ('OLS', 'LWR' or empty for default)
    #icregmode = 'LWR';  # information criteria regression mode ('OLS', 'LWR' or empty for default)

    morder    = 'BIC';  # model order to use ('actual', 'AIC', 'BIC' or supplied numerical value)
    #momax     = 1;     # maximum model order for model order estimation

    TR=1
    acmaxlags = 1000; 
    #fs = TR
    #seed = 0
    #fres = []
    X= matlab.double(data.T.tolist())
    #[AIC,BIC,moAIC,moBIC] = eng.tsdata_to_infocrit(X,momax,icregmode, nargout = 4)
    #morder = moBIC
    morder = lag
    [A,SIG] = eng.tsdata_to_var(X,morder,regmode, nargout = 2)
    [G,info] = eng.var_to_autocov(A,SIG,acmaxlags, nargout = 2)
    F = eng.autocov_to_pwcgc(G)
    F = np.asarray(F)
    adj_mat = np.nan_to_num(F)
    ce_out = adj_mat
    thresh = np.percentile(adj_mat,(1-alpha)*100)
    val_out=(adj_mat > thresh).astype(int)
    return val_out, ce_out
#%%
def val_neuropc_ges(data,lag=1):
    import networkx as nx
    #from pcalg import *
    C_iter=[]
    C_cf_iter=[]
    data_trans = data_transformed(data, lag)
    #alpha=0.001
    #d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
    pcalg = importr('pcalg')
    methods = importr('methods')
    #kpcalg = importr('kpcalg', robject_translations = d)
    subsampsize=50
    pval_iter=[]
    for inneriter in tqdm(range(25)):
        maxp = 10
        data_trans_pd=pd.DataFrame(data_trans[random.choices(range(data_trans.shape[0]),k=subsampsize),:])
        data_trans_pd.columns = list(range(data_trans.shape[1]))
        # res = ges(data_trans_pd,"local_score_marginal_general",maxp,{'kfold':1,'lambda':10})
        # graphobj = res['G']
        # graphobj_edges=[]
        # edges_str=[]
        # for idx0 in range(len(graphobj.get_graph_edges())):
        #     edges_str.append(str(graphobj.get_graph_edges()[idx0]))
        # for idx1 in range(p):
        #     for idx2 in range(p):
        #         if 'x'+str(idx1)+' --> x'+str(idx2) in edges_str or 'x'+str(idx1)+' <-> x'+str(idx2) in edges_str or 'x'+str(idx1)+' o-> x'+str(idx2) in edges_str:
        #             graphobj_edges.append((idx1,idx2))
        # g=nx.DiGraph()
        # g.add_nodes_from(range(p))
        # g.add_edges_from(graphobj_edges)
        pandas2ri.activate()
        df = robjects.conversion.py2rpy(data_trans_pd)
        score = methods.new("GaussL0penObsScore",df)
        out= pcalg.ges(score)
        graphobj1= out.rx2('essgraph')
        out2=methods.__dict__['as'](graphobj1,"graphNEL")
        base=importr("base")
        dollar = base.__dict__["@"]
        graphedges=dollar(out2, "edgeL")
        import re
        #print('l_'+str(data_trans_pd.columns))
        graphedgespy=[(node1,graphedges[node1][0][node2]-1) for node1 in data_trans_pd.columns for node2 in range(len(graphedges[node1][0]))]
        #print('edges_'+str(graphedgespy))
        g=nx.DiGraph()
        #print('nodes_'+str(data_trans_pd.columns))
        g.add_nodes_from(data_trans_pd.columns)
        #print('gnodes_'+str(g.nodes()))
        g.add_edges_from(graphedgespy)
        #print('gedges_'+str(g.edges()))
        #print('cenodesbefore_'+str(g.nodes()))
        causaleff = causaleff_ida(g,np.asarray(data_trans))
        #pvals = np.zeros((data_trans_pd.shape[1],data_trans_pd.shape[1]))
        G,causaleffin=return_finaledges(g,causaleff,lag,data.shape[1])
        A_rr=G
        C_iter.append(A_rr)
        C_cf_iter.append(causaleffin)
        pval_iter.append(None)
    val_out=np.mean(np.asarray(C_iter),axis=0)#>=0.55).astype(int)
    ce_out=np.mean(np.asarray(C_cf_iter),axis=0)
    pval_out = None #np.mean(np.asarray(pval_iter),axis=0)
    return val_out,ce_out,pval_out

# %%this is same as pc
# def val_neuropc_v2(data_trans):
#     alpha=0.1
#     d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
#     kpcalg = importr('kpcalg', robject_translations = d)
#     data_trans_pd = pd.DataFrame(data_trans)
#     #dat=robjects.r.data('data_trans_pd')
#     pandas2ri.activate()
#     df = robjects.conversion.py2rpy(data_trans_pd)
#     out=kpcalg.kpc(**{'suffStat' : rlc.TaggedList((df,"hsic.perm"),tags=('data','ic.method')),#robjects.r('list(data=data_trans, ic.method="hsic.perm")'),#list(data=data_trans, ic.method="hsic.perm"),
#     'indepTest' : kpcalg.kernelCItest,
#     'alpha' : alpha,
#     'labels' : data_trans_pd.columns.astype(str),
#     'u2pd' : "relaxed",
#     'skel.method' : "stable",
#     'verbose' : robjects.r('F')})
#     base=importr("base")
#     dollar = base.__dict__["@"]
#     graphobj=dollar(out, "graph")
#     graph=importr("graph")
#     graphedges=graph.edges(graphobj)#, "matrix")
#     import re
#     graphedgespy={int(key): np.array(re.findall(r'-?\d+\.?\d*', str(graphedges.rx2(key)))[1:]).astype(int) for key in graphedges.names}
#     g=nx.DiGraph(graphedgespy)
#     causaleff = causaleff_ida(g,data_trans)
#     #G,causaleffin=return_finaledges(g,causaleff,lag,smspikes.shape[0])
#     A=nx.adjacency_matrix(g).toarray()
#     val_matrix = A
#     return val_matrix, causaleff
