# % Statistical test for kernel conditional independence of X and Y given Z with 
# % incomplete Cholesky factorization for low rank approximation of Gram matrices
# %
# % Arguments:
# % X          n x p vector of data points
# % Y          n x m vector of data points
# % Z          n x r vector of data points
# % alpha      significance level
# % reps       number of replicates for the bootstrap test
# %
# % Output:
# % sig        boolean indicator of whether the test was significant for the given alpha
# % p          resulting p-value
# %
# % Adapted from  Robert Tillman's Matlab Code  [rtillman@cmu.edu]
#import numpy as np
# from hsic_medbw import *
# from hsic_inchol import * 
# from hsic_condIC import *
# from hsic_pickK import *

#from sklearn.cluster import KMeans
def hsic_CI(X,Y,Z=None,alpha=0.05,reps=50):
    from arch import bootstrap
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    import rpy2.rlike.container as rlc
    from rpy2.robjects import pandas2ri, numpy2ri
    import pandas as pd 
    import numpy as np
    import networkx as nx
    d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
    kpcalg = importr('kpcalg', robject_translations = d)
    #data_trans_pd=pd.DataFrame(data)
    pandas2ri.activate()
    numpy2ri.activate()
    base=importr("base")
    if Z is None:
        # Xdf = robjects.conversion.py2rpy(X)
        # Ydf = robjects.conversion.py2rpy(Y)
        out = kpcalg.hsic_gamma(X,Y)
        testStat=out.rx2('statistic')[0]
        nullapprox = np.zeros(reps)
        
        band = bootstrap.optimal_block_length(Y)
        #print("band "+str(np.max(band.iloc[:,0])))
        bs = bootstrap.StationaryBootstrap(np.max(band.iloc[:,0]),Y)
        #bs = bootstrap.StationaryBootstrap(20,Y)
        Y1all = np.asarray([x[0][0] for x in bs.bootstrap(reps)])

        for idx in range(reps):
            Y1 = Y1all[idx,]
            out1 = kpcalg.hsic_gamma(X,Y1)
            nullapprox[idx]=out1.rx2('statistic')[0]
        # get p-value from empirical cdf
        p = np.sum(nullapprox>2*testStat)/reps#length(find(nullapprox>=testStat))/reps;
        #% determine significance
        sig=(p<=alpha)
    else:
        out = kpcalg.hsic_clust(X,Y,Z,1,1)
        testStat=out.rx2('statistic')[0]
        nullapprox = np.zeros(reps)

        band = bootstrap.optimal_block_length(Y)
        #print("band "+str(np.max(band.iloc[:,0])))
        bs = bootstrap.StationaryBootstrap(np.max(band.iloc[:,0]),Y)
        Y1all = np.asarray([x[0][0] for x in bs.bootstrap(reps)])

        for idx in range(reps):
            Y1 = Y1all[idx,]
            out1 = kpcalg.hsic_clust(X,Y1,Z,1,1)
            nullapprox[idx]=out1.rx2('statistic')[0]
        # get p-value from empirical cdf
        p = np.sum(nullapprox>2*testStat)/reps#length(find(nullapprox>=testStat))/reps;
        #% determine significance
        sig=(p<=alpha)
    return p
# def hsiccondTestIC(X,Y,Z=None,alpha=0.05,reps=50):#function [sig,p,testStat] = 
#     n = X.shape[0]
#     if Z is not None:
#         if n!=Y.shape[0] or n!=Z.shape[0]:
#             raise ValueError('X, Y, and Z must have the same number of data points')
#     else:
#         if n!=Y.shape[0]:
#                 raise ValueError('X and Y must have the same number of data points')
#     if alpha<0 or alpha>1:
#         raise ValueError('alpha must be between 0 and 1');
#     if reps<=0 or reps!=int(reps):
#         raise ValueError('number of reps must be a positive integer');

#     #smoothing constant for conditional cross covariance operator
#     epsilon=1e-4
#     #threshold for eigenvalues to consider in low rank Gram matrix approximations
#     tol = 1e-4
#     if Z is None:
#         #set kernel size to median distance between points
#         maxpoints = 1000
#         sigx = hsic_medbw(X, maxpoints)
#         sigy = hsic_medbw(Y, maxpoints)

#         #low rank approximation of Gram matrices using incomplete Cholesky factorization
#         [K, Pk] = hsic_inchol(X,sigx,tol)
#         [L, Pl] = hsic_inchol(Y,sigy,tol)

#         #% center Gram matrices factoring in permutations made during low rank approximation
#         Kc = K[Pk,:] - np.tile((np.sum(K)/n),(n,1))
#         Lc = L[Pl,:] - np.tile((np.sum(L)/n),(n,1))
#         H = np.identity(n) - (1/n) * np.ones((n, n))
#         testStat = (1/n)**2 * np.trace(Kc @ H @ Lc @ H)
#         nullapprox = np.zeros(reps)

#         band = bootstrap.optimal_block_length(Y)
#         bs = bootstrap.StationaryBootstrap(np.median(band.iloc[:,0]),Y)
#         Y1all = np.asarray([x[0][0] for x in bs.bootstrap(reps)])

#         for idx in range(reps):
#             Y1 = Y1all[idx,]
#             sigy1 = hsic_medbw(Y1, maxpoints)
#             [L1, Pl1] = hsic_inchol(Y1,sigy1,tol)
#             Lc1 = L1[Pl1,:] - np.tile((np.sum(L1)/n),(n,1))

#             nullapprox[idx]=(1/n)**2 * np.trace(Kc @ H @ Lc1 @ H)


#         # get p-value from empirical cdf
#         p = np.sum(nullapprox>2*testStat)/reps#length(find(nullapprox>=testStat))/reps;

#         #% determine significance
#         sig=(p<=alpha)
#     else:
#         #augment X and Y for conditional test
#         X = np.hstack((X,Z))
#         Y = np.hstack((Y,Z))

#         #set kernel size to median distance between points
#         maxpoints = 1000
#         sigx = hsic_medbw(X, maxpoints)
#         sigy = hsic_medbw(Y, maxpoints)
#         sigz = hsic_medbw(Z, maxpoints)

#         #low rank approximation of Gram matrices using incomplete Cholesky factorization
#         [K, Pk] = hsic_inchol(X,sigx,tol)
#         [L, Pl] = hsic_inchol(Y,sigy,tol)
#         [M, Pm] = hsic_inchol(Z,sigz,tol)

#         #% center Gram matrices factoring in permutations made during low rank approximation
#         Kc = K[Pk,:] - np.tile((np.sum(K)/n),(n,1))
#         Lc = L[Pl,:] - np.tile((np.sum(L)/n),(n,1))
#         Mc = M[Pm,:] - np.tile((np.sum(M)/n),(n,1))

#         # % compute the U-statistic
#         # %pairs = nchoosek(1:n,2);
#         # %bz = n*(n-1)/sum(rbf(Z(pairs(:,1)),Z(pairs(:,2)),sigz).^2);

#         #% compute HSIC dependence value
#         testStat = hsic_condIC(Kc,Lc,Mc,epsilon)

#         #% first cluster Z;
#         nc = hsic_pickK(Z)
#         kmeans = KMeans(n_clusters=nc,max_iter=1000).fit(Z)  
#         clusters = kmeans.cluster_centers_
#         #clusters = kmeans(Z,nc,'EmptyAction','drop','MaxIter',1000,'Display','off');
#         # %[centers,clusters,datapoints] = MeanShiftCluster(Z,sigz,false);
#         # %nc = length(centers);

#         # % simulate null distribution and bootstrap test
#         nullapprox = np.zeros(reps)
#         # % permute within clusters
#         #Plnew = range(n)
#         bs={}
#         for j in range(nc):
#             maskj = clusters==j
#             band = bootstrap.optimal_block_length(Y[maskj,:])
#             bs[j] = bootstrap.StationaryBootstrap(np.median(band.iloc[:,0]),Y[maskj,:])
#         Y1all = np.zeros((reps,Y.shape[0],Y.shape[1]))
#         for j in range(nc):
#             Y1all[:,clusters==j,:] = np.asarray([x[0][0] for x in bs[j].bootstrap(reps)])#np.asarray(list(bs[j].bootstrap(reps)[0][0]))
#     #        for Yiter in bs.bootstrap(reps):
#     #            Y1 = Yiter[0][0]
#         for idx in range(reps):
#             Y1 = Y1all[idx,:,:]
#             sigy1 = hsic_medbw(Y1, maxpoints)
#             [L1, Pl1] = hsic_inchol(Y1,sigy1,tol)
#             Lc1 = L1[Pl1,:] - np.tile((np.sum(L1)/n),(n,1))
#             #pj = indj(randperm(length(indj)))
#             #Plnew(indj) = Plnew(pj);
#             nullapprox[idx]=hsic_condIC(Kc,Lc1,Mc,epsilon)
#     #            idx=idx+1

#         # get p-value from empirical cdf
#         p = np.sum(nullapprox>2*testStat)/reps#length(find(nullapprox>=testStat))/reps;

#         #% determine significance
#         sig=(p<=alpha)
#     return sig,p,testStat