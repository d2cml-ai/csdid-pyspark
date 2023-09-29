# 



from pyspark.sql import  SparkSession, Row, functions as F
from pyspark.sql.functions import \
	lit, col, when, expr, countDistinct,\
	monotonically_increasing_id, desc

from pyspark.sql.window import Window
import numpy as np
from scipy.stats import mstats, norm
from joblib import Parallel, delayed
import pandas as pd

# utils
def multiplier_bootstrap(inf_func, biters): # This function comes from c++
    n, K = inf_func.shape
    biters = int(biters)
    innerMat = np.zeros((n, K))
    Ub = np.zeros(n)
    outMat = np.zeros((biters,K))

    for b in range(biters):
        # draw Rademechar weights
        # Ub = ( np.ones(n) - 2 * np.round(np.random.rand(n)) )[:, np.newaxis]
        Ub = np.random.choice([1, -1], size=(n, 1))
        innerMat = inf_func * Ub
        outMat[b] = np.mean(innerMat, axis=0)

    return outMat

def TorF(cond, use_isTRUE=False):

    if not isinstance(cond, np.ndarray) or cond.dtype != bool:
        raise ValueError("cond should be a logical vector")
    if use_isTRUE:
        cond = np.array([x is True for x in cond])
    else:
        cond[np.isnan(cond)] = False
    return cond

def run_multiplier_bootstrap(inf_func, biters, pl=False, cores=1):
    ngroups = int(np.ceil(biters / cores))
    chunks = [ngroups] * cores
    chunks[0] += biters - sum(chunks)

    n = inf_func.shape[0]

    def parallel_function(biters):
        return multiplier_bootstrap(inf_func, biters)

    if n > 2500 and pl and cores > 1:
        results = Parallel(n_jobs=cores)(
            delayed(parallel_function)(biters) for biters in chunks
        )
        results = np.vstack(results)
    else:
        results = multiplier_bootstrap(inf_func, biters)

    return results

def mboot(inf_func, DIDparams, pl=False, cores=1):
    # Setup needed variables
    data            = DIDparams['data'] 
    idname          = DIDparams['idname']
    clustervars     = DIDparams['clustervar']
    biters          = DIDparams['biters']
    tname           = DIDparams['tname']
    tlist = DIDparams['tlist']
    alp             = DIDparams['alp']
    panel           = DIDparams['panel']

    # Get n observations (for clustering below)
    if panel:
        data = data.filter(col(tname) == tlist[0])


    # Convert inf_func to matrix
    inf_func = np.asarray(inf_func)

    # Set correct number of units
    n = inf_func.shape[0]

    # Drop idname if it is in clustervars
    if clustervars is not None and idname in clustervars:
        clustervars.remove(idname)

    if clustervars is not None:
        if isinstance(clustervars, list) and isinstance(clustervars[0], str):
            raise ValueError("clustervars need to be the name of the clustering variable.")

    # We can only handle up to 2-way clustering
    if clustervars is not None and len(clustervars) > 1:
        raise ValueError("Can't handle that many cluster variables")

    if clustervars is not None:
        # Check that cluster variable does not vary over time within unit
        clust_tv = data.groupby(idname)[clustervars[0]].nunique() == 1
        if not clust_tv.all():
            raise ValueError("Can't handle time-varying cluster variables")
    # clustervars='year'    
    # Multiplier bootstrap
    n_clusters = n
    if not clustervars:
        bres = np.sqrt(n) * run_multiplier_bootstrap(inf_func, biters, pl, cores)
    else:
        n_clusters = len(data[clustervars].drop_duplicates())
        cluster = data[[idname, clustervars]].drop_duplicates().values[:, 1]
        cluster_n = data.groupby(cluster).size().values
        cluster_mean_if = pd.DataFrame(inf_func).groupby(cluster).sum().values / cluster_n
        bres = np.sqrt(n_clusters) * run_multiplier_bootstrap(cluster_mean_if, biters, pl, cores)

    # Handle vector and matrix case differently to get nxk matrix
    if isinstance(bres, np.ndarray) and bres.ndim == 1:
        bres = np.expand_dims(bres, axis=0)
    elif isinstance(bres, np.ndarray) and bres.ndim > 2:
        bres = bres.transpose()

    # Non-degenerate dimensions
    ndg_dim = np.logical_and(~np.isnan(np.sum(bres, axis=0)), np.sum(bres ** 2, axis=0) > np.sqrt(np.finfo(float).eps) * 10)
    bres = bres[:, ndg_dim]

    # Bootstrap variance matrix (this matrix can be defective because of degenerate cases)
    V = np.cov(bres, rowvar=False)

    # Bootstrap standard error
    quantile_75 = np.quantile(bres, 0.75, axis=0, method="inverted_cdf")
    quantile_25 = np.quantile(bres, 0.25, axis=0, method="inverted_cdf")
    qnorm_75 = norm.ppf(0.75)
    qnorm_25 = norm.ppf(0.25)   
    bSigma = (quantile_75 - quantile_25) / (qnorm_75 - qnorm_25)
        
    # Critical value for uniform confidence band
    bT = np.max(np.abs(bres / bSigma), axis=1)
    bT = bT[np.isfinite(bT)]
    crit_val = np.quantile(bT, 1 - alp, method="inverted_cdf")
    
    # Standard error
    se = np.full(ndg_dim.shape, np.nan)
    se[ndg_dim] = bSigma / np.sqrt(n_clusters)

    return {'bres': bres, 'V': V, 'se': se, 'crit_val': crit_val}
