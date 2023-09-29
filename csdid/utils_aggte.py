from .mboot import * 

import numpy as np
import scipy.stats as stats
import pandas as pd

def wif(keepers, pg, weights_ind, G, group):
    # note: weights are all of the form P(G=g|cond)/sum_cond(P(G=g|cond))
    # this is equal to P(G=g)/sum_cond(P(G=g)) which simplifies things here
    pg = np.array(pg)
    group = np.array(group)
    
    # effect of estimating weights in the numerator
    if1 = np.empty((len(weights_ind), len(keepers)))
    for i, k  in enumerate(keepers):
        numerator = (weights_ind * 1 * TorF(G == group[k])) - pg[k]
        # denominator = sum(np.array(pg)[keepers]) )[:, None]  
        denominator = np.sum(pg[keepers])

        result = numerator[:, None]  / denominator
        if1[:, i] = result.squeeze()
    
    # effect of estimating weights in the denominator
    if2 = np.empty((len(weights_ind), len(keepers)))
    for i, k  in enumerate(keepers):
        numerator = ( weights_ind * 1 * TorF(G == group[k]) ) - pg[k]
        # result = numerator.to_numpy()[:, None]  @ multipler[:, None].T
        if2[:, i] = numerator.squeeze()
    if2 = np.sum(if2, axis=1)    
    multiplier = ( pg[keepers] / sum( pg[keepers] ) ** 2 )   
    if2 = np.outer( if2 , multiplier)

    # if1 = [((weights_ind * 1*TorF(G==group[k])) - pg[k]) / sum(pg[keepers]) for k in keepers]
    # if2 = np.dot(np.array([weights_ind*1*TorF(G==group[k]) - pg[k] for k in keepers]).T, pg[keepers]/(sum(pg[keepers])**2))
    wif_factor = if1 - if2
    return wif_factor

def get_agg_inf_func(att, inffunc, whichones, weights_agg, wif=None):
    # enforce weights are in matrix form
    weights_agg = np.asarray(weights_agg)

    # multiplies influence function times weights and sums to get vector of weighted IF (of length n)
    thisinffunc = np.dot(inffunc[:, whichones], weights_agg)

    # Incorporate influence function of the weights
    if wif is not None:
        thisinffunc = thisinffunc + np.dot(wif, np.array(att[whichones]))
        
    # return influence function
    return thisinffunc


def get_se(thisinffunc, DIDparams=None):
    alpha = 0.05
    bstrap = False
    if DIDparams is not None:
        bstrap = DIDparams['bstrap']
        alpha = DIDparams['alp']
        cband = DIDparams['cband']
        n = len(thisinffunc)

    if bstrap:
        bout = mboot(thisinffunc, DIDparams)
        return bout['se']
    else:
        return np.sqrt(np.mean((thisinffunc)**2) / n)
