import numpy             as np
#import pandas            as pd
#import tables            as tb
#import matplotlib.pyplot as plt
#from scipy               import optimize
#import invisible_cities.core.fit_functions as fitf


#--- general utilies

def remove_nan(vals : np.array) -> np.array:
     return vals[~np.isnan(vals)]


def in_range(vals : np.array, range : tuple = None) -> np.array(bool):
    if (range is None): return vals >= np.min(vals)
    sel = (vals >= range[0]) & (vals < range[1])
    return sel

def centers(xs):
    return 0.5* ( xs[1: ] + xs[: -1])

def stats(vals : np.array, range : tuple = None):
    vals = remove_nan(vals)
    sel  = in_range(vals, range)
    vv = vals[sel]
    mean, std, evts, oevts = np.mean(vv), np.std(vv), len(vv), len(vals) - len(vv)
    return evts, mean, std, oevts


def str_stats(vals, range = None, formate = '6.2f'):
    evts, mean, std, ovts = stats(vals, range)
    s  = 'events '+str(evts)+'\n'
    s += (('mean {0:'+formate+'}').format(mean))+'\n'
    s += (('std  {0:'+formate+'}').format(std))
    return s
