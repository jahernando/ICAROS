import numpy             as np
#import pandas            as pd
#import tables            as tb
#import matplotlib.pyplot as plt
#from scipy               import optimize
#import invisible_cities.core.fit_functions as fitf


#--- general utilies

def remove_nan(vals : np.array) -> np.array:
    """ returns the np.array without nan
    """
    return vals[~np.isnan(vals)]


def in_range(vals : np.array, range : tuple = None, upper_limit_in = False) -> np.array(bool):
    """ returns a np.array(bool) with the elements of val that are in range
    inputs:
        vals : np.array
        range: tuple(x0, x1)
    returns
        np.array(bool) where True/False indicates if the elements of vals is in rage
    """
    if (range is None): return vals >= np.min(vals)
    sel1 = (vals >= range[0])
    sel2 = (vals <= range[1]) if upper_limit_in else (vals < range[1])
    return sel1 & sel2

def centers(xs : np.array) -> np.array:
    """ returns the center between the participn
    inputs:
        xs: np.array
    returns:
        np.array with the centers of xs (dimension len(xs)-1)
    """
    return 0.5* ( xs[1: ] + xs[: -1])


def arscale(x, scale = 1.):
    """ return an arrey between [0., 1.]
    inputs:
        x    : np.array,
        scale: float (1.)
    returns:
        np.arry with scaled balues, [0, scale]
    """
    xmin, xmax = np.min(x), np.max(x)
    rx = scale * (x - xmin)/(xmax - xmin)
    return rx

def arstep(x, step):
    """ returns an array with bins of step size from x-min to x-max (inclusive)
    inputs:
        x    : np.array
        step : float, step-size
    returns:
        np.array with the bins with step size
    """
    return np.arange(np.min(x), np.max(x) + step, step)


def stats(vals : np.array, range : tuple = None):
    vals = remove_nan(vals)
    sel  = in_range(vals, range)
    vv = vals[sel]
    mean, std, evts, oevts = np.mean(vv), np.std(vv), len(vv), len(vals) - len(vv)
    return evts, mean, std, oevts


def str_stats(vals, range = None, formate = '6.2f'):
    evts, mean, std, ovts = stats(vals, range)
    s  = 'entries '+str(evts)+'\n'
    s += (('mean {0:'+formate+'}').format(mean))+'\n'
    s += (('std  {0:'+formate+'}').format(std))
    return s
