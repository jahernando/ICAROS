import numpy as np
import random
from hipy import utils as ut
from dataclasses import dataclass

import matplotlib.pyplot as plt

"""
    functions extending plt
"""

def canvas(nx : int, ny : int = 2, height : float = 6., width : float = 4.) -> callable:
    """ create a canvas with nx and ny subplots,
    return a function to move to next subplot in the canvas
    """
    icd = 0
    nx = int(ns/ny + ns%ny)
    def subplot(i : int = 1):
        icd += i
        assert icd <= nx * ny
        plt.subplot(nx, ny, icd)
    return subplot


def hist(x : np.array, stats : bool = True, xylabels : tuple = None, **kargs):
    """ decorate hist to label the statistic and to write the x-y labels
    """

    if (not ('histtype' in kargs.keys())):
        kargs['histtype'] = 'step'

    if (stats):
        range   = kargs['range']   if 'range'   in kargs.key else None
        formate = kargs['formate'] if 'formate' in kargs.key else '6.2f'
        ss = ut.str_stats(x, range = range, formate = formate)

        #range = kargs['range'] if 'range' in kargs.keys() else (np.min(x), np.max(x))
        #sel = (x >= range[0]) & (x <= range[1])
        #mean = np.mean(x)
        #std  = np.std(x)
        #sentries  =  f'entries = {len(x)}'
        #smean     =  r'mean = {:7.2f}'.format(mean)
        #sstd      =  r'std  = {:7.2f}'.format(std)
        #sstat     =  f'{sentries}\n{smean}\n{sstd}'
        if ('label' in kargs.keys()):
            kargs['label'] += '\n' + sstat
        else:
            kargs['label'] = sstat

    c = plt.hist(x, bins, **kargs)

    if (xylabels is not None):
        xlabel, ylabel = xylabels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    if ('label' in kargs.keys()):
        plt.legend()

    return c
