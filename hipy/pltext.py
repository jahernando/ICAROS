import numpy as np
import random
import hipy.utils as ut
from dataclasses import dataclass

import matplotlib.pyplot as plt
plt.style.context('seaborn-colorblind')

"""
    functions extending plt
"""


def canvas(ns : int, ny : int = 2, height : float = 5., width : float = 6.) -> callable:
    """ create a canvas with ns subplots and ny-columns,
    return a function to move to next subplot in the canvas
    """
    nx  = int(ns / ny + ns % ny)
    plt.figure(figsize = (width * ny, height * nx))
    def subplot(iplot):
        assert iplot <= nx * ny
        plt.subplot(nx, ny, iplot)
    return subplot


def hist(x : np.array, bins : int, stats : bool = True, xylabels : tuple = None, **kargs):
    """ decorate hist:
    options: stats (bool) True, label the statistics a
             xylabels tuple(str) None; to write the x-y labels
    """

    if (not ('histtype' in kargs.keys())):
        kargs['histtype'] = 'step'

    if (stats):
        range   = kargs['range']   if 'range'   in kargs.keys() else None
        formate = kargs['formate'] if 'formate' in kargs.keys() else '6.2f'
        ss = ut.str_stats(x, range = range, formate = formate)

        if ('label' in kargs.keys()):
            kargs['label'] += '\n' + ss
        else:
            kargs['label'] = ss

    c = plt.hist(x, bins, **kargs)

    if (xylabels is not None):
        xlabel, ylabel = xylabels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    if ('label' in kargs.keys()):
        plt.legend()

    return c

def plt_inspect_df(df, labels = None, bins = 100, ranges = {}, ncolumns = 2):
    """ histogram the variables of a dataframe
    df     : dataframe
    labels : tuple(str) list of variables. if None all the columns of the DF
    bins   : int (100), number of nbins
    ranges : dict, range of the histogram, the key must be the column name
    ncolumns: int (2), number of columns of the canvas
    """
    if (labels is None):
        labels = list(df.columns)
    #print('labels : ', labels)
    subplot = pltext.canvas(len(labels), ncolumns)
    for i, label in enumerate(labels):
        subplot(i + 1)
        values = ut.remove_nan(df[label].values)
        xrange = None if label not in ranges.keys() else ranges[label]
        pltext.hist(values, bins, range = xrange)
        plt.xlabel(label); plt.grid();
