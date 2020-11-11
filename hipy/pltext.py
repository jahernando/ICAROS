import numpy as np
import random
import hipy.utils as ut
import hipy.hfit  as hfitm
from dataclasses import dataclass

import matplotlib.pyplot as plt
from cycler import cycler


"""
    functions extending plt
"""

def style():
    """ mathplot style
    """

    plt.rcParams['axes.prop_cycle'] = cycler(color='kbgrcmy')
    plt.style.context('seaborn-colorblind')
    return


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


def hist(x : np.array, bins : int, stats : bool = True, xylabels : tuple = None,
        grid = True, ylog = False, **kargs):
    """ decorate hist:
    options:
    stats (bool) True, label the statistics a
    xylabels tuple(str) None; to write the x-y labels
    grid  (bool) True, set the grid option
    ylog  (bool) False, set the y-escale to log
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

        if (type(xylabels) == str):
            plt.xlabel(xylabels)

        if (type(xylabels) == tuple):
            xlabel, ylabel = xylabels[0], xylabels[1]
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)


    if ('label' in kargs.keys()):
        plt.legend()

    if (grid): plt.grid(True)

    if (ylog): plt.yscale('log')

    return c

#--- hfit


def hfit(x, fun, guess = None, bins = 100, range = None,
            parnames = None, formate = '6.2f'):
    """ fit and plot a histogram to a function with guess parameters
    inputs:
    x    : np.array, values to build the histogram
    fun  : callable(x, *parameters) or string, function to fit
           str  = ['gaus', 'line', 'exp' ], for gaussian, line fit
    guess: tuple (None), values of the guess/initial parameters for the fit
           if fun is a predefined function, no need to add initial gess parameters
    bins : int (100), tuple, bins of the histogram
    range: tuple (None), range of the values to histogram
    parnames : tuple(str) (None), names of the parameters
    """
    fun, guess, fnames = hfitm._predefined_function(fun, guess, x)
    ys, xs, _ = hist(x, bins, range = range, stats = False)
    pars, parscov = hfitm.hfit(x, fun, guess, bins, range)
    xcs = 0.5* (xs[1:] + xs[:-1])
    parnames = parnames if parnames is not None else fnames
    ss  = str_parameters(pars, parscov, parnames, formate = formate)
    plt.plot(xcs, fun(xcs, *pars), label = ss);
    plt.legend()
    return pars, parscov


#---- DATA FRAME

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
    subplot = canvas(len(labels), ncolumns)
    for i, label in enumerate(labels):
        subplot(i + 1)
        values = ut.remove_nan(df[label].values)
        xrange = None if label not in ranges.keys() else ranges[label]
        hist(values, bins, range = xrange)
        plt.xlabel(label); plt.grid();
