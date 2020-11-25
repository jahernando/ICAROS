import numpy as np
import random

import operator
import functools

import hipy.utils as ut
import hipy.hfit  as hfitm
from dataclasses import dataclass

import invisible_cities.core.fit_functions as fitf

import matplotlib.pyplot as plt
from cycler import cycler

# cmaps:
# magma, inferno, Red, Greys, Blues, spring, summer, autumn, winter,
# cool, Wistia, hot, jet

# colorbar
# cbar = plt.colorbar(heatmap)
# cbar.ax.set_yticklabels(['0','1','2','>3'])
# cbar.set_label('# of contacts', rotation=270)
#


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
    ## TODO: problem with formate-change key name - conflict
    """

    if (not ('histtype' in kargs.keys())):
        kargs['histtype'] = 'step'

    if (stats):
        range   = kargs['range']          if 'range'         in kargs.keys() else None
        formate = kargs['stats_format']   if 'stats_format'  in kargs.keys() else '6.3f'
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


def hfit(x, bins, fun, guess = None, range = None,
            parnames = None, formate = '6.2f', **kargs):
    """ fit and plot a histogram to a function with guess parameters
    inputs:
    x    : np.array, values to build the histogram
    bins : int, tuple, bins of the histogram
    fun  : callable(x, *parameters) or string, function to fit
           str  = ['gaus', 'line', 'exp' ], for gaussian, line fit
    guess: tuple (None), values of the guess/initial parameters for the fit
           if fun is a predefined function, no need to add initial gess parameters
    range: tuple (None), range of the values to histogram
    parnames : tuple(str) (None), names of the parameters
    """
    fun, guess, fnames = hfitm._predefined_function(fun, guess, x)
    ys, xs, _ = hist(x, bins, range = range, stats = False, **kargs)
    pars, parscov = hfitm.hfit(x, bins, fun, guess, range)
    xcs = 0.5* (xs[1:] + xs[:-1])
    parnames = parnames if parnames is not None else fnames
    ss  = hfitm.str_parameters(pars, parscov, parnames, formate = formate)
    kargs['label'] = ss if 'label' not in kargs.keys() else kargs['label'] + '\n' + ss
    plt.plot(xcs, fun(xcs, *pars), **kargs);
    plt.legend()
    return pars, parscov


#---- Profile

def hprofile(x, y, nbins = 10, std = False, xrange = None , yrange = None,  **kargs):
    """
    """
    xs, ys, eys = fitf.profileX(x, y, nbins, xrange, yrange, std = std)
    plt.errorbar(xs, ys, yerr = eys, **kargs)
    return xs, ys, eys


def hprofile_scatter(x, y, nbins = 10, std = False, xrange = None , yrange = None,  **kargs):
    """
    """
    plt.scatter(x, y, **kargs)
    kargs['alpha'] = 1.
    xs,ys, eys = hprofile(x, y, nbins, std, xrange, yrange, **kargs)
    return xs, ys, eys


#def hpscatter(uvar, vvar, ulabel = '', vlabel = '', urange = None , vrange = None,
#              nbins_profile = 10, **kargs):
#    plt.scatter(uvar, vvar, **kargs)
#    kargs['alpha'] = 1.
#    if ('c' in kargs.keys()): del kargs['c']
#    #kargs['c']     = kargs['c'] if 'c' in kargs.keys() else 'black'
#    hprofile(uvar, vvar, ulabel, vlabel, urange, vrange, nbins_profile, **kargs)
#    return


def hprofile_in_sigma(x, y, nbins = 20, nsigma = 2, niter = 10, **kargs):
    """ plot profile after n-iterations selection entries in the nsigma window in x.
    inputs:
        x     : np.array, x-values
        y     : np.array, y-values
        nbins : int, number of profile bins in the x-range
        nsigma: float, number of std in the y-bins to accept in the next iteration
        niter : int, number of iterations
        kargs : extra matplot plot key-options
    returns:
        xs    : x-points of the n-iteration profile
        ys    : y-points
        eys   : y-errors
    """

    def in_sigma(x, y, xs, ys, eys, nsigma):
        xx = np.copy(x)
        xx[xx <= xs[0]]  = xs[0]
        xx[xx >= xs[-1]] = xs[-1]
        x0  = np.min(xx)
        dx  = xs[1] - xs[0]
        ix  = ((xx-x0 - 0.5* dx ) / dx).astype(int)
        nbins = len(ys)
        ix[ ix >= nbins ] = nbins - 1
        yr = ys[ix]
        sel = np.abs(y - yr) < nsigma * eys[ix]
        return (x[sel], y[sel])

    xs, ys, eys = None, None, None
    for i in range(niter):
        ix, iy = (x, y) if i == 0 else in_sigma(x, y, xs, ys, eys, nsigma)
        xs, ys, eys = fitf.profileX(ix, iy, nbins, std = True);
        if (i == niter - 1):
            #hprofile(x, y, nbins, **kargs)
            plt.errorbar(xs, ys, yerr = eys, **kargs)

    return xs, ys, eys



#---- DATA FRAME

def df_inspect(df, labels = None, bins = 100, ranges = {}, ncolumns = 2):
    """ histogram the variables of a dataframe
    inputs:
        df      : dataframe
        labels  : tuple(str) list of variables. if None all the columns of the DF
        bins    : int (100), number of nbins
        ranges  : dict, range of the histogram, the key must be the column name
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
        plt.xlabel(label);


def df_corrmatrix(xdf, xlabels):
    """ plot the correlation matrix of the selected labels from the dataframe
    inputs:
        xdf     : DataFrame
        xlabels : tuple(str) list of the labels of the DF to compute the correlation matrix
    """
    _df  = xdf[xlabels]
    corr = _df.corr()
    fig = plt.figure(figsize=(12, 10))
    #corr.style.background_gradient(cmap='Greys').set_precision(2)
    plt.matshow(abs(corr), fignum = fig.number, cmap = 'Greys')
    plt.xticks(range(_df.shape[1]), _df.columns, fontsize=14, rotation=45)
    plt.yticks(range(_df.shape[1]), _df.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    return


def df_corrprofile(df, name, labels, switch = False, **kargs):
    """ plot the scatter and profile plot between the name-variable
    of the df, DataFrame, vs each variable in labels list
    inpùts:
        df    : DataFrame
        name  : str, name of the variable for the x-axis profile
        labels: list(str), names of the variable sfor the y-axis profile
        swicth: bool, False. Switch x-variable and y-variable
    """
    sargs = copy(kargs)

    subplot = canvas(len(labels), len(labels))
    for i, label in enumerate(labels):
        subplot(i + 1)
        xlabel, ylabel = (name, label) if switch is False else (label, name)
        kargs['alpha'] = 0.1    if 'alpha' not in sargs.keys() else kargs['alpha']
        kargs['c']     = 'grey' if 'c'     not in sargs.keys() else kargs['c']
        plt   .scatter (df[xlabel], df[ylabel], **kargs)
        kargs['alpha'] = 1.
        kargs['lw']    = 1.5 if 'lw'    not in sargs.keys() else kargs['lw']
        kargs['c']     = 'black'
        hprofile(df[xlabel], df[ylabel], **kargs)
        plt.xlabel(xlabel, fontsize = 12); plt.ylabel(ylabel, fontsize = 12);
    plt.tight_layout()
    return

def df_prod(df, labels):
    """ multiply element-wise the different colums with label in labels of the dataFrame, df
    inputs:
        df    :   DataFrame
        labels: tup(str), names of the columns
    """
    return functools.reduce(operator.mul, [df[label] for label in labels])

# backward compatibility
plt_inspect_df = df_inspect
corrmatrix     = df_corrmatrix
