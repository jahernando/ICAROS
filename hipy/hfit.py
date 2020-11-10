import numpy as np
import scipy.optimize as optimize

import matplotlib.pyplot as plt


"""
    Module for histogram fitting

    predefined functions: gaus, lin, exp, gausline, gausexp

"""


functions = ['gaus', 'line', 'exp', 'gausline', 'gausexp']

current_module = __import__(__name__)


def _predefined_function(fun, guess, x):

    fnames = None
    if (type(fun) == str):
        assert fun in functions
        if (guess is None):
            guess = eval('g'+fun)(x)
        fnames = getattr(current_module.hfit, 'n' + fun)
        fun    = getattr(current_module.hfit, 'f' + fun)

    return fun, guess, fnames

def hfit(x, fun, guess = None, bins = 100, range = None):
    """ fit a histogram to a function with guess parameters
    inputs:
    x    : np.array, values to build the histogram
    fun  : callable(x, *parameters) or string, function to fit
           str  = ['gaus', 'line', 'exp' ], for gaussian, line fit
    guess: tuple (None), values of the guess/initial parameters for the fit
           if fun is a predefined function, no need to add initial gess parameters
    bins : int (100), tuple, bins of the histogram
    range: tuple (None), range of the values to histogram
    """

    fun, guess, _ = _predefined_function(fun, guess, x)

    range = range if range is not None else (np.min(x), np.max(x))
    yc, xe = np.histogram(x, bins, range)
    xc = 0.5 * (xe[1:] + xe[:-1])
    fpar, fcov = optimize.curve_fit(fun, xc, yc, guess)

    return fpar, np.sqrt(np.diag(fcov))


def plt_hfit(x, fun, guess = None, bins = 100, range = None,
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
    fun, guess, fnames = _predefined_function(fun, guess, x)
    ys, xs, _ = plt.hist(x, bins, range, histtype = 'step')
    pars, parscov = hfit(x, fun, guess, bins, range)
    xcs = 0.5* (xs[1:] + xs[:-1])
    parnames = parnames if parnames is not None else fnames
    ss  = str_parameters(pars, parscov, parnames, formate = formate)
    plt.plot(xcs, fun(xcs, *pars), label = ss)
    plt.legend()
    return pars, parscov



def str_parameters(pars, covpars, parnames = None, formate = '6.2f'):
    s = ''
    for i, par in enumerate(pars):
        namepar = r'$a_'+str(i)+'$' if parnames is None else parnames[i]
        covpar   = covpars[i]
        s += namepar + ' = '
        s += (('{0:'+formate+'}').format(par))   + r'$\pm$'
        s += (('{0:'+formate+'}').format(covpar))+ '\n'
    return s



def fgaus(x, a, b, c):
    """ return a gausian function
    """
    if (c <= 0.): return 0.
    return a * np.exp(- (x-b)**2 / (2* c**2) )


def ggaus(x):
    """ return guess parameters for a guassian function
    """
    return (len(x), np.mean(x), np.std(x))


ngaus = [r'$N_\mu$', r'$\mu$', r'$\sigma$']


def fline(x, a, b):
    """ return a line a* x + b
    """
    return a * x + b


def gline(x):
    """ return guess parameters for a line function
    """
    ys, xe = np.histogram(x, 2)
    xc = 0.5 * (xe[1:] + xe[:-1])
    a = (ys[1] - ys[0])/ (xc[1] - xc[0])
    b = ys[0] - a * xc[0]
    return a, b


nline = ['a', 'b']


def fexp(x, a, b):
    """ an exponential function a * exp(-b * x)
    """
    return a * np.exp( - b * x)


def gexp(x):
    """ guess parameters for an exponential
    """
    ys, xs = np.histogram(x, 2)
    xcs = 0.5 * (xs[1:] + xs[:-1])
    dx = xcs[1] - xcs[0]
    b = - (np.log(ys[1]) - np.log(ys[0]))/dx
    a = ys[0] * np.exp(b * xcs [0])
    return (a, b)


nexp = [r'$N_\tau$', r'$\tau$']

def fgausline(x, na, mu, sig, a, b):
    return fgaus(x, na, mu, sig) + fline(x, a, b)


def ggausline(x):
    return list(ggaus(x)) + list(gline(x))


ngausline = ngaus + nline


def fgausexp(x, na, mu, sig, nb, tau):
    return fgaus(x, na, mu, sig) + fexp(x, nb, tau)


def ggausexp(x):
    return list(ggaus(x)) + list(gexp(x))


ngausexp = ngaus + nexp

#------------
