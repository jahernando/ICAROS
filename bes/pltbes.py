import numpy  as np
import pandas as pd
import tables as tb

import hipy.utils  as ut
import hipy.pltext as pltext
import bes.bes     as bes


#---- analysis - plotting

def str_energy_resolution(pars, upars, label = ''):
    ss    = label + '\n' if label != '' else ''
    ss  += r' $\sigma$ = {0:6.4f} $\pm$ {1:6.4f}'.format(abs(pars[2]), upars[2]) + '\n'
    mu, sigma, umu, usigma = pars[1], pars[2], upars[1], upars[2]
    R  = 235. * abs(sigma)/mu
    uR = 235. * np.sqrt(usigma**2 + R**2 * umu**2)/mu
    ss += r' R  = {0:6.4f} $\pm$ {1:6.4f}'.format(R, uR)+ '\n'
    return ss


def pfit_energy(enes, bins, **kargs):
    pars, upars = pltext.hfit(enes, bins, 'gausline', **kargs)
    label = kargs['label'] if 'label' in kargs.keys() else ''
    ss = str_energy_resolution(pars, upars, label = label)
    return ss, pars, upars
