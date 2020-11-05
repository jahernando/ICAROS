import numpy as np
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt


def hist(x , bins, stats = True, xylabels = None, **kargs):

    if (not ('histtype' in kargs.keys())):
        kargs['histtype'] = 'step'

    if (stats):
        range = kargs['range'] if 'range' in kargs.keys() else (np.min(x), np.max(x))
        sel = (x >= range[0]) & (x <= range[1])
        mean = np.mean(x)
        std  = np.std(x)
        sentries  =  f'entries = {len(x)}'
        smean     =  r'mean = {:7.2f}'.format(mean)
        sstd      =  r'std  = {:7.2f}'.format(std)
        sstat     =  f'{sentries}\n{smean}\n{sstd}'
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
