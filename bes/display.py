import numpy             as np
import pandas            as pd
import tables            as tb
import matplotlib.pyplot as plt

import hipy.pltext      as pltext
import hipy.utils       as ut


from mpl_toolkits.mplot3d    import Axes3D


def track(x, y, z, ene, scale = 10., title = '', cmap = 'magma'):

    rene = ut.arscale(ene, scale)

    ax   = plt.subplot(111, projection = '3d')
    ax.scatter(x, y, z, c = rene, s = rene, alpha = 0.2, cmap = cmap)
    ax.set_xlabel('X (mm)');
    ax.set_ylabel('Y (mm)');
    ax.set_zlabel('Z (mm)');
    plt.gcf().colorbar();
    #ax.colorbar()

    return


def event(x, y, z, ene, scale = 10., rscale = 9., chamber = False, **kargs):
    """ Draw an event with hits x, y, z, ene
    inputs:
        x   : np.array, x-hit positions
        y   : np.array, y-hit positions
        z   : np.array, z-hit positions
        ene : np.array, energy or intensity of the hits
        scale  : float, scale factor of the markers
        rscale : float, scale factor of the size with energy/intensity
        chamber: bool (false), in NEW x,y,z frame
    """

    if (not 'alpha'  in kargs.keys()): kargs['alpha']  = 0.5
    if (not 'cmap'   in kargs.keys()): kargs['cmap']   = 'Greys'
    if (not 'marker' in kargs.keys()): kargs['marker'] = 's'

    rene = ut.arscale(ene)

    zsize, xysize = (0., 500.), (-200., 200)

    fig = plt.figure(figsize=(12, 9));
    #plt.subplots(2
    ax3D = fig.add_subplot(221, projection='3d')
    size   = scale       * (1. + rscale * rene)
    color  = np.max(ene) * rene
    p3d = ax3D.scatter(z, x, y, s = size, c = color, **kargs)
    ax3D.set_xlabel('z (mm)')
    ax3D.set_ylabel('x (mm)')
    ax3D.set_zlabel('y (mm)')
    if chamber:
        ax3D.set_xlim(zsize)
        ax3D.set_ylim(xysize)
        ax3D.set_zlim(xysize)

    plt.subplot(2, 2, 2)
    plt.scatter(x, z, s = size, c = color, **kargs)
    ax = plt.gca()
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    if chamber:
        plt.xlim(xysize); plt.ylim(zsize)
    plt.colorbar();

    plt.subplot(2, 2, 3)
    plt.scatter(z, y, s = size, c = color, **kargs)
    ax = plt.gca()
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('y (mm)')
    if chamber:
        plt.xlim(zsize); plt.ylim(xysize)
    plt.colorbar();

    plt.subplot(2, 2, 4)
    plt.scatter(x, y, s = size, c = color, **kargs)
    ax = plt.gca()
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    if chamber:
        plt.xlim(xysize); plt.ylim(xysize)
    plt.colorbar();
    plt.tight_layout()
    return


def wf(z, e, eraw = None, step = 2.):
    """ Draw the (e, z) wave-form
    inputs:
        z    : np.array, z-hit positions
        e    : np.array, energy or intensity of the hits
        eraw : np.array (optional), energy raw of the hits
        step : float (2), wf-step size
    """

    bins =  ut.arstep(z, step)
    nplots = 1 if eraw is None else 2
    subplot = pltext.canvas(nplots)

    subplot(1)
    pltext.hist(z, bins, weights = e, alpha = 0.5, stats = False, lw = 2)
    plt.xlabel('z (mm)'); plt.ylabel('E (keV)')

    if (eraw is None): return

    plt.gca().twinx();
    pltext.hist(z, bins, weights = eraw, alpha = 0.4, stats = False,
                lw = 2, color = 'blue')
    plt.xlabel('z (mm)'); plt.ylabel('E (adc)', c = 'blue')

    subplot(2)
    pltext.hist(z, bins, weights = e/eraw, stats = False)
    plt.xlabel('z (mm)'); plt.ylabel('calibration (keV/adc)')
    plt.tight_layout()

    return


def xyspot(x, y, e, eraw = None, step = 10.):
    """ Draw the (x, y) enery spot
    inputs:
        x    : np.array, x-hit positions
        y    : np.array, y-hit positions
        e    : np.array, energy or intensity of the hits
        eraw : np.array (optional), energy raw of the hits (optional)
        step : float (10), xy-step size
    """


    bins = (ut.arstep(x, step), ut.arstep(y, step))

    nplots = 1 if eraw is None else 3
    subplot = pltext.canvas(nplots)

    subplot(1)
    plt.hist2d(x, y, bins, weights = e, cmap = 'Greys');
    plt.xlabel('x (mm)'); plt.ylabel('y (mm)');
    cbar = plt.colorbar(); cbar.set_label('Energy (keV)')

    subplot(2)
    plt.hist2d(x, y, bins, weights = eraw, cmap = 'Greys');
    plt.xlabel('x (mm)'); plt.ylabel('y (mm)');
    cbar = plt.colorbar(); cbar.set_label('Energy (adc)')

    subplot(3)
    plt.hist2d(x, y, bins, weights = e/eraw, cmap = 'Greys');
    plt.xlabel('x (mm)'); plt.ylabel('y (mm)');
    cbar = plt.colorbar(); cbar.set_label('calibration factor (keV/adc)')

    return
