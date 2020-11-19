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

#--- WFs

def wf(z, erec, eraw = None, step = 2.,
       xylabels = ('z (mm)', 'Energy (keV)', 'Energy (adc)'), **kargs):
    """ Draw the (e, z) wave-form
    inputs:
        z    : np.array, z-hit positions
        erec : np.array, energy or intensity of the hits
        eraw : np.array (optional), energy raw of the hits
        step : float (2), wf-step size
        xylabels :  tuple(str), labes of x, y, erec and eraw
    """

    nplots = 1 if eraw is None else 2
    subplot = pltext.canvas(nplots)

    xlabel, elabel = xylabels[:2]

    bins = np.arange(np.min(z), np.max(z) + step, step)

    subplot(1)
    pltext.hist(z, bins, weights = erec, stats = False, **kargs)
    plt.xlabel(xlabel); plt.ylabel(elabel)

    if (eraw is None): return

    e2label = xylabels[2]
    plt.gca().twinx()
    pltext.hist(z, bins, weights = eraw, stats = False, **kargs)
    plt.ylabel(e2label)

    subplot(2)
    wf_rec, wf_zs = np.histogram(z, bins, weights = erec)
    wf_raw, wf_zs = np.histogram(z, bins, weights = eraw)

    wf_fc  = wf_rec/wf_raw
    wf_zcs = ut.centers(wf_zs)

    plt.plot(wf_zcs, wf_fc, marker = 'o');
    plt.xlabel(xlabel); plt.ylabel(elabel + '/' + e2label)

    return


def xyspot(x, y, erec, eraw = None, step = 10.,
           xylabels = ('x (mm)', 'y (mm)', 'Energy (keV)', 'Energy (adc)'),
           **kargs):
    """ Draw the (x, y) enery spot
    inputs:
        x    : np.array, x-hit positions
        y    : np.array, y-hit positions
        erec : np.array, energy or intensity of the hits
        eraw : np.array (optional), energy raw of the hits (optional)
        step : float (10), xy-step size
        xylabels :  tuple(str), labes of x, y, erec and eraw
    """

    bins = (ut.arstep(x, 10), ut.arstep(y, 10.))
    xlabel, ylabel, elabel = xylabels[:3]

    nplots = 1 if eraw is None else 3
    subplot = pltext.canvas(nplots)

    subplot(1)
    qrecs, xs, ys = np.histogram2d(x, y, bins, weights = erec);
    xms, yms = np.meshgrid(ut.centers(xs), ut.centers(ys))
    plt.hist2d(xms.flatten(), yms.flatten(), bins, weights = qrecs.T.flatten(), **kargs);
    plt.xlabel(xlabel); plt.ylabel(ylabel);
    cbar = plt.colorbar(); cbar.set_label(elabel)
    # is the same:
    # plt.hist2d(x, y, bins, weights = erec);

    if (nplots == 1): return
    e2label = xylabels[3]

    subplot(2)

    qraws, xs, ys = np.histogram2d(x, y, bins, weights = eraw);
    plt.hist2d(xms.flatten(), yms.flatten(), bins, weights = qrecs.T.flatten(), **kargs);
    plt.xlabel(xlabel); plt.ylabel(ylabel);
    cbar = plt.colorbar(); cbar.set_label(e2label)


    subplot(3)
    fc = qrecs/qraws
    fc[np.isnan(fc)] = 0.
    plt.hist2d(xms.flatten(), yms.flatten(), bins, weights = fc.T.flatten(), **kargs);
    plt.xlabel(xlabel); plt.ylabel(ylabel);
    cbar = plt.colorbar(); cbar.set_label(elabel + '/' + e2label)

    plt.tight_layout()
    return
