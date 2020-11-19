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

    if (not 'alpha'  in kargs.keys()): kargs['alpha']  = 0.2
    if (not 'cmap'   in kargs.keys()): kargs['cmap']   = 'magma'
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
