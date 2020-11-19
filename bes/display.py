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


def event(x, y, z, ene, scale = 250., title = '', cmap = 'magma'):

    rene = ut.arscale(ene)

    fig = plt.figure(figsize=(12, 9));
    #plt.subplots(2
    ax3D = fig.add_subplot(221, projection='3d')
    size   = scale       * (1. + 2. * rene)
    color  = np.max(ene) * rene
    p3d = ax3D.scatter(z, x, y, s = size, c = color, alpha = 0.2, marker='s')
    ax3D.set_xlabel('z (mm)')
    ax3D.set_ylabel('x (mm)')
    ax3D.set_zlabel('y (mm)')
    plt.title(title)

    plt.subplot(2, 2, 3)
    plt.scatter(x, z, s = size, c = color, alpha = 0.2, cmap = cmap, marker = 's')
    ax = plt.gca()
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    plt.colorbar();

    plt.subplot(2, 2, 2)
    plt.scatter(z, y, s = size, c = color, alpha = 0.2, cmap = cmap, marker = 's')
    ax = plt.gca()
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('y (mm)')
    plt.colorbar();

    plt.subplot(2, 2, 4)
    plt.scatter(x, y, s = size, c = color, alpha = 0.2, cmap = cmap, marker = 's')
    ax = plt.gca()
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.colorbar();
    plt.tight_layout()
    return
