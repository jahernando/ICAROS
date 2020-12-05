import numpy             as np
import pandas            as pd

import matplotlib.pyplot as plt

import hipy.utils        as ut
import hipy.pltext       as pltext
#import hipy.hfit         as hfit

#import bes.bes           as bes
import bes.clouds        as clouds
#import bes.chits         as chits
import bes.display       as nplay


#---- utils

def _cells(df, ndim):
    return [df['x'+str(i)].values for i in range(ndim)]


def _ocells(cells, i = 0):
    return cells[i:] + cells[:i]


def _csel(vals, sel):
    return [val[sel] for val in vals]


_karg = pltext.karg


#
# Low level elements
#

def dcloud_cells(cells, enes = None, xaxis = 0, **kargs):
    """ Draw cells, if enes, with its energy
    """

    ndim, nsize = len(cells), len(cells[0])

    enes = np.ones(nsize) if enes is None else enes

    kargs = _karg('marker',   's', kargs)
    kargs = _karg('c' ,  enes, kargs)
    #kargs = _karg('s' ,  10 * enes, kargs)

    ax = plt.gca()
    xcells = _ocells(cells, xaxis)
    ax.scatter(*xcells, **kargs)
    return
    #if (chamber): draw_chamber(coors, ax)


def dcloud_nodes(cells, enodes, **kargs):
    """ Draw cells that enodes > 0
    """

    kargs = _karg('marker',   '*', kargs)

    sel = enodes > 0

    kargs = _karg('s', enodes[sel], kargs)

    dcloud_cells(_csel(cells, sel), enodes[sel], **kargs)
    return


def dcloud_grad(cells, epath, xaxis = 0, **kargs):
    """ Draw the gradient of the cells
    """

    ndim = len(cells)
    ncells = _csel(cells, epath)
    coors  = _ocells(cells , xaxis) if xaxis != 0 else cells
    vcoors = _ocells(ncells, xaxis) if xaxis != 0 else ncells

    xdirs =[vcoor - coor for vcoor, coor in zip(vcoors, coors)]
    opts = {'scale_units': 'xy', 'scale' : 2.} if ndim == 2 else {'length' : 0.4}

    plt.quiver(*coors, *xdirs, **opts, **kargs)


def dcloud_segments(cells, epass, epath, lpath, xaxis = 0, **kargs):
    """ Draw the segments associated to the pass with epass > 0
    """

    kids = list(np.argwhere(epass > 0))
    xcells   = _ocells(cells, xaxis) if xaxis != 0 else cells
    segments = [get_segment(xcells, get_pass_path(kid, epath, lpath)) for kid in kids]
    for segment in segments:
        #print(segment)
        plt.plot(*segment, **kargs)


#
#  High Level
#

def dcloud_steps(dfclouds, xaxis = 0):

    cells  = _cells(dfclouds, 3)
    enes   = dfclouds.ene.values
    enodes = dfclouds.enode.values
    nodes  = dfclouds.node.values
    epath  = dfclouds.epath.values
    lpath  = dfclouds.lpath.values
    epass  = dfclouds.epass.values


    subplot = pltext.canvas(6, 2, 10, 12)

    subplot(1, '3d')
    dcloud_cells(cells, 1000. * enes, alpha = 0.1, xaxis = xaxis);
    dcloud_grad(cells, epath, xaxis = xaxis)

    subplot(2, '3d')
    dcloud_cells(cells, nodes, alpha = 0.1, xaxis = xaxis);
    dcloud_nodes(cells, 1000. * enodes, marker = 'o', alpha = 0.8, xaxis = xaxis)

    subplot(3, '3d')
    dcloud_cells(cells, 1000. * nodes, alpha = 0.1, xaxis = xaxis);
    dcloud_grad(cells, lpath, xaxis = xaxis)

    subplot(4, '3d')
    dcloud_cells(cells,     1000. * nodes , alpha = 0.1, xaxis = xaxis);
    dcloud_nodes(cells,     1000. * enodes, marker = 'o', alpha = 0.8, xaxis = xaxis)
    dcloud_nodes(cells, 5 * 1000. * epass , marker = '^', alpha = 0.8, xaxis = xaxis)

    subplot(5, '3d')
    dcloud_cells   (cells, alpha = 0.05, xaxis = xaxis);
    dcloud_nodes   (cells, 1000. * enodes, alpha = 0.8, marker = 'o', xaxis = xaxis)
    dcloud_segments(cells, epass, epath, lpath, xaxis = xaxis)

    return
