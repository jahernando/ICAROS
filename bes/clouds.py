import numpy             as np
import pandas            as pd
#import tables            as tb


import hipy.utils  as ut

#--- utilities

arstep = ut.arstep

def to_indices(cells, bins):
    """ converts the cells x,y,z positions into indices (ix, iy, iz)
    """
    icells =  [np.digitize(icell, ibin) - 1 for icell, ibin in zip(cells, bins)]
    return icells


def to_coors(vs):
    """ convert a list of m-size of vector of n-dim into n-dim list of coordinates eatchof m-dize (x1, x2,x)
    """
    ndim = len(vs[0])
    xs = [[vi[i] for vi in vs] for i in range(ndim)]
    return xs


def to_vectors(vals):
    """ convert a n-list of m-size into a list of m-size of n-vectors
    """
    ndim, nsize = len(vals), len(vals[0])
    vvals = [np.array([val[k] for val in vals]) for k in range(nsize)]
    return np.array(vvals)

def to_ids(icoors, scale = 1000):
    """ generate a unique id for coordinates (x1, x2, ...), xi
    a m-size arrpy with the xi-components
    icoor are always integer and positive indices!!
    """
    ndim, nsize = len(icoors), len(icoors[0])

    #ndim  = len(icoors)
    #nsize = len(icoors[0])
    #for i in range(ndim): assert len(icoors[i]) == nsize

    ids  = [np.sum([(scale**i) * icoors[i][k] for i in range(ndim)]) for k in range(nsize)]
    ids  = np.array(ids).astype(int)
    return ids


def get_moves_updown(ndim):

    def u1(idim):
        ui1 = np.zeros(ndim)
        ui1[idim] = 1
        return ui1.astype(int)
    vs = []
    for i in range(ndim):
        vs.append(u1(i))
        vs.append(-u1(i))
    vs.pop(0)
    return vs

def get_moves(ndim):
    """ returns movelments of combination of 1-unit in each direction
    i.e for ndim =2 returns [(1, 0), (1, 1), (0, 1,) (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    """

    u0 = np.zeros(ndim)
    def u1(idim):
        ui1 = np.zeros(ndim)
        ui1[idim] = 1
        return ui1.astype(int)

    vs = (u0, u1(0), -1 * u1(0))
    for idim in range(1, ndim):
        us = (u0, u1(idim), -1 * u1(idim))
        vs = [(vi + ui).astype(int) for vi in vs for ui in us]
    vs.pop(0)

    return vs


#--- clouds

def potential(coors, steps, weights = None):
    """ compute the clouds potential
    inputs:
        coors: tuple(arrays), a m-dim size list with the n-size coordinates
        steps: tuple(float), a m-dim size with the size in each of the m-dimensions
        weights: array, a n-size value of the weights
    returns:
        energy: array with the energy of the voxels in the m-dim space
        bins  : m-dim size list with the bins (edges) of the space for each coordinate
    """

    ndim  = len(coors)
    nsize = len(coors[0])
    weights = weights if weights is not None else np.ones(nsize)

    assert len(steps)   == ndim
    for i in range(ndim): assert len(coors[i]) == nsize
    assert len(weights) == nsize

    bins         = [arstep(x, step, True) for x, step in zip(coors, steps)]
    bins_centers = [ut.centers(ibin) for ibin in bins]

    #icoors   = [np.digitize(x, xbins) -1 for x, xbins in zip(coors, bins)]
    icoors       = [np.digitize(coors[i], bins[i]) - 1 for i in range(ndim)]
    #idcoors  = get_ids(icoors)
    #idpoints = get_ids(ipoints)

    #icoors       = [np.digitize(coors[i], bins[i]) - 1 for i in range(ndim)]

    pot, edges = np.histogramdd(coors, bins, weights = weights)

    return pot, edges


def voxels(potential, bins):
    """ from the potential space and the bins returns the voxels with its potential
    returns:
        xcells: tuple(array), a m-dim list of n-size arrays with the coordinats of the voxels
        potentials: array a n-size array with the potential of the voxels
    """

    centers      = [ut.centers(ibin) for ibin in bins]
    sel          = potential > 0
    cells        = to_coors(np.argwhere(sel))
    ndim, nsize  = len(cells), len(cells[0])
    xcells       = [centers[i][cells[i]] for i in range(ndim)]
    weights      = potential[cells]

    return xcells, weights
    

def neighbours(potential, bins):
    """ returns the number of neighbours with potential
    returns:
        xcells: tuple(array), a m-dim list of n-size arrays with the coordinats of the voxels
        counts: array a n-size array with the number of neighbourgs
    """

    shape        = potential.shape
    steps        = [ibin[1] - ibin[0] for ibin in bins]
    centers      = [ut.centers(ibin) for ibin in bins]

    sel          = potential > 0
    cells        = to_coors(np.argwhere(sel))
    ndim, nsize  = len(cells), len(cells[0])
    xcells       = [centers[i][cells[i]] for i in range(ndim)]
    weights      = potential[cells]

    counts        = np.full(shape, 0)
    counts[sel]   = 1

    #moves = get_moves_updown(ndim)
    moves = get_moves(ndim)

    for move in moves:
        coors_next         = [xcells[i] + steps[i] * move[i] for i in range(ndim)]
        potential_next, _  = np.histogramdd(coors_next, bins, weights = weights)

        isel                = potential_next > 0
        counts[sel & isel] += 1

    return xcells, counts[cells]


def gradient(potential, bins):
    """ returns the grandient potential within neighbourgs
    returns:
        xcells: tuple(array), a m-dim list of n-size arrays with the coordinats of the voxels
        deltas: array a n-size array with the increase of the potential
        dirs  : a m-dim size list with the n-size coordiates of the gradent direction
    """


    shape        = potential.shape
    steps        = [ibin[1] - ibin[0] for ibin in bins]
    centers      = [ut.centers(ibin) for ibin in bins]

    sel          = potential > 0
    cells        = to_coors(np.argwhere(sel))
    ndim, nsize  = len(cells), len(cells[0])
    xcells       = [centers[i][cells[i]] for i in range(ndim)]
    weights      = potential[cells]

    nn_potential   = np.copy(potential)
    nn_ids         = np.full((*shape, ndim), 0)

    #moves = get_moves_updown(ndim)
    moves = get_moves(ndim)

    for move in moves:
        coors_next         = [xcells[i] + steps[i] * move[i] for i in range(ndim)]
        potential_next, _  = np.histogramdd(coors_next, bins, weights = weights)


        isel                     = potential_next > nn_potential
        nn_potential[sel & isel] = potential_next[sel & isel]

        if (np.sum(sel & isel) > 0):
            nn_ids[sel & isel]    = -1 * np.array(steps) * move


    deltas  = nn_potential[cells] - potential[cells]
    dirs    = to_coors(nn_ids[cells])

    return xcells, deltas, dirs

def paths(cells, bins, steps, dirs):
    """ from the gradiend directions (dirs) compute the paths for each voxel
    to its node:
    returns:
        node : array, n-size array with the index of the node in the list of cells
        ipath: array, n-size array with the index of next voxel in the path to its node
        paths: list(list), list of indices of the voxels in the path to its node.
               # TODO, we want to return this?
    """

    ndim, nsize = len(cells), len(cells[0])
    print('dimensions ', ndim, 'size ', nsize)
    icells = to_indices(cells, bins)

    idcells    = to_ids(icells)
    nn_ipath   = np.arange(nsize) # to_ids(icells)
    nn_inode   = np.arange(nsize)

    ipos  = to_vectors(icells)
    idirs = (to_vectors(dirs)/np.array(steps)).astype(int)

    vnull = np.zeros(ndim)

    def _path(i, ipath):

        ipath.append(i)

        if (np.all(idirs[i] == vnull)):
            return ipath
        #print(True, 'pos i ', i, 'icoors', ipos[i], 'id', idcells[i], 'dir ', idirs[i],
        #  'idnext', nn_kcell[i])

        iloc  =  ipos[i] + idirs[i]
        idloc =  to_ids([(iloc[i],) for i in range(ndim)])[0]

        isel = np.isin(idcells, idloc)
        ii = int(np.argwhere(isel))
        nn_ipath[i] = ii

        return _path(ii, ipath)

    paths = []
    for i in range(nsize):
        ipath       = _path(i, [])
        nn_inode[i] = ipath[-1]
        paths.append(ipath)

    return nn_inode, nn_ipath, paths


def energy_nodes(ene, nnode):
    """ returns the energy of the nodes, from the ene, energy of the voxels, and nnode,
    index of the node of the voxel
    returns:
        nenode: array, n-size array with the sum of the energy of the voxels in the node
                       for the nodes, for the rest of the voxels is zero.
    """
    nsize = len(nnode)
    enodes = np.zeros(nsize)
    ks = np.unique(nnode)
    for ki in ks:
        sel = nnode == ki
        enodes[ki] = np.sum(ene[sel])
    return enodes
