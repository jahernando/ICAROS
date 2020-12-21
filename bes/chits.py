import numpy             as np
import pandas            as pd
import tables            as tb
import matplotlib.pyplot as plt
import bes.bes           as bes

from invisible_cities.reco import corrections as cof


#----- utility functions to get the DF

alpha = 2.76e-4

def get_chits_filename(run_number, label = 'ds_rough'):
    datadir    = f"/home/hernando/data/NEW"
    run_number = str(run_number)
    filename   = datadir + f'/chits_{label}_{run_number}.h5'
    return filename


def get_krmap_filename(run_number):
    map_fname = '/home/jrenner/analysis/NEW/maps/map_'+str(run_number)+'_config_NoChecks.h5'
    return map_fname



##------


def get_hits(hh, labels = ('X', 'Y', 'Z', 'DT', 'Ec', 'E', 'time'), vdrift = None):
    def _get(label):
        if (label == 'DT'):
            #vdrift = np.mean(maps.t_evol.dv)
            return hh['Z'].values/vdrift
        #if (label == 'DZ'):
        #    return hh['Z'].values - np.min(hh['Z'].values)
        return hh[label].values
    hits = [_get(label) for label in labels]
    return hits


def get_maps(map_fname):
    maps      = cof.read_maps(map_fname)
    return maps


def in_axis(val, axis):
    nsize = len(val)
    valin = np.zeros(nsize)
    valin = [np.sum(val[axis == k]) for k in axis]
    return np.array(valin)


def share_ene(ene, q, z, sel):

    etot = np.sum(ene)
    e0s = in_axis(ene, z)
    q0s = in_axis(q  , z)

    qnew         = np.copy(q)
    qnew[~sel]   = 0.
    q0s          = in_axis(qnew, z)
    sel0         = q0s > 0
    qnew[sel0]   = qnew[sel0]/q0s[sel0]
    enew         = e0s * qnew
    etotnew      = np.sum(enew)
    enew         = enew * etot / etotnew

    return enew, qnew * q0s


def get_filter_hits(evt, q0):
    """ returns the x, y, z, eraw, erec, q, time of this hits above q > q0
    (notice that total initil eraw is shared between the filtered hits)
    """

    labels = ['X', 'Y', 'Z', 'E', 'Ec', 'Q', 'time']

    x, y, z, eraw, erec, q, times = get_hits(evt, ['X', 'Y', 'Z', 'E', 'Ec', 'Q', 'time'])

    sel = q > q0
    if (np.sum(sel) > 0):
        eraw, q = share_ene(eraw, q, z, sel)
        x, y, z, eraw, erec, q, times = x[sel], y[sel], z[sel], eraw[sel], erec[sel], q[sel], times[sel]

    return x, y, z, eraw, erec, q, times


def get_corrfac(maps):
    """ Return correction faction as variables (x, y, z) using the map
    """

    vdrift    = np.mean(maps.t_evol.dv)
    print('drift velocity ', vdrift)
    _corrfac  = cof.apply_all_correction(maps, apply_temp = True,
                                         norm_strat = cof.norm_strategy.kr)
    def corrfac(x, y, z, times):
        dt = z/vdrift
        return _corrfac(x, y, dt, times)

    return corrfac


def init_hits_summary(nsize = 1):
    labels =  ['eraw', 'erec', 'q', 'nhits', 'zmin', 'zmax', 'dz', 'rmax',
               'erawmax', 'qmax', 'erecmax', 'nhitsout']
    return bes.df_zeros(labels, nsize)


def hits_summary(ddhits, q0 = 0., corrfac = None):
    x, y, z, eraw, erec, q, times = get_filter_hits(ddhits, q0)
    nout = np.sum(np.isnan(erec))
    if (corrfac is not None):
        cfac   = corrfac(x, y, z, times)
        nout   = np.sum(np.isnan(cfac))

    rmax = np.max(np.sqrt(x * x + y * y))
    idat = {'eraw'  : np.sum(eraw),
            'erec'  : np.sum(erec),
            'q'     : np.sum(q),
            'nhits' : len(x),
            'zmin'  : np.min(z),
            'zmax'  : np.max(z),
            'dz'    : np.max(z) - np.min(z),
            'rmax'  : np.max(np.sqrt(x * x + y * y)),
            'erawmax'  : np.max(eraw),
            'qmax'     : np.max(q),
            'erecmax'  : np.max(erec),
            'nhitsout' : nout
            }
    return idat



#-----------------------------------------------
# Study of the corrections - City
#-----------------------------------------------



def dfh_corrections(dfh, maps, alpha = alpha):
    """ create a a DF per event with the correction factors starting from hits and maps
    """

    def dfh_extend_(dfh):
        ddmin = dfh.groupby('event').min()
        dftemp = pd.DataFrame({'event': ddmin.index.values,
                               'Zmin' : ddmin.Z.values})
        dfext = pd.merge(dfh, dftemp, on = 'event')
        dfext['DZ'] = dfext['Z'].values - dfext['Zmin'].values
        dfext['R'] = np.sqrt(dfext.X**2 + dfext.Y**2)
        return dfext

    ## extend the DF of Hits with Zmin and DZ
    dfh = dfh_extend_(dfh)

    ## get correction functions using maps
    vdrift    = np.mean(maps.t_evol.dv)
    #print('drift velocity ', vdrift)

    corrfac  = cof.apply_all_correction(maps, apply_temp = True,
                                        norm_strat = cof.norm_strategy.kr)

    def corrfac_geo(x, y):
        get_xy_corr_fun  = cof.maps_coefficient_getter(maps.mapinfo, maps.e0)
        return cof.correct_geometry_(get_xy_corr_fun(x,y))


    def corrfac_lt(x, y, dt):
        get_lt_corr_fun   = cof.maps_coefficient_getter(maps.mapinfo, maps.lt)
        return cof.correct_lifetime_(dt, get_lt_corr_fun(x, y))

    def corrfac_dz(dz, alpha = alpha, scale = 2.):
        return (1. + scale * alpha * dz)


    def corrfac_norma(x, y, t):
        val0 = corrfac(x, y, 0, t)
        val  = corrfac_geo(x, y)
        return val0/val


    ## get variables from hits
    labels = ['X', 'Y', 'Z', 'DT', 'Ec', 'E', 'time', 'Zmin', 'DZ']
    x, y, z, dt, erec, eraw, time, z0, dz = get_hits(dfh, labels, vdrift)

    # extend the dfh
    dfh['Echeck']     = eraw * corrfac(x, y, dt, time) #* corrfac_dz(dz)
    dfh['Ecorr']      = eraw * corrfac(x, y, dt, time) * corrfac_dz(dz)
    dfh['Egeo']       = eraw * corrfac_geo(x, y)
    dfh['Elt']        = eraw * corrfac_lt(x, y, dt)
    #dfh['Elt_center'] = eraw * corrfac_lt_center(x, y, dt)
    #dfh['Elt_z0']     = eraw * corrfac_lt_z0(x, y, z0)
    dfh['Enorma']     = eraw * corrfac_norma(x, y, dt)
    dfh['Edz']        = eraw * corrfac_dz(dz)

    # compute event variables
    ddmin = dfh.groupby('event').min()
    ddmax = dfh.groupby('event').max()
    ddave = dfh.groupby('event').mean()
    ddsum = dfh.groupby('event').sum()

    eraw  = ddsum['E'].values

    enum = dfh.groupby('event').apply(lambda x : np.sum(x['E'].values[np.isnan(x['Ec'])]))

    dz   = ddmax['Z'].values - ddmin['Z'].values
    ec   = ddsum['Ec'].values
    edz  = bes.energy_correction(ec, dz, alpha)

    events = ddmin.index.values

    ddc = {'event'  : events,
           'X'      : ddave['X'] .values,
           'Y'      : ddave['Y'].values,
           'Z'      : ddave['Z'].values,
           'Ec'     : ddsum['Ec'].values,
           'E'      : ddsum['E'].values,
           'DZ'     : ddmax['Z'].values - ddmin['Z'].values,
           'time'   : ddave['time'].values,
           'Edz'    : edz,
           'Ecorr'  : ddsum['Ecorr'].values,
           'Echeck' : ddsum['Echeck'].values,
           'fgeo'   : ddsum['Egeo'].values   / eraw,
           'flt'    : ddsum['Elt'].values    / eraw,
           #'flt_center' : ddsum['Elt_center'].values    / eraw,
           #'flt_z0' : ddsum['Elt_z0'].values / eraw,
           'fdz'    : ddsum['Edz'].values    / eraw,
           'fdz_global'   : edz/ec,
           'fnorma' : ddsum['Enorma'].values / eraw,
           'Enan'   : enum.values,
           'Zmin'   : ddmin['Zmin'].values,
           'Zmax'   : ddmax['Z'].values,
           'Rmax'   : ddmax['R'].values
           }

    evts = ddmin.index

    ddc = pd.DataFrame(ddc, index = ddmin.index)

    return ddc


def get_hitsrevisited_df(runs, sample_label = 'ds', hit_type = 'CHITs.highTh', alpha = alpha):

    fnames = [get_chits_filename(run, sample_label + '_rough') for run in runs]
    dfhs   = [pd.read_hdf(fname, hit_type) for fname in fnames]

    fnames = [get_krmap_filename(run) for run in runs]
    maps   = [get_maps(fname) for fname in fnames]

    ddhs   = [dfh_corrections(dfh, imap, alpha) for dfh, imap in zip(dfhs, maps)]

    ddh    = bes.df_concat(ddhs, runs)

    return ddh
