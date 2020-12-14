import numpy             as np
import pandas            as pd
import tables            as tb

#to_df = pd.DataFrame.from_records
from invisible_cities.reco import corrections as cof

import hipy.utils        as ut
import bes.bes           as bes
import bes.chits         as chits
import clouds        as clouds


get_chits_filename = chits.get_chits_filename
get_krmap_filename = chits.get_krmap_filename
get_maps           = chits.get_maps
get_hits           = chits.get_hits


def load_data(runs, sample_label = 'ds'):
    """ load hits and maps data
    """

    fnames  = [get_chits_filename(run, sample_label + '_rough') for run in runs]
    print(fnames)
    dfhits   = [pd.read_hdf(fname, 'CHITs.lowTh') for fname in fnames]
    dfhitHTs = [pd.read_hdf(fname, 'CHITs.highTh') for fname in fnames]


    fnames = [get_krmap_filename(run) for run in runs]
    dmaps  = [get_maps(fname) for fname in fnames]

    return (dfhits, dfhitHTs, dmaps)


def get_corrfac(maps):
    """ given maps, return the correction factor function based on (x, y, z, times)
    """

    vdrift    = np.mean(maps.t_evol.dv)
    print('drift velocity ', vdrift)
    _corrfac  = cof.apply_all_correction(maps, apply_temp = True,
                                         norm_strat = cof.norm_strategy.kr)
    def corrfac(x, y, z, times):
        dt = z/vdrift
        return _corrfac(x, y, dt, times)

    return corrfac


def cloudsdia(runs, sample_label = 'ds', ntotal = 10000):
    """ a City: read the hits and the maps, runs clouds and recompute energy, returns a DF
    """

    dfhits, dfhitHTs, dmaps = load_data(runs, sample_label)

    ddhs   = [cloudsdia_(dfh, dfhHT, dmap, ntotal) for dfh, dfhHT, dmap in zip(dfhits, dfhitHTs, dmaps)]

    nruns = len(runs)
    dfsum    = bes.df_concat([ddhs[0] for i in range(nruns)], runs)
    dfiso    = bes.df_concat([ddhs[1] for i in range(nruns)], runs)
    #key = 'evt_outcells'
    #print('ddh', ddh[key])

    return dfsum, dfiso


def cloudsdia_(dfhit, dfhitHT, maps, ntotal = 100000):
    """ City Engine: loops in events, run clouds and makes the summary
    """

    def hits_summary(x, y, z, eraw, erec, q):
        rmax = np.max(np.sqrt(x * x + y * y))
        idat = {'eraw'  : np.sum(eraw),
                'erec'  : np.sum(erec),
                'q'     : np.sum(q),
                'nhits' : len(x),
                'zmin'  : np.min(z),
                'zmax'  : np.max(z),
                'dz'    : np.max(z) - np.min(z),
                'rmax'  : np.max(np.sqrt(x * x + y * y))
                }
        return idat


    corrfac = get_corrfac(maps)

    nsize = len(dfhit.groupby('event'))
    print('size', nsize)

    labels  = ['event', 'eraw', 'erec', 'q', 'nhits', 'zmin', 'zmax', 'dz', 'rmax',
               'erawHT', 'erecHT', 'qHT', 'nhitsHT', 'zminHT', 'zmaxHT', 'dzHT', 'rmaxHT']
    labels += ['evt_ntracks', 'evt_nisos', 'evt_eisos', 'evt_ncells', 'evt_nnodes', 'evt_nrangs',
               'evt_ecells', 'evt_enodes', 'evt_erangs', 'evt_outcells', 'evt_outnodes', 'evt_outrangs',
               'evt_zmin', 'evt_zmax', 'evt_dz', 'evt_rmax', 'evt_enode1', 'evt_enode2',
               'trk_ncells', 'trk_nnodes', 'trk_nrangs', 'trk_ecells', 'trk_enodes', 'trk_erangs',
               'trk_outcells', 'trk_outnodes', 'trk_outrangs',
               'trk_zmin', 'trk_zmax', 'trk_dz', 'trk_rmax', 'trk_enode1', 'trk_enode2']

    dat = {}
    for label in labels:
        dat[label] = np.zeros(min(nsize, ntotal))

    dfiso = None

    n = -1
    for i, evt in dfhit.groupby('event'):

        n += 1
        if (n >= ntotal): continue

        dat['event'][n] = i

        # get HT hits info
        evtHT = dfhitHT.groupby('event').get_group(i)
        x, y, z, eraw, erec, q, times = get_hits(evtHT, ['X', 'Y', 'Z', 'E', 'Ec', 'Q', 'time'])
        idat = hits_summary(x, y, z, eraw, erec, q)
        for key in idat.keys():
            dat[key + 'HT'][n] = idat[key]

        # get hits info
        x, y, z, eraw, erec, q, times = get_hits(evt, ['X', 'Y', 'Z', 'E', 'Ec', 'Q', 'time'])
        idat = hits_summary(x, y, z, eraw, erec, q)
        for key in idat.keys():
            dat[key][n] = idat[key]

        if (n % 100 == 0):
            print('event : ', i, ', size : ', len(eraw))

        # clouds
        coors = (x, y, z)
        steps = (10., 10., 2.)
        dfclouds = clouds.clouds(coors, steps, eraw)
        in_cells = clouds.get_values_in_cells(coors, steps, eraw)
        dfclouds['erec'], _, _ = in_cells(coors, erec)
        dfclouds['q'], _, _    = in_cells(coors, q)
        dfclouds = cloud_calibrate(dfclouds, corrfac, times[0])

        ## info from clouds
        idat = cloud_summary(dfclouds)
        for key in idat.keys():
            dat[key][n] = idat[key]
        #key = 'evt_outcells'
        #print(key, idat[key], dat[key][n])

        # summary of isolated clouds
        idfiso = cloud_iso_summary(dfclouds)
        idfiso['event'] = i
        dfiso = idfiso if dfiso is None else pd.concat((dfiso, idfiso), ignore_index = True)

    dfsum = pd.DataFrame(dat)
    return dfsum, dfiso


def cloud_order_tracks(df):
    """ returns the ids of the tracks ordered by energy
    """

    rangers  = df.ranger .values
    erangers = df.eranger.values

    # get the largest energetic track
    sel      = rangers > -1
    kids     = np.unique(rangers[sel])
    enes     = np.array([np.sum(erangers[rangers == kid]) for kid in kids])
    kids, enes = clouds.sorted_by_energy(kids, enes)
    enes     = np.array(enes)
    return kids, enes


def cloud_summary(df):
    """ returns a summary of the cloud results (a dictionary)
    """

    # get data
    x        = df.x0     .values
    y        = df.x1     .values
    z        = df.x2     .values
    ecells   = df.ene    .values
    enodes   = df.enode  .values
    erangs   = df.eranger.values
    trackid  = df.track  .values
    rangers  = df.ranger .values
    #erangers = df.eranger.values
    nsize    = len(x)
    labels   = list(df.columns)
    cout     = df.cout   .values if 'cout' in labels else np.full(nsize, False)

    # get the largest energetic track
    kids, enes = cloud_order_tracks(df)
    kid_track_best = kids[0]
    ntracks        = len(kids)

    # compute isolated tracks
    nran_trk = np.array([np.sum(rangers == kid) for kid in kids])
    ksel     = nran_trk == 1
    nisos    = np.sum(ksel)  # number of isolated ranges
    eisos    = np.sum(enes[ksel]) # energy of the isolated ranges

    # general information about tracks and isolated tracks
    dvals = {'evt_ntracks' : ntracks,
             'evt_nisos'   : nisos,
             'evt_eisos'   : eisos}


    # selections
    enodes0   = df.enode0.values   if 'enodes0'   in labels else enodes
    erangs0   = df.eranger0.values if 'erangers0' in labels else erangs
    sel_nodes = enodes0 > 0.
    sel_rangs = erangs0 > 0.

    def _vals(sel = None):

        sel = np.full(nsize, True) if sel is None else sel

        ncells   = np.sum(sel)
        nnodes   = np.sum(sel & sel_nodes)
        nrangs   = np.sum(sel & sel_rangs)

        esumcells = np.sum(ecells[sel])
        esumnodes = np.sum(enodes[sel])
        esumrangs = np.sum(erangs[sel])

        outcells  = np.sum(cout[sel])
        outnodes  = np.sum(cout[sel & sel_nodes])
        outrangs  = np.sum(cout[sel & sel_rangs])

        zmin = np.min(z[sel])
        zmax = np.max(z[sel])
        dz   = zmax - zmin
        rmax = np.max(np.sqrt(x[sel]*x[sel] + y[sel]*y[sel]))

        xenodes = np.sort(enodes[sel & sel_nodes])
        enode1  = xenodes[-1]
        enode2  = xenodes[-2] if (len(xenodes) >= 2) else 0.

        vals = {'ncells'   : ncells,
                'nnodes'   : nnodes,
                'nrangs'   : nrangs,
                'ecells'   : esumcells,
                'enodes'   : esumnodes,
                'erangs'   : esumrangs,
                'outcells' : outcells,
                'outnodes' : outnodes,
                'outrangs' : outrangs,
                'zmin'     : zmin,
                'zmax'     : zmax,
                'dz'       : dz,
                'rmax'     : rmax,
                'enode1'   : enode1,
                'enode2'   : enode2}

        return vals

    # info from the event
    dval = _vals()
    for label in dval.keys():
        dvals['evt_' + label] = dval[label]

    # info from the best track
    kidbest = kids[0]
    dval = _vals(trackid == kid_track_best)
    for label in dval.keys():
        dvals['trk_' + label] = dval[label]

    #for label in dvals.keys():
    #    print(label, dvals[label])

    return dvals


def cloud_calibrate(df, corrfac, itime):

    x, y, z = df.x0.values, df.x1.values, df.x2.values

    nsize  = len(x)
    times  = itime * np.ones(nsize)
    cfac   = corrfac(x, y, z, times)
    cout   = np.isnan(cfac)
    cfac[cout] = 0.

    df['cfac']     = cfac
    df['cout']     = cout

    df['ene0']     = df['ene'].values[:]
    df['ene']      = cfac * df['ene'].values

    df['enode0']   = df['enode'].values[:]
    df['enode']    = cfac * df['enode'].values

    df['eranger0'] = df['eranger'].values[:]
    df['eranger']  = cfac * df['eranger'].values

    return df


def cloud_iso_summary(df):

    x        = df.x0     .values
    y        = df.x1     .values
    z        = df.x2     .values
    ecells   = df.ene    .values
    enodes   = df.enode  .values
    erangs   = df.eranger.values
    trackid  = df.track  .values
    rangers  = df.ranger .values

    cout     = df.cout   .values
    q        = df.q      .values
    erec     = df.erec   .values


    # order the tracks by energy
    kids, enes = cloud_order_tracks(df)

    # compute isolated tracks
    nran_trk = np.array([np.sum(trackid == kid) for kid in kids])

    ksel     = nran_trk == 1
    nisos    = np.sum(ksel)  # number of isolated ranges
    eisos    = np.sum(enes[ksel]) # energy of the isolated ranges

    #best track
    kid_track_best = kids[0]
    dz = np.max(z[trackid == kid_track_best]) - np.min(z[trackid == kid_track_best])

    ksel     = np.array(kids).astype(int)[ksel]
    #print(ksel)

    idat = {'x' : x[ksel],
            'y' : y[ksel],
            'z' : z[ksel],
            'q' : q[ksel],
            'erec': erec[ksel],
            'eraw': eraw[ksel],
            'out' : cout[ksel],
            'xb'  : np.ones(nisos) * x[kid_track_best],
            'yb'  : np.ones(nisos) * y[kid_track_best],
            'zb'  : np.ones(nisos) * z[kid_track_best],
            'dz'  : np.ones(nisos) * dz
           }

    return pd.DataFrame(idat)
