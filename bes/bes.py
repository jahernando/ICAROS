import numpy  as np
import pandas as pd
import tables as tb

import hipy.utils as ut


#---- DataFrame

to_df = pd.DataFrame.from_records


def energy_correction(energy, dz, a = 2.76e-4):
    """ Apply Josh's energy correction by delta-z effect
    """
    return energy/(1 - a * dz)


def load_dfs(filename):
    """ load DF from esmeralda:
    input:
        filename: str, complete filename
    returns:
        dfe, DF, event data frame
        dfs, DF, summary data frame
        dft, DF, tracks data frame
    """
    f = tb.open_file(filename, 'r')

    dfe = to_df(f.root.DST.Events.read())
    dfs = to_df(f.root.Summary.Events.read())
    dft = to_df(f.root.Tracking.Tracks.read())


    dft['dz_track'] = dft['z_max'] - dft['z_min']
    dft['enecor']   = energy_correction(dft['energy'].values, dft['dz_track']. values)

    return dfe, dfs, dft


def get_dft(filenames):
    """ return the track DF from Esmeralda filenames
    inputs:
        filenames: tup(str), list of complete Esmeralda's filenames
    returns:
        dft: DF, track data frame
    """

    dft = None

    for i, filename in enumerate(filenames):

        print('data filename: ', filename)

        idfe, idfs, idft = load_dfs(filename)

        dft = idft if i == 0 else dft.append(idft, ignore_index = True)

    return dft

def get_df_zeffect(filename):

    dfe, dfs, dft = load_dfs(filename)

    dfe_section = dfe[['event', 'time', 'nS2', 'S1e', 'S2e', 'S2q', 'Nsipm']]
    dfs_section = dfs[['event', 'evt_energy', 'evt_ntrks', 'evt_nhits']]
    dft_section = dft[['event', 'trackID', 'energy', 'length', 'numb_of_voxels',
                       'numb_of_hits', 'numb_of_tracks',
                       'x_min', 'y_min', 'z_min', 'r_min', 'x_max', 'y_max', 'z_max', 'r_max',
                       'x_ave', 'y_ave', 'z_ave', 'r_ave',
                       'extreme1_x', 'extreme1_y', 'extreme1_z',
                       'extreme2_x', 'extreme2_y', 'extreme2_z',
                       'blob1_x', 'blob1_y', 'blob1_z', 'blob2_x',
                       'blob2_y', 'blob2_z', 'eblob1', 'eblob2',
                       'ovlp_blob_energy', 'dz_track', 'enecor']]

    dd = pd.merge(dft_section, dfe_section, on = 'event')
    dd = pd.merge(dd         , dfs_section, on = 'event')

    return dd

#----  Selections


class Selections(dict):
    """ dictorinay to hold selection (np.array(bool)) with some extendions:
    """


    def logical_and(self, *names):
        """ return the selection that is the logical and of the names selections
        names: list of names of the selections
        """

        assert len(names) >= 2
        name0, name1 = names[0], names[1]
        sel = self[name0] & self[name1]

        for name in names[2:]:
            sel = sel & self[name]

        return sel


def get_ranges():

    ranges = {}

    ranges['numb_of_tracks.one']   = (0.5, 1.5)

    ranges['nS2'] = (0.5, 1.5)

    ranges['energy']    = (0., 3.)

    ranges['energy.cs'] = (0.65, 0.71)
    ranges['energy.ds'] = (1.55, 1.75)
    ranges['energy.ph'] = (2.5, 3.)

    ranges['enecor.cs'] = (0.65, 0.71)
    ranges['enecor.ds'] = (1.55, 1.75)
    ranges['enecor.ph'] = (2.5, 3.)

    ranges['z_min']  = (50., 500.)
    ranges['z_max']  = (50., 500.)
    ranges['r_max']  = ( 0., 180.)

    ranges['dz_track.cs']  = ( 8., 32.)
    ranges['dz_track.ds']  = (24., 72.)
    ranges['dz_track.ph'] = (35., 130.)

    return ranges


def get_selections(dft, ranges, selections = None):

    selections = Selections() if selections is None else selections

    for key in ranges.keys():
            var = key.split('.')[0]

            if var in dft.columns:
                selections[key] = ut.in_range(dft[var], ranges[key])

    return selections