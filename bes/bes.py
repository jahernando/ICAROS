import numpy  as np
import pandas as pd
import tables as tb


to_df = pd.DataFrame.from_records


def energy_correction(energy, dz, a = 2.76e-4):
    return energy/(1 - a * dz)


def load_dfs(filename):
    f = tb.open_file(filename, 'r')

    dfe = to_df(f.root.DST.Events.read())
    dfs = to_df(f.root.Summary.Events.read())
    dft = to_df(f.root.Tracking.Tracks.read())


    dft['dz_track'] = dft['z_max'] - dft['z_min']
    dft['enecor']   = energy_correction(dft['energy'].values, dft['dz_track']. values)

    return dfe, dfs, dft


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
    ranges['dz_tracks.ph'] = (35., 130.)

    return ranges


def get_selections(dft, ranges, selections = None):

    selections = bes.Selections() if selections is None else selections

    for key in ranges.keys():
            var = key.split('.')[0]

            if var in dft.columns:
                selections[key] = ut.in_range(dft[var], ranges[key])

    return selections
