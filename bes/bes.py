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


def get_ranges():

    ranges = {}
    ranges['energy']    = (0., 3.)
    ranges['energy_cs'] = (0.65, 0.71)
    ranges['energy_ds'] = (1.55, 1.75)
    ranges['energy_ph'] = (2.5, 3.)

    ranges['z']  = (50., 500.)
    ranges['r']  = ( 0., 180.)

    return ranges


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


def dft_selections(dft):

    tsels = Selections()
    tsels['onetrack']      = dft.numb_of_tracks == 1

    tsels['fidutial_zmin'] = dft.z_min > 50.
    tsels['fidutial_zmax'] = dft.z_max < 500.
    tsels['fidutial_rmax'] = dft.r_max < 180.
    tsels['fidutial']      = tsels.logical_and('fidutial_zmin', 'fidutial_zmax', 'fidutial_rmax')

    tsels['dz_cs'] = (dft.dz_track >  8.) & (dft.dz_track <  32.)
    tsels['dz_ds'] = (dft.dz_track > 24.) & (dft.dz_track <  72.)
    tsels['dz_ph'] = (dft.dz_track > 35.) & (dft.dz_track < 130.)

    return tsel
