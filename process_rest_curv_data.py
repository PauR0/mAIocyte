#!usr/bin/env/python3

import os

import numpy as np

import pandas as pd

import argparse

import matplotlib.pyplot as plt
import pyvista as pv

def griddata(x, y, z, binsize=0.01, retbin=True, retloc=True):
    """
    https://scipy-cookbook.readthedocs.io/items/Matplotlib_Gridding_irregularly_spaced_data.html
    Place unevenly spaced 2D data on a grid by 2D binning (nearest
    neighbor interpolation).

    Parameters
    ----------
    x : ndarray (1D)
        The idependent data x-axis of the grid.
    y : ndarray (1D)
        The idependent data y-axis of the grid.
    z : ndarray (1D)
        The dependent data in the form z = f(x,y).
    binsize : scalar, optional
        The full width and height of each bin on the grid.  If each
        bin is a cube, then this is the x and y dimension.  This is
        the step in both directions, x and y. Defaults to 0.01.
    retbin : boolean, optional
        Function returns `bins` variable (see below for description)
        if set to True.  Defaults to True.
    retloc : boolean, optional
        Function returns `wherebins` variable (see below for description)
        if set to True.  Defaults to True.

    Returns
    -------
    grid : ndarray (2D)
        The evenly gridded data.  The value of each cell is the median
        value of the contents of the bin.
    bins : ndarray (2D)
        A grid the same shape as `grid`, except the value of each cell
        is the number of points in that bin.  Returns only if
        `retbin` is set to True.
    wherebin : list (2D)
        A 2D list the same shape as `grid` and `bins` where each cell
        contains the indicies of `z` which contain the values stored
        in the particular bin.

    Revisions
    ---------
    2010-07-11  ccampo  Initial version
    """
    # get extrema values.
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # make coordinate arrays.
    xi      = np.arange(xmin, xmax+binsize, binsize)
    yi      = np.arange(ymin, ymax+binsize, binsize)
    xi, yi = np.meshgrid(xi,yi)

    # make the grid.
    grid           = np.zeros(xi.shape, dtype=x.dtype)
    nrow, ncol = grid.shape
    if retbin: bins = np.copy(grid)

    # create list in same shape as grid to store indices
    if retloc:
        wherebin = np.copy(grid)
        wherebin = wherebin.tolist()

    # fill in the grid.
    for row in range(nrow):
        for col in range(ncol):
            xc = xi[row, col]    # x coordinate.
            yc = yi[row, col]    # y coordinate.

            # find the position that xc and yc correspond to.
            posx = np.abs(x - xc)
            posy = np.abs(y - yc)
            ibin = np.logical_and(posx < binsize/2., posy < binsize/2.)
            ind  = np.where(ibin == True)[0]

            # fill the bin.
            bin = z[ibin]
            if retloc: wherebin[row][col] = ind
            if retbin: bins[row, col] = bin.size
            if bin.size != 0:
                binval         = np.median(bin)
                grid[row, col] = binval
            else:
                grid[row, col] = np.nan   # fill empty bins with nans.

    # return the grid
    if retbin:
        if retloc:
            return grid, bins, wherebin
        else:
            return grid, bins
    else:
        if retloc:
            return grid, wherebin
        else:
            return grid
#

def plot_data(data_df):

    x = data_df['APD'].to_numpy()
    y = data_df['DI'].to_numpy()
    z = data_df['APD+1'].to_numpy()

    binsize = 1
    grid, bins, binloc = griddata(x, y, z, binsize=binsize)
    zmin    = grid[np.where(np.isnan(grid) == False)].min()
    zmax    = grid[np.where(np.isnan(grid) == False)].max()

    palette = plt.matplotlib.colors.LinearSegmentedColormap('jet3',plt.cm.datad['jet'],2048)
    palette.set_under(alpha=0.0)

    extent = (x.min(), x.max(), y.min(), y.max()) # extent of the plot
    plt.imshow(grid, extent=extent, cmap=palette, origin='lower', vmin=zmin, vmax=zmax, aspect='auto', interpolation='bilinear')
    plt.xlabel('APD')
    plt.ylabel('DI')
    plt.title('APD+1')
    plt.colorbar()
    plt.show()
#

def load_dataframes(path, ext='.csv'):

    df_dir = f'{path}/Rest_Curve_data'

    df_names = [f for f in os.listdir(df_dir) if f.endswith(ext)]


    APD_names = [f for f in df_names if f.startswith('APD_DI_APD+1')]
    col_dtypes = {'S1':int, 'S2':int, 'APD':float, 'DI':float , 'APD+1':float, 'cell_type':str, 'node_id':int}
    dfs = [pd.read_csv(f'{df_dir}/{name}', dtype=col_dtypes) for name in APD_names]
    APD_df = pd.DataFrame(columns=dfs[0].columns)
    APD_df = APD_df.append(dfs, ignore_index=True)

    col_dtypes = {'S1':int, 'S2':int, 'DI_or':float , 'DI_dest':float, 'CV':float, 'cell_type':str, 'node_or':str, 'node_dest':str}
    CV_names = [f for f in df_names if f.startswith('DI_CV')]
    dfs = [pd.read_csv(f'{df_dir}/{name}', dtype=col_dtypes) for name in CV_names]
    CV_df = pd.DataFrame(columns=dfs[0].columns)
    CV_df = CV_df.append(dfs, ignore_index=True, )

    return APD_df, CV_df
#


def compute_restitution_curves(path=None, data_df=None):

    if path is None and data_df is not None:
        print("ERROR: Either path or data_df must be passed...")
        return

    if path:
        data_df = load_dataframes(path)

    x = data_df[['APD', 'DI', 'APD+1']].to_numpy()

    p = pv.Plotter()
    p.add_mesh(x, scalars = x[:,2])
    p.add_axes()
    p.show()

    #plot_data(data_df)



if __name__ == '__main__':

    if __name__ == '__main__':

        parser = argparse.ArgumentParser(description="""Processing S1S2 data for generating restitution curves """,
                                        usage = """ """)

        parser.add_argument('-m',
                            '--myo',
                            dest='myo',
                            default='ttendo',
                            type=str,
                            help="""Flag to specify if ttepi - ttmid - ttendo. Default ttendo.""")

        parser.add_argument('-b',
                            '--border-zone',
                            dest='bz',
                            action='store_true',
                            help="""Set the node type to Border Zone.""")

        parser.add_argument('-w',
                            '--overwrite',
                            dest='w',
                            action='store_true',
                            help=""" Overwrite existing files.""")

        parser.add_argument('path',
                            action='store',
                            type=str,
                            nargs='?',
                            help="""Path to an existing case of cases.""")


        args = parser.parse_args()

        compute_restitution_curves(path=args.path)

