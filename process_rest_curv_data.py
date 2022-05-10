#!usr/bin/env/python3

import os

import numpy as np

from scipy.optimize import curve_fit

import pandas as pd

import argparse

import matplotlib.pyplot as plt
import seaborn as sns

def exp_(x, a, b, c):

    return a * (1 - b*np.exp(-x/c))


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

def make_APDRC_file(data, fname, debug=False, w=False):

    if os.path.exists(fname) and not w:
        print(f"Warning {fname} already exists and overwitting option is set to False. Nothing will be saved....")

    #Removing noise just in case .....
    #data = data.drop(data[(data.DI < 17) & (data['APD+1'] > 152)].index)
    #data = data.drop(data[(data.DI < 25) & (data['APD+1'] > 150)].index)
    #data = data.drop(data[data.DI < 10].index)
    data_df = data[['DI','APD+1']].drop_duplicates(subset='DI', keep="last")
    data_df = data_df[['DI','APD+1']].sort_values('DI')

    x = data_df['DI'].to_numpy()
    y = data_df['APD+1'].to_numpy()


    #p0 iteration was neded in some cases to converge, however it may a local minimum :|
    abc, _ = curve_fit(exp_, x, y, p0=[389.02417036, 0.5086576, 112.45328555])
    xx = np.linspace(min(5, x.min()), max(x.max(), 700), 200)

    if debug:
        print("a b c: ", abc )
        plt.plot(x, y, 'bo')
        plt.plot(xx, exp_(xx, *abc), 'k-')
        plt.show()


    save_arr = np.array((xx, exp_(xx, *abc))).T.astype(str)
    np.savetxt(fname, save_arr, fmt="""\"%s\"""", delimiter=',')
#

def make_CVRC_file(data, fname, debug=False, s=10, w=False):

    if os.path.exists(fname) and not w:
        print(f"Warning {fname} already exists and overwitting option is set to False. Nothing will be saved....")
        return False

    data['DI'] = data[['DI_or', 'DI_dest']].mean(axis=1)
    data = data.drop(data[(data.DI < 27) & (data['CV'] > 0.04)].index)
    data = data.drop(data[data.CV < 0].index)
    data = data.drop(data[(data.DI < 30) & (data['CV'] > 0.045)].index)
    data_df = data.drop_duplicates(subset='DI', keep="last")
    data_df = data_df.sort_values('DI')

    x = data_df['DI'].to_numpy()
    y = data_df['CV'].to_numpy()

    #p0 iteration was neded in some cases to converge, however it may a local minimum :|
    abc, _ = curve_fit(exp_, x, y, p0=[ 0.07130694, 0.91347783, 52.24539094])
    xx = np.linspace(min(5, x.min()), max(x.max(), 700), 200)

    if debug:
        print("a b c: ", abc )
        fig, ax = plt.subplots()
        sns.scatterplot(x='DI', y='CV', hue='node_dest', style='node_or', data=data_df, ax=ax)
        ax.plot(xx, exp_(xx, *abc), 'k-')
        plt.show()

    save_arr = np.array( (xx, exp_(xx, *abc) *s) ).T.astype(str)
    np.savetxt(fname, save_arr, fmt="""\"%s\"""", delimiter=',')
#

def compute_restitution_curves(path=None, APD_df=None, CV_df=None, cell_type=None, bz=False, w=False, debug=False):

    if path is None and APD_df is None and CV_df is not None:
        print("ERROR: Either path or data_df must be passed...")
        return

    if path:
        APD_df, CV_df = load_dataframes(path)

    if path is not None and APD_df is not None:
        b='Sanas'
        if bz:
            b='BZ'
        fname = f"{path}/Rest_Curve_data/RestitutionCurve_{b}_APD_{cell_type}.csv"
        make_APDRC_file(APD_df, fname, debug=debug, w=w)

    if path is not None and CV_df is not None:
        b='Sanas'
        s=10
        if bz:
            b='BZ'
            s*=0.25
        fname = f"{path}/Rest_Curve_data/RestitutionCurve_{b}_CV_{cell_type}.csv"
        make_CVRC_file(CV_df, fname, debug=debug, s=s, w=w)



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

    parser.add_argument('-d',
                        '--debug',
                        dest='debug',
                        action='store_true',
                        help="""Run in debug mode. Which essentialy is showing some plots...""")

    parser.add_argument('path',
                        action='store',
                        type=str,
                        nargs='?',
                        help="""Path to an existing case of cases.""")


    args = parser.parse_args()

    compute_restitution_curves(path=args.path, cell_type=args.myo, bz=args.bz, w=args.w, debug=args.debug)
