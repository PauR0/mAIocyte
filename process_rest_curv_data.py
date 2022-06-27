#!usr/bin/env/python3

import os, sys

import argparse

import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import linregress

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def exp_(x, a, b, c):
    return a * (1 - b*np.exp(-x/c))

def exp_2(x, a, b, c, d):
    return a*np.exp(b*x)+ c*np.exp(d*x)
    #return a*np.exp(b*(i*x-h))+c*np.exp(d*(i*x-h)) + g

def poly_rat(x, a, b, c, d, h, k):
    return (a* (x-k)**2+b) / (c*(x-k)+d) + h

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

def load_data(path, ext='csv', rebuild=False, w=False):

    APD_data = load_APD_data(path=path, ext=ext, rebuild=rebuild, w=w)
    CV_data  = load_CV_data( path=path, ext=ext, rebuild=rebuild, w=w)

    return APD_data, CV_data
#

def load_APD_data(path, ext='csv', rebuild=False, w=False):

    col_dtypes = {'S1':int, 'S2':int, 'APD':float, 'DI':float , 'APD+1':float, 'cell_type':str, 'node_id':int}

    APD_data = load_df_files(path, DS_fname='/APD_DS.csv',  col_dtypes=col_dtypes, key='APD_DI_APD+1', rebuild=rebuild, ext=ext)

    return APD_data
#

def load_CV_data(path, ext='csv', rebuild=False, w=False):

    col_dtypes = {'S1':int, 'S2':int, 'DI_or':float , 'DI_dest':float, 'CV':float, 'cell_type':str, 'node_or':str, 'node_dest':str}

    CV_data = load_df_files(path, DS_fname='CV_DS.csv', col_dtypes=col_dtypes, key='DI_CV', rebuild=rebuild,  ext=ext, w=w)

    return CV_data
#

def load_df_files(path, DS_fname, col_dtypes, key, rebuild=False, ext='csv', abs_path=False, w=False):

    df_dir = f'{path}/Rest_Curve_data'
    if not abs_path:
        DS_fname = f"{df_dir}/{DS_fname}"

    if os.path.exists(DS_fname) and not rebuild:
        df = pd.read_csv(DS_fname, dtype=col_dtypes)

    else:
        df_names = [f for f in os.listdir(df_dir) if f.startswith(key) and f.endswith(ext)]
        dfs = [pd.read_csv(f'{df_dir}/{name}', dtype=col_dtypes) for name in df_names]
        df  = pd.DataFrame(columns=dfs[0].columns)
        df  = df.append(dfs, ignore_index=True)
        if not os.path.exists(DS_fname) or w:
            df.to_csv(DS_fname)
        elif os.path.exists(DS_fname):
            print(f"{DS_fname} wont be overwritten")

    return df
#

def make_APDR_curve(data, fname=None, debug=False, w=False):

    if fname:
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
    xx = np.linspace( x.min(), x.max(), 200)

    if debug:
        res = y - exp_(x, *abc)
        print("APD exp params: ", abc )
        print(f"Residuals: mean {res.mean()}, std {res.std()}, max {res.max()}")
        _, [ax1, ax2] = plt.subplots(1,2, sharex=True)
        ax1.scatter(x, y, label='samples')
        ax1.plot(xx, exp_(xx, *abc), 'k-', label='model')
        ax2.plot(x, res, 'r', label='residuals')
        plt.show()


    if fname:
        save_arr = np.array((xx, exp_(xx, *abc))).T.astype(str)
        np.savetxt(fname, save_arr, fmt="""\"%s\"""", delimiter=',')

    return abc
#

def make_CVR_curve(data, fname=None, debug=False, s=10, w=False):

    """"
    if fname:
        if os.path.exists(fname) and not w:
            print(f"Warning {fname} already exists and overwitting option is set to False. Nothing will be saved....")
            return False
    """

    data['DI'] = data[['DI_or', 'DI_dest']].mean(axis=1)
    data['CV'] = data['CV'].abs()
    #data = data.drop(data[(data.DI < 27) & (data['CV'] > 0.04)].index)
    #data = data.drop(data[data.CV < 0].index)
    #data = data.drop(data[(data.DI < 30) & (data['CV'] > 0.045)].index)
    #data_df = data.drop_duplicates(subset='DI', keep="last")
    #data_df = data_df.sort_values('DI')
    data_df = data

    x = data_df['DI'].to_numpy()
    y = data_df['CV'].to_numpy()

    #p0 iteration was neded in some cases to converge, however it may a local minimum :|
    abc, _ = curve_fit(exp_, x, y, p0=[ 0.07130694, 0.91347783, 52.24539094])
    xx = np.linspace(min(5, x.min()), max(x.max(), 700), 200)

    if debug:
        print("a b c: ", abc )
        _, ax = plt.subplots()
        sns.scatterplot(x='DI', y='CV', hue='S1', data=data_df, ax=ax)
        ax.plot(xx, exp_(xx, *abc), 'k-')
        plt.show()

    if fname:
        save_arr = np.array( (xx, exp_(xx, *abc) *s) ).T.astype(str)
        np.savetxt(fname, save_arr, fmt="""\"%s\"""", delimiter=',')
#

def compute_restitution_curves(path=None, APD_df=None, CV_df=None, cell_type=None, bz=False, w=False, debug=False):

    if path is None and APD_df is None and CV_df is not None:
        print("ERROR: Either path or data_df must be passed...")
        return

    if path:
        APD_df, CV_df = load_data(path)

    if path is not None and APD_df is not None:
        b='Sanas'
        if bz:
            b='BZ'
        fname = f"{path}/Rest_Curve_data/RestitutionCurve_{b}_APD_{cell_type}.csv"
        make_APDR_curve(APD_df, fname=fname, debug=debug, w=w)

    if path is not None and CV_df is not None:
        b='Sanas'
        s=10
        if bz:
            b='BZ'
            s*=0.25
        fname = f"{path}/Rest_Curve_data/RestitutionCurve_{b}_CV_{cell_type}.csv"
        make_CVR_curve(CV_df, fname=fname, debug=debug, s=s, w=w)
#

def make_APDR_surface(data, fname=None, debug=False, w=False):

    apd   = data['APD'].to_numpy()
    di    = data['DI'].to_numpy()
    apd_1 = data['APD+1'].to_numpy()

    abc = make_APDR_curve(data, debug=False)

    apd_1_diff = apd_1 - exp_(di, *abc)
    linreg = linregress(apd, apd_1_diff)

    aapdd = np.linspace(apd.min(), apd.max(), 100)
    ddii = np.linspace(di.min(), di.max(), 100)


    if debug:
        print("APDRS exp params: ", abc )
        print(f"APDRS linear params: slope {linreg.slope} intrcpt {linreg.intercept} r2 {linreg.rvalue**2} p value {linreg.pvalue}")
        #The p-value for a hypothesis test whose null hypothesis is that the slope is zero, using Wald Test with t-distribution of the test statistic

        X, Y = np.meshgrid(ddii, aapdd)
        Z = exp_(X, *abc) + (linreg.slope * Y + linreg.intercept)

        fig = plt.figure()
        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132, projection='3d')
        ax2 = fig.add_subplot(133)
        ax0.scatter(di, apd_1, c=apd_1)
        ax0.plot(ddii, exp_(ddii, *abc), 'k')

        ax1.scatter(di, apd, apd_1)#, c=data['S1'])
        ax1.plot_surface(X, Y, Z, linewidth=0, cmap='viridis', antialiased=False, alpha=0.5)

        sns.scatterplot(x='APD', y=apd_1_diff, data=data, ax=ax2, hue='S1', legend=True)#palette='viridis',
        ax2.scatter(aapdd, (linreg.slope * aapdd + linreg.intercept))
        plt.show()

    if fname:
        save_arr = np.zeros( (ddii.shape[0]+1, aapdd.shape[0]+1))
        save_arr[0,1:] = ddii
        save_arr[1:,0] = aapdd
        save_arr[1:,1:] = [exp_(d, *abc) + (linreg.slope * aapdd + linreg.intercept) for d in ddii]

        save_arr = save_arr.astype(str)
        np.savetxt(fname, save_arr, fmt="\"%s\"", delimiter=',')
#

def make_CVR_surface(data, fname=None, debug=False, w=False):


    data['DI'] = data[['DI_or', 'DI_dest']].mean(axis=1)
    #apd  = data['APD'].to_numpy()
    di  = data['DI'].to_numpy()
    cv  = data['CV'].abs().to_numpy()

    abc, _ = curve_fit(exp_, xdata=di, ydata=cv, p0=[0.07130694, 0.91347783, 52.24539094])

    cv_res = cv - exp_(di, *abc)
    #linreg = linregress(apd, cv_res)
    ddii = np.linspace(di.min(), di.max(), 200)
    #aapdd = np.linspace(apd.min(), apd.max(), 200)


    if debug:

        #X, Y = np.meshgrid(ddii, aapdd)
        #Z = exp_(X, *abc) + (linreg.slope * Y + linreg.intercept)

        fig = plt.figure()
        ax0 = fig.add_subplot(121)
        #ax1 = fig.add_subplot(132, projection='3d')
        ax2 = fig.add_subplot(122)

        ax0.scatter(di, cv, s=5)
        ax0.plot(ddii, exp_(ddii, *abc), 'k')

        #ax1.plot_surface(X, Y, Z, linewidth=0, cmap='viridis', antialiased=False, alpha=0.5)

        sns.scatterplot(x='S1', y=cv_res, hue='S1', data=data, ax=ax2)
        #ax2.scatter(aapdd, (linreg.slope * aapdd + linreg.intercept))
        plt.show()
#

def compute_restitution_surfaces(path, cell_type=None, w=False, debug=False):

    APD_data, CV_data = load_data(path)

    make_APDR_surface(data=APD_data, debug=debug, w=w)
    make_CVR_surface(data=CV_data, debug=debug, w=w)
#

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Processing S1S2 data for generating restitution curves """,
                                    usage = """ """)

    parser.add_argument('-m',
                        '--myo',
                        dest='myo',
                        default='',
                        type=str,
                        help="""Flag to specify the cell model, must be in {ttepi, ttmid, ttendo, ttepibz, ttmidbz, ttendobz}""")

    parser.add_argument('--APDRC',
                        dest='APDRC',
                        action='store_true',
                        help=""" Generate APD restiturion curve.""")

    parser.add_argument('--APDRS',
                        dest='APDRS',
                        action='store_true',
                        help="""Generate APD restiturion surface.""")

    parser.add_argument('--CVRC',
                        dest='CVRC',
                        action='store_true',
                        help="""Generate conduction velocity restiturion curve.""")

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

    APD=None
    CV=None

    if args.myo.startswith('ttendo'):
        cell_type = 'Endo'
    elif args.myo.startswith('ttepi'):
        cell_type = 'Epi'
    elif args.myo.startswith('ttmid'):
        cell_type = 'Mid'
    else:
        print("WARNING --myo argument could not be understood...")
        sys.exit(1)

    s=10
    aux='Sanas'
    if 'bz' in args.myo.lower():
        s*=0.25
        aux='BZ'

    if args.APDRC:
        APD = load_APD_data(path=args.path, ext='csv', rebuild=False, w=args.w)
        fname = f"{args.path}/Rest_Curve_data/RestitutionCurve_{aux}_APD_{cell_type}.csv"
        make_APDR_curve(data=APD, fname=fname, debug=args.debug, w=args.w)

    if args.APDRS:
        if APD is None:
            APD = load_APD_data(path=args.path, ext='csv', rebuild=False, w=args.w)
        fname = f"{args.path}/Rest_Curve_data/RestitutionSurface_{aux}_APD_{cell_type}.csv"
        make_APDR_surface(data=APD, fname=fname, debug=args.debug, w=args.w)

    if args.CVRC:
        CV = load_CV_data(path=args.path, ext='csv', rebuild=False, w=args.w)
        fname = f"{args.path}/Rest_Curve_data/RestitutionCurve_{aux}_CV_{cell_type}.csv"
        make_CVR_curve(CV, fname=fname, debug=args.debug, s=s, w=args.w)
#
