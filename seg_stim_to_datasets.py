import os

import json

import argparse

import colorsys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline, splrep, splev

from PCA import STDPCA

from sens_elv_exp import compute_max_der_and_perc_repolarization
from peaks import peakdetect



def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors
#

def load_files(path, abs_path=False, save=True):

    """
    TODO:Documentation
    """

    if not abs_path:
        path += '/seg_stimuli'
        fname = path + '/AP_DS.pickle'

    if path.endswith('.csv'):
        fname=path

    if True:#not os.path.exists(fname):

        data = []
        max_stim=np.array([])

        for f in [f for f in os.listdir(path) if f.endswith('.json')]:

            with open(path+'/'+f, 'r', encoding='utf-8') as jf:
                stim = json.load(jf)
            stim['AP'] = np.array(stim['AP'])
            stim['AP+1'] = np.array(stim['AP+1'])

            if stim['AP'].shape[0] > max_stim.shape[0]:
                max_stim = stim['AP']

            if stim['AP+1'].shape[0] > max_stim.shape[0]:
                max_stim = stim['AP+1']

            data.append(stim)

        data = normalize_dimensions(data, max_stim, debug=False)
        data = pd.DataFrame(data)

        if save:
            data.to_pickle(fname)

    else:
        data = pd.read_pickle(fname)

    return data
#

def normalize_dimensions(data, max_stim, fields=None, debug=False):
    """
    Arguments:
    -----------

        data : list
            List containing the values of

        max_stim : np.ndarray
            The array containing the longest signal
    """

    if fields is None:
        fields=['AP', 'AP+1']

    max_length = max_stim.shape[0]
    max_stim_spl = UnivariateSpline(x=np.arange(0,max_length,step=1), y=max_stim, k=3, s=0, ext='const')

    for st in data:
        if st['node_id'] == '176' and st['S1'] == 350 and st['S2'] == 290:
            plt.rcParams.update({'font.size': 18})
            _, [ax1, ax2] = plt.subplots(1, 2, sharey=True)
            ax1.plot(np.arange(0, len(st['AP']))*st['t_delta'], st['AP'], 'k-', label='AP')
            ax1.set_title('AP & Activation Time')
            ax1.axvline(st['t_act'], color='red', linestyle='--', label='act_time')
            ax2.set_title('Next AP')
            ax2.plot(np.arange(0, len(st['AP+1']))*st['t_delta'], st['AP+1'], 'k-', label='next AP')
        for f in fields:
            l = st[f].shape[0]
            if l < max_length:
                new_arr = np.zeros( (max_length,) ) # Allocate the space
                new_arr[:l] = st[f] # Insert the signal

                #To fill the 0s, we get the points from the longest stimulus
                i = np.argmin(np.abs(max_stim - st[f][-1]))
                x=np.linspace(i, max_length, max_length-l)
                new_arr[l:] = max_stim_spl(x)
                st[f] = new_arr
            else:
                st[f] = st[f][:max_length]
        if st['node_id'] == '176' and st['S1'] == 350 and st['S2'] == 290:
            plt.rcParams.update({'font.size': 18})
            ap_v8 = feature_vec9(st['AP'], t_delta=st['t_delta'])
            ap_v8 = np.concatenate((ap_v8[:4].reshape(-1,1),ap_v8[4:].reshape(-1,1)), axis=1)
            ax1.plot(np.arange(0, len(st['AP']))*st['t_delta'], st['AP'], color='k',linestyle=':', label='_AP')
            ax1.plot(ap_v8[:,0], ap_v8[:,1], 'go', color='g', ls='None', marker='.', label='P8')

            ax2.plot(np.arange(0, len(st['AP+1']))*st['t_delta'], st['AP+1'], color='k',linestyle=':', label='_next AP')
            ap_v8 = feature_vec9(st['AP+1'], t_delta=st['t_delta'])
            ap_v8 = np.concatenate((ap_v8[:4].reshape(-1,1),ap_v8[4:].reshape(-1,1)), axis=1)
            ax2.plot(ap_v8[:,0], ap_v8[:,1], color='g', ls='None', marker='.', label='P8')
            plt.tight_layout()
            ax1.legend()
            plt.show()
        if debug:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
            fig.suptitle(f"S1 {st['S1']}  S2 {st['S2']}  cell type {st['cell_type']}  node id {st['node_id']}")
            ax1.set_title('AP')
            ax1.plot(np.arange(0, len(st['AP'])) * st['t_delta'], st['AP'])
            ax1.axvline(st['t_act'], color='k')
            ax2.set_title('AP+1')
            ax2.plot(np.arange(0, len(st['AP+1'])) * st['t_delta'], st['AP+1'])
            plt.show()
            c = input('Continue showing?(*/n)')
            if c.lower() == 'n':
                debug=False

    return data
#

def high_to_low_to_high(pca, ap, dim, ap_mean=None, t_delta=0.02):

    """
    Plot a given AP curve in the original dimensionality, and then plot it after having reduced it.
    It could by like:
    AP -> M(AP) -> M^-1(M(AP))
    This is done for the desired dims. it can be a single one, or a list of posible dimensionalities.

    Arguments:

        pca : STDPCA
            The standard scaler pca object

        ap : np.ndarray
            The action potential samples

        dim : int, list
            The dimension or dimensions to be displayed (bigger than 0)

        t_delta : float
            The space between ap


    """

    if isinstance(dim, int):
        dim = [dim]

    fig, axes = plt.subplots(len(dim)+2, 1, sharex=True)
    x = np.arange(0, ap.shape[0]) * t_delta
    cmap = plt.cm.viridis
    md = min(dim)
    Md = max(dim)

    if ap_mean is not None:
        axes[0].plot(x, ap_mean, color='r', label = 'mean')
    axes[0].plot(x, ap, 'k')
    axes[-1].plot(x, ap, 'k', label='original')

    for i, d in enumerate(dim):
        ap_tr = pca.transfrom(ap.reshape(1,-1), n_dim=d)
        ap_rec = pca.inv_transform(ap_tr.reshape(1,-1))
        c=cmap( (d-md)/(Md-md) )
        if ap_mean is not None:
            axes[i+1].plot(x, ap_mean, color='r')
        axes[i+1].plot(x, ap_rec, color=c)
        axes[i+1].set_title(', '.join(['{:.2f}'.format(c) for c in ap_tr]))
        axes[-1].plot(x, ap_rec, color=c, label=str(d))
        axes[-1].set_ylabel('AP (mV)')

    if ap_mean is not None:
        axes[-1].plot(x, ap_mean, color='r', label = 'mean')
    axes[-1].set_xlabel('t (ms)')
    axes[-1].legend(loc='lower right')
    plt.show()
#

def plot_curves(stimuli_df):
    """
    Make a plot with all the stimuli of the db

    """

    f, (ax1,ax2) = plt.subplots(2,1)
    x = np.arange(0, stimuli_df.loc[0,'AP'].shape[0]) * stimuli_df.loc[0,'t_delta']
    mS1 = stimuli_df['S1'].min()
    MS1 = stimuli_df['S1'].max()

    mS2 = stimuli_df['S2'].min()
    MS2 = stimuli_df['S2'].max()

    for _, stim in stimuli_df.iterrows():
        c=plt.cm.viridis((stim['S1']-mS1)/(MS1-mS1))
        ax1.plot(x, stim['AP'], color=c )
        #c=plt.cm.viridis((stim['S2']-mS2)/(MS2-mS2))
        ax2.plot(x, stim['AP+1'], color=c)
    plt.show()
#

def t_act_to_di(seg_stim):

    DI = []
    t = np.array([])
    for i, r in seg_stim.iterrows():
        if t.shape != r['AP'].shape:
            t = np.arange(0, r['AP'].shape[0]) * r['t_delta']
        _, apd90 = compute_max_der_and_perc_repolarization(t, r['AP'], perc=0.9)
        di = r['t_act'] - apd90
        DI.append(di)

    return DI
#

def make_sample_pickle(seg_stim, tsplits, ppsplits, fname=None):
    """

    TODO: Documentation

    """

    t_delta = seg_stim.loc[0,'t_delta']

    times = np.unique( np.hstack( [np.linspace(tsplits[i], tsplits[i+1], ppsplits[i]) for i in range(len(tsplits)-1)]))
    ids = (times/t_delta).astype(int)

    f = lambda x : x[ids]
    extractor = lambda x : x.map(f) if x.name in ['AP', 'AP+1'] else x

    ld_stim = seg_stim.copy()
    ld_stim['DI'] = t_act_to_di(ld_stim)
    ld_stim = ld_stim.apply(extractor)

    ld_stim['times']    = ld_stim['AP'].apply(lambda x: times)
    ld_stim['tsplits']  = ld_stim['AP'].apply(lambda x: tsplits)
    ld_stim['ppsplits'] = ld_stim['AP'].apply(lambda x: ppsplits)

    if fname:
        ld_stim.to_pickle(fname)

    return ld_stim
#

def apply_spca(seg_stimulus):


    #Processing AP
    AP = np.array(seg_stimulus['AP'].tolist())
    AP_1  = np.array(seg_stimulus['AP+1'].tolist())
    X_pca = np.concatenate( (AP,AP_1), axis=0)
    X_pca = X_pca[np.random.randint(0,X_pca.shape[0], 100)]

    ap_mean = X_pca.mean(axis=0)
    for ap in X_pca:
        plt.plot(range(0,X_pca.shape[1]), ap)
    plt.plot(range(0,X_pca.shape[1]), ap_mean, 'k-')
    plt.show()
    pca = STDPCA(n_dim=5)
    pca_info_fname = path=f'{path}/seg_stimuli/pca_info.npy'
    if not os.path.exists(pca_info_fname):
        pca.fit_X(X=X_pca, transform_X=True)
        #pca.save_pca_data(pca_info_fname, abs_path=True)
        pca.plot_pca_variance()
    else:
        pca.load_pca_data(pca_info_fname, abs_path=True)
        #pca.set_X(X_pca, transform_X=True)

    m = seg_stimulus.sort_values(by='S1', ascending=True).head(n=1)
    high_to_low_to_high(pca, m['AP+1'].to_numpy()[0], range(1,6), ap_mean=ap_mean, t_delta=m['t_delta'].to_numpy())

    M = seg_stimulus.sort_values(by='S1', ascending=False).head(n=1)
    high_to_low_to_high(pca, M['AP+1'].to_numpy()[0], range(1,6), ap_mean=ap_mean, t_delta=M['t_delta'].to_numpy())


    sigmas = np.concatenate( (pca.get_X_tr().min(axis=1), pca.get_X_tr().max(axis=1)), axis=0).T
    pca.ppal_dirs(sigmas=sigmas, show=True,
                  x=np.arange(0, AP.shape[1])*seg_stimulus.loc[0,'t_delta'])
    AP_tr = pca.get_X_tr()[:AP.shape[0]]
    AP_1_tr = pca.get_X_tr()[AP.shape[0]:]

    return pca, AP_tr, AP_1_tr
#

def make_stim_data_sets(path):

    """

    TODO: document

    """

    seg_stimulus = load_files(path)

    p8_name = f"{path}/seg_stimuli/AP_P8.pickle"

    make_P8_pickle(seg_stim=seg_stimulus, fname=p8_name)

    tsplits = [0, 10, 30, 250, 450, 1000]
    ppsplit = [20, 5, 10, 40, 5]

    ld_name = f"{path}/seg_stimuli/AP_LD_{sum(ppsplit)}.pickle"
    ld_stim = make_sample_pickle(seg_stim=seg_stimulus, tsplits=tsplits, ppsplits=ppsplit, fname=ld_name)

    """
    ids = (ld_stim['S1'] > 300) & (ld_stim['S1'] < 350) & (ld_stim['node_id'] == '168')
    sel_stim = seg_stimulus.loc[ids, :]
    _, [ax1,ax2] = plt.subplots(1,2)
    cs = _get_colors(len(sel_stim.index))

    j = 0
    for i, r in sel_stim.iterrows():
        t = np.arange(0,r['AP'].shape[0])*r['t_delta']
        ax1.plot(t, r['AP'],   c=cs[j], label=r['S1'])
        ax2.plot(t, r['AP+1'], c=cs[j], label=ld_stim.loc[i, 'DI'])
        j+=1
    ax1.legend()
    ax2.legend()
    plt.show()

    ids = (ld_stim['DI'] > 200) & (ld_stim['DI'] < 210) & (ld_stim['node_id'] == '168')
    sel_stim = seg_stimulus.loc[ids, :]
    _, [ax1,ax2] = plt.subplots(1,2)
    cs = _get_colors(len(sel_stim.index))
    j=0
    for i, r in sel_stim.iterrows():
        t = np.arange(0,r['AP'].shape[0])*r['t_delta']
        ax1.plot(t, r['AP'],   c=cs[j], label=r['S1'])
        ax2.plot(t, r['AP+1'], c=cs[j], label=ld_stim.loc[i,'DI'])
        j+=1
    ax1.legend()
    ax2.legend()
    plt.show()
    """
#

def compute_an(x, y, T, n, dx=None):
    wn = n * 2*np.pi / T
    if dx is None:
        dx = np.gradient(x)
    return 2/T * np.sum( (y * np.cos( x * wn ) * dx) )
#

def compute_bn(x, y, T, n, dx=None):
    wn = n * 2*np.pi / T
    if dx is None:
        dx = np.gradient(x)
    return 2/T * np.sum( (y * np.sin( x * wn ) * dx) )
#

def compute_fourier_coeffs(x, y, N, T=None):
    """
    Compute the first N fourier coefficients of a periodic signal.
    TODO:DOCUMENT PROPERLY
    """

    if T is None:
        T=x.min()-x.max()

    dx = np.gradient(x)

    an = [compute_an(x, y, T, 0, dx=dx)]
    bn = [0]

    for n in range(1, N):
        an.append(compute_an(x, y, T, n, dx=dx))
        bn.append(compute_bn(x, y, T, n, dx=dx))

    return np.array(an), np.array(bn)
#

def fourier_eval(t, an, bn, T):
    return an[0]/2 + np.sum( [an[n]*np.cos(t * n * 2 * np.pi / T) + bn[n]*np.sin(t * n * 2 * np.pi / T) for n in range(1, an.size)], axis=0)
#

def explore_fourier_coeff_rep(path):
    """

    Let's try Fourier's series and its coefficients

    """


    stimuli_df = load_files(path)

    AP_1  = np.array(stimuli_df['AP+1'].tolist())

    some = AP_1[np.random.randint(0,AP_1.shape[0], 3)]

    x = np.arange(0, stimuli_df.loc[0,'AP'].shape[0]) * stimuli_df.loc[0,'t_delta']
    T = x[-1]
    t = np.linspace(0,T, 100)
    terms = 15

    for ap in some:
        _, ax = plt.subplots(1,1)
        ax.plot(x, ap, 'k', label='f')
        an, bn = compute_fourier_coeffs(x, ap, terms, T)
        ax.plot(x[[0,-1]], [an[0]/2]*2, 'k', label='f')

        for n in range(2, terms+1):
            ap_appr = fourier_eval(t, an[:n], bn[:n], T)
            ax.plot(t, ap_appr, label=f'$f_{{{n}}}$')
        plt.legend()
        plt.show()
#

def get_function_control_points(knots, coeff, k, padded = True):

    """
    Compute the control points (t*_j,c_j)_j=1^n of a spline function
    where c_j are the coefficients and t*_j is the average position
    of the knots for a given coefficient.
            t*_j = (t_{j+1}+...+t_{j+k}) / k,     1<=j<=n

    This functions assumes the coeff array/list to be paded with k+1 zero.
    This is due to scipy

    Args:
    -----
        knots : np.array/list of float
            The knot vector of the given spline function
        coeff : np.array/list of float
            The coefficients of the given spline function
        k : int
            Degree of the given spline function
        padded : bool (optional)
            Default True. Whether the coeff array contains k+1 trailing zeros.

    Returns:
    ---------
        control_points : np.array
    """

    if padded: coeff = coeff[:-(k+1)]
    t_ = [np.mean(knots[i+1 : i+k+1]) for i in range(len(coeff))]
    control_points = np.array((t_, coeff))
    return control_points
#

def fit_piecewise_curve(t, ap, t_delta=0.02, t_splits=None, n_pts_splits=None, k=3, debug=True):

    """
    Plot a piecewise interpolation of the ap curve.
    """


    if not isinstance(t, np.ndarray):
        t = np.ndarray(t)
    if not isinstance(ap, np.ndarray):
        ap = np.ndarray(ap)

    if t_splits is None:
        t_splits = [0, 10, 30, 250,400, 1000]
    if n_pts_splits is None:
        n_pts_splits = [20, 5, 10, 10, 5]

    ts = np.unique( np.hstack( [np.linspace(t_splits[i], t_splits[i+1], n_pts_splits[i]) for i in range(len(t_splits)-1)]))
    ids = (ts/t_delta).astype(int)
    print(np.unique(t[ids]).shape, ts.shape)
    tck = splrep(ts, ap[ids], k=3, s=0)
    #ttt = np.hstack(([ttt[0]]*(k+1), ttt[2:-2], [ttt[-1]]*(k+1)))
    print("coeff: ", tck[1])
    print("ap: ", ap[ids])
    ap_appr = splev(t, tck)

    plt.plot(t, ap, c='k', label='ap')
    plt.plot(t[ids], ap[ids], 'ko')
    plt.plot(t, ap_appr, c='r', label=f'spline curve')
    plt.show()

    input()
#

def explore_spline_rep(path):
    """

    Let's explore the multiple options offered by spline interpolation.

    """


    stimuli_df = load_files(path)

    AP_1  = np.array(stimuli_df['AP+1'].tolist())

    some = AP_1[np.random.randint(0,AP_1.shape[0], 3)]
    x = np.arange(0, stimuli_df.loc[0,'AP'].shape[0]) * stimuli_df.loc[0,'t_delta']

    k=3

    for ap in some:
        t = np.arange(0, ap.shape[0]) * stimuli_df.loc[0,'t_delta']
        fit_piecewise_curve(t, ap, t_splits=None)
#

def feature_vec9(ap_i, t_act=None, t_delta=None, apdx=0.9):

    tt = np.arange(0,len(ap_i))*t_delta
    _max, _min = peakdetect(ap_i, tt )

    xmax = [p[0] for p in _max]
    ymax = [p[1] for p in _max]
    xmin = [p[0] for p in _min]
    ymin = [p[1] for p in _min]

    _, [t, apd90] = compute_max_der_and_perc_repolarization(t=tt, ap=ap_i, perc=apdx, full_output=True)


    # Señal AP: 2 maximos, un minimo y el apd90
    if (len(xmin) == 1 and  len(xmax) == 2):
        if t_act != None:
            vec9 = [t_act-t, xmax[0], xmin[0], xmax[1], t, ymax[0], ymin[0], ymax[1], apd90]
            return np.array(vec9)
        else:
            vec8 = [xmax[0], xmin[0], xmax[1], t, ymax[0], ymin[0], ymax[1], apd90]
            return np.array(vec8)
    else:
      vec8 = np.zeros(8)
      t_max = np.argmax(ap_i)
      max = ap_i[t_max]

      for i in range(4):
          vec8[i] = t_max + i*1000
          vec8[i+4] = ap_i[t_max+ i*1000]

      return vec8
#

def make_P8_pickle(seg_stim, fname):


    ld_stim = seg_stim.copy()
    ld_stim['DI'] = t_act_to_di(ld_stim)

    f = lambda x : feature_vec9(x, t_delta=ld_stim.loc[0,'t_delta'])
    extractor = lambda x : x.map(f) if x.name in ['AP', 'AP+1'] else x
    ld_stim = ld_stim.apply(extractor)

    if fname:
        ld_stim.to_pickle(fname)

    return ld_stim
#




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=""" Module for building consistent datasets from segmented signals. """,
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

    make_stim_data_sets(path=args.path)
    #explore_fourier_coeff_rep(path=args.path)
    #explore_spline_rep(path=args.path)
