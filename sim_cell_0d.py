import argparse

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt


from ap_data_sets_paths import *

from cell import LUTActPotCurve, LUTRegresor, Cell

def get_initial_state(seg_stim, **kwargs):

    nr = len(seg_stim)
    ids = np.ones((nr,), dtype=bool)
    for k, v in kwargs.items():
        if k in seg_stim.columns:
            ids = ids & (seg_stim[k] == v).to_numpy()

    i0 = seg_stim.loc[ids, 'DI'].idxmax()

    return i0, seg_stim.loc[i0, 'AP+1']

def load_curve_pickles(cell_type):

    if cell_type == 'ttepi':
        curve_ds_path = EPI_HD_PATH
        curve_fv_path = EPI_P80_PATH
    elif cell_type == 'ttendo':
        curve_ds_path = ENDO_HD_PATH
        curve_fv_path = ENDO_P80_PATH
    elif cell_type == 'ttmid':
        curve_ds_path = MID_HD_PATH
        curve_fv_path = MID_P80_PATH
    elif cell_type == 'ttepibz':
        curve_ds_path = EPI_BZ_HD_PATH
        curve_fv_path = EPI_BZ_P80_PATH
    elif cell_type == 'ttendobz':
        curve_ds_path = ENDO_BZ_HD_PATH
        curve_fv_path = ENDO_BZ_P80_PATH
    elif cell_type == 'ttmidbz':
        curve_ds_path = MID_BZ_HD_PATH
        curve_fv_path = MID_BZ_P80_PATH
    else:
        print(f"Given cell_type arg not recognized ({cell_type}) , it must be in {{ttendo, ttmid, ttepi, ttendobz, ttmidbz, ttepibz}} ")
        return None, None

    print("Loading Cell DS from: ", curve_ds_path)
    curve_ds  = pd.read_pickle(curve_ds_path)
    curve_fvs = pd.read_pickle(curve_fv_path)

    return curve_ds, curve_fvs

def prepare_trainig_set(trset):

    AP = np.array(trset['AP'].tolist())
    DI = np.array(trset['DI']).reshape(-1,1)
    X = np.concatenate((DI, AP), axis=1)
    Y = np.array(trset['AP+1'].tolist())

    return X, Y

def prepare_curve_ds(curve_ds):
    t_delta = curve_ds.loc[0,'t_delta']
    curves_DS = np.array(curve_ds['AP+1'].tolist())
    return curves_DS, t_delta

class CellSim:

    def __init__(self, cell_type=None, CL0=600):

        self.id = None
        self.debug = False

        self.cell      : Cell   = None
        self.cell_type : str    = cell_type #{'ttepi', 'ttendo', 'ttmid', 'ttepibz', 'ttendobz', 'ttmidbz'}
        self.CL0       : float  = CL0  #The cycle length of the initial state of the myocyte expressed in ms

        #Time(ms) parameters
        self.t_ini : float = 0
        self.t_end : float = None
        self.dt    : float = 0.1

        self.act_times  : np.ndarray = None
        self.time_extra : float = 500

        self.times : np.ndarray = None
        self.ap    : np.ndarray = None

        self.__curve_fvs : pd.DataFrame = None
        self.__curve_ds  : pd.DataFrame = None
    #

    def restart(self, reload_pickles=True):

        if reload_pickles:
            self.cell        = None
            self.cell_type   = None
            self.CL0         = 600
        else:
            self.cell.restart()
            self.set_cell_initial_state()

        self.t_ini : float = 0
        self.t_end : float = None
        self.dt    : float = 0.1

        self.act_times  : np.ndarray = None
        self.time_extra : float = 500

        self.times : np.ndarray = None
        self.ap    : np.ndarray = None
    #

    def build_cell(self):

        self.__curve_ds, self.__curve_fvs = load_curve_pickles(cell_type=self.cell_type)

        X,Y = prepare_trainig_set(trset=self.__curve_fvs)
        state_regresor    = LUTRegresor(X=X, Y=Y, train=True)

        curves_DS, t_delta = prepare_curve_ds(curve_ds=self.__curve_ds)
        state_to_curve = LUTActPotCurve(curves_DS=curves_DS, curves_fvs=Y, t_delta=t_delta)

        self.cell = Cell(state_reg=state_regresor, act_pot_curve=state_to_curve)
        self.set_cell_initial_state()
    #

    def set_cell_initial_state(self, CLO=None):
        if CLO is not None:
            self.CL0=CLO
        _, s0 = get_initial_state(self.__curve_fvs, S1=self.CL0)
        self.cell.set_state(s0)
    #

    def load_act_times(self, fname, update_t_end=True):
        self.act_times = np.sort(np.load(fname))

        if update_t_end:
            self.infer_t_end_from_last_act_t()
    #

    def infer_t_end_from_last_act_t(self):
            self.t_end = self.act_times[-1] + self.time_extra
    #

    def run_simulation(self):

        delay=self.CL0

        #Allocate_space

        if self.times is not None:
            times = self.times
            self.t_end = times[-1]
        elif self.times is None and [self.t_ini, self.t_end, self.dt] != [None]*3:
            N = int((self.t_end-self.t_ini)/self.dt) + 1
            times = np.linspace(self.t_ini, self.t_end, N)
        else:
            print("ERROR: times, t_ini, t_end and dt are None.....")

        ap = np.empty(times.shape)
        APDs = []

        old_act_t = - delay
        for act_t in self.act_times:

            if act_t != self.act_times[0]:
                APDs.append(old_act_t + self.cell.apd)

            #Compute AP
            ids = np.argwhere((old_act_t <= times) & (times <= act_t)).ravel()
            t = times[ids]
            if t.any():
                if self.debug:
                    print("Procesing from ",t.min()," to ", t.max())
                ap[ids] = self.cell.eval_curve(t-old_act_t)

            #Compute di from act_t
            di = act_t - (self.cell.apd + old_act_t)

            #Update Cell state
            self.cell.get_next_state(di=di, update=True)
            old_act_t = act_t

            #From the last act_t to t_end
            ids = np.argwhere((old_act_t <= times) & (times <= self.t_end)).ravel()
            t = times[ids]
            if t.any():
                ap[ids] = self.cell.eval_curve(t-old_act_t)

            self.times = times
            self.ap    = ap
            self.APDs  = APDs
        if self.debug:
            self.plot_sim()
    #

    def plot_sim(self, ax=None, show=True):
        if ax is None:
            _, ax = plt.subplots()
        for apd in self.APDs:
            ax.axvline(apd, linestyle=':', color='red', alpha=0.5)
        for act_t in self.act_times:
            ax.axvline(act_t, linestyle='-.', color='blue', alpha=0.5)
        ax.plot(self.times,self.ap, 'k')
        if show:
            plt.show()
    #


def exec_cell_0D_simulation(cell_type,
                            act_t_fname,
                            cl0=600,
                            t_ini=0,
                            dt=0.1,
                            t_end=None,
                            ext=500,
                            dbg=False):

    sim = CellSim(cell_type=cell_type, CL0=cl0)
    sim.debug=dbg
    sim.build_cell()

    #Time(ms) parameters
    sim.t_ini      = t_ini
    sim.dt         = dt
    sim.time_extra = ext

    sim.load_act_times(act_t_fname)
    if t_end is not None:
        sim.t_end = t_end


    sim.run_simulation()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Script to create ELVIRA
                                    cases, using a template case.""",
                                    usage=""" """)

    parser.add_argument('-m',
                        '--myo',
                        dest='myo',
                        type=str,
                        help="""Flag to specify the cellular model.
                        It must be in {'ttepi','ttmid','ttendo', 'ttepibz','ttmidbz','ttendobz'} .""")

    parser.add_argument('-c',
                        '--cl0',
                        dest='cl0',
                        type=float,
                        default=800,
                        help="""The initial cycle length to be assumed.""")

    parser.add_argument('-i',
                        '--t-ini',
                        dest='t_ini',
                        type=float,
                        default=0.0,
                        help="""The time to start the simulation expressed in ms.""")

    parser.add_argument('-e',
                        '--t-end',
                        dest='t_end',
                        type=float,
                        default=5000,
                        help="""The time to end the simulation expressed in ms. We recommend to use the --ext argument instead of this one.""")

    parser.add_argument('--ext',
                        dest='ext',
                        type=float,
                        default=500,
                        help="""The extension of time added after the last activation time expressed in ms. This is the recommended way to set t_end.""")

    parser.add_argument('-m',
                        '--myo',
                        dest='myo',
                        type=str,
                        help="""Flag to specify the cellular model.
                        It must be in {'ttepi','ttmid','ttendo', 'ttepibz','ttmidbz','ttendobz'} .""")

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


    parser.add_argument('act_times_file',
                        action='store',
                        type=str,
                        nargs='?',
                        default=None,
                        help="""Activation times file. Currently only npy files are supported.""")



    args = parser.parse_args()
    exec_cell_0D_simulation(cell_type=args.myo, act_t_fname=args.act_t_fname)
