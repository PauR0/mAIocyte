#/usr/bin/env python

import numpy as np

import pandas as pd

from init import load_trset

from cell import Cell, LUTRegresor, LUTActPotCurve

import matplotlib.pyplot as plt

def get_initial_state(seg_stim, **kwargs):

    nr = len(seg_stim)
    ids = np.ones((nr,), dtype=bool)
    for k, v in kwargs.items():
        if k in seg_stim.columns:
            ids = ids & (seg_stim[k] == v).to_numpy()

    i0 = seg_stim.loc[ids, 'DI'].idxmax()

    return i0, seg_stim.loc[i0, 'AP+1']


if __name__ == '__main__':

    HD_dataset_path = "../../ELVIRA/EPI/seg_stimuli/AP_DS.pickle"
    trset_path      = "../../ELVIRA/EPI/seg_stimuli/AP_P8.pickle"
    #trset_path      = "../../ELVIRA/EPI/seg_stimuli/AP_LD_50.pickle"


    seg_stim_HD = pd.read_pickle(HD_dataset_path)
    curves_DS = np.array(seg_stim_HD['AP+1'].tolist())
    t_delta = seg_stim_HD.loc[0, 't_delta']
    X, Y, seg_stim_ld = load_trset(trset_path=trset_path, full_output=True)


    p8_to_curve = LUTActPotCurve(curves_DS=curves_DS, curves_fvs=Y, t_delta=t_delta)
    p8_lut_regr  = LUTRegresor(X=X, Y=Y, train=True)
    P8_cell = Cell(state_reg=p8_lut_regr, act_pot_curve=p8_to_curve)


    CL0 = 800
    i0, s0 = get_initial_state(seg_stim_ld, S1=CL0, node_id='176')
    P8_cell.set_state(s0)

    s1_per_s2 = 9
    S1 = 350
    S2 = 280
    act_times = (np.arange(0,s1_per_s2)*S1).tolist() + [S1*(s1_per_s2-1) + S2]
    delay=CL0
    t_ini = 0
    t_max = 5300
    dt=0.2
    N = int((t_max-t_ini)/dt)

    times = np.linspace(t_ini, t_max, N)
    ap = np.empty(times.shape)
    APDs = []

    old_act_t = - delay
    for act_t in act_times:

        if act_t != act_times[0]:
            APDs.append(old_act_t + P8_cell.apd)

        print(P8_cell)
        #Compute AP
        ids = np.argwhere((old_act_t <= times) & (times <= act_t)).ravel()
        t = times[ids]
        if t.any():
            print("Procesing from ",t.min()," to ", t.max())
            ap[ids] = P8_cell.eval_curve(t-old_act_t)

        #Compute di from act_t
        di = act_t - (P8_cell.apd + old_act_t)

        #Update Cell state
        P8_cell.get_next_state(di=di, update=True)
        old_act_t = act_t

    #From the last act_t to t_max
    ids = np.argwhere((old_act_t <= times) & (times <= t_max)).ravel()
    t = times[ids]
    if t.any():
        ap[ids] = P8_cell.eval_curve(t-old_act_t)

    for apd in APDs:
        plt.axvline(apd, linestyle=':', color='red', alpha=0.5)

    for act_t in act_times:
        plt.axvline(act_t, linestyle='-.', color='blue', alpha=0.5)
    plt.plot(times, ap, 'k')
    plt.show()