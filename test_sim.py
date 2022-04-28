import sys, os
import pandas as pd
import numpy as np
import pyvista as pv
from vtk import VTK_HEXAHEDRON
import matplotlib.pyplot as plt


from sim_cell_0d import Simulation0D


def geom_from_elvira_case(case_dir):

    nodes     = np.loadtxt(case_dir+'/data/nodes.dat', skiprows=2, usecols=(-3,-2,-1))
    elem      = (np.loadtxt(case_dir+'/data/elements.dat', skiprows=2) - 1)[:,2:].astype(int)
    c0        = np.ones((elem.shape[0],1))*8
    elems     = np.concatenate( (c0, elem), axis=1).astype(int)
    off       = np.arange(0, elem.shape[0])*9
    elem_type = np.array([VTK_HEXAHEDRON]*elems.shape[0])

    return pv.UnstructuredGrid(off, elems, elem_type, nodes)

if __name__ == '__main__':
    elv_case_output = sys.argv[1]

    j = elv_case_output.rfind('output')
    elv_case = elv_case_output[:j]

    print("Working with", elv_case)
    act_times_path = elv_case_output+'/act_times.pickle'
    all_act_times = pd.read_pickle(act_times_path)
    skp= 3800  #4000
    end= 13799 #13000


    ref_node = 773
    ref_node_i =  all_act_times.index[all_act_times['node_id'] == str(ref_node)]



    print("Cell: ",all_act_times.loc[0,'cell_type'])
    sim = Simulation0D()
    sim.debug=False

    aux={}
    for i, row in all_act_times.iterrows():

        nid = row['node_id']
        elv_sim_path = f'{elv_case_output}/node_{nid}.npy'
        if i % 200 == 0:
            print(f"Iteration {i} out of {all_act_times.shape[0]}")
            print("---------------------------------------------")

        elv_sim = np.load(elv_sim_path)[skp:end]
        if i==0:
            SIM     = np.zeros(shape=(len(all_act_times), elv_sim.shape[0]))
            DIFF    = np.zeros(shape=(len(all_act_times), elv_sim.shape[0]))

        if sim.cell_type != row['cell_type']:
            sim.cell_type = row['cell_type']
            sim.build_cell()
        else:
            sim.restart(reload_pickles=False)

        #Time(ms) parameters
        #sim.t_ini      = 0
        #sim.dt         = 0.2
        #sim.time_extra = 50
        sim.times = elv_sim[:,0]

        act_t_ = row['act_t']
        sim.act_times = act_t_
        sim.run_simulation()

        ap_dif = elv_sim[:,1] - sim.ap

        if nid in ['1050','2104']:
            aux[f'AP_{nid}'] = sim.ap
            aux[f'DIF_{nid}'] = ap_dif

        SIM[int(nid)-1] = sim.ap
        DIFF[int(nid)-1] = ap_dif


        if False:#nid in ['1051','2105']:
            fig,ax = plt.subplots()
            #for act_t in sim.act_times:
            #    ax.axvline(act_t, linestyle='-.', color='blue', alpha=0.5)
            ax.set_xlabel("t(ms)",fontsize=14)
            ax.set_ylabel("AP(mV)",color="black",fontsize=14)
            ax.plot(sim.times, elv_sim[:,1], 'k-' , label='Biophisical')
            ax.plot(sim.times, sim.ap,       'c:', linewidth=3, label='prediction')
            ax.plot(sim.times, ap_dif, color="red", label='difference')
            ax.fill_between(sim.times, sim.ap, elv_sim[:,1], color='red', alpha=0.5, label='_difference')
            ax.legend()
            plt.show()
            input()

    for i in ['1051','2105']:
        SIM[int(nid)] = sim.ap
        DIFF[int(nid)] = ap_dif

    np.save(elv_case+'/AP_SIM.npy', SIM)
    np.save(elv_case+'/DIFF_AP_SIM.npy', DIFF)
    #SIM = np.load(elv_case+'/AP_SIM.npy')
    pv.set_plot_theme("paraview")
    pv.global_theme.colorbar_orientation = 'vertical'
    geom = geom_from_elvira_case(elv_case)
    for select_t in [7970, 7990, 8140]:
        it = np.argmin(np.abs(sim.times - select_t)).ravel()[0]
        print("it", it)
        geom[f'AP_{select_t}'] = SIM[:,it]
        geom['DIF_'+str(select_t)] = DIFF[:,it]
    geom.save(elv_case+'/geom_sim.vtk')
