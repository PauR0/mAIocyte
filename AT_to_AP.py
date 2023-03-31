#! /usr/bin/env python3

import os, sys
import datetime as dt
import numpy as np
import pyvista as pv
from tqdm import tqdm

from arrithmic3D_case_reader import A3DENS
from CellSim import CellSim
from param_utils import read_AP_json, write_AP_json

def load_cell_types_from_mesh(fname):

    """
        'EndoToEpi':
            0 : ttendo
            1 : ttmid
            2 : ttepi

        'Cell_type':
            0 : healthy
            1 : BZ
            2 : core
    """

    mesh = pv.read(fname)
    cell_types = np.empty( (mesh.points.shape[0],), dtype=object)
    cell_types[mesh['EndoToEpi']==0] = 'ttendo'
    cell_types[mesh['EndoToEpi']==1] = 'ttmid'
    cell_types[mesh['EndoToEpi']==2] = 'ttepi'
    cell_types[mesh['Cell_type']==1] += 'bz'
    cell_types[mesh['Cell_type']==2] = 'core'

    return cell_types
#

def build_cell_models(cellmodels):

    cell_models = {}
    for ct in cellmodels:
        if ct != 'core':
            cell_models[ct] = CellSim(cell_type=ct)
            cell_models[ct].build_cell()

    return cell_models
#

def ens_to_act_times(ens):

    return np.array(ens.fields['Activation_Map(ms)']['data'])
#

def AT_to_AP(enscase_fname,
             mesh_fname,
             ap_params):


    path, _ = os.path.split(enscase_fname)

    ens = A3DENS()
    ens.read_ens_case_file(enscase_fname)
    ens.load_variables(var='Activation_Map(ms)')
    act_times = ens_to_act_times(ens)

    cell_types = load_cell_types_from_mesh(mesh_fname)
    cell_models = build_cell_models(np.unique(cell_types))

    t_ini = ap_params['data']['t_ini']
    t_delta = ap_params['data']['t_delta']
    t_extra = ap_params['data']['t_extra']

    if ap_params['data']['t_end'] is None:
        t_end = act_times.max() + t_extra
        ap_params['data']['t_end'] = t_end
    N = int((t_end-t_ini)/t_delta)
    times = np.linspace(t_ini, t_end, N)

    save_freq = 1000
    now = dt.datetime.now()
    ap_params['metadata']['date'] = now.strftime('%d-%m-%Y')
    ap_params['metadata']['time'] = now.strftime('%H:%M')
    ap_params['data']['shape'] = (len(cell_types), times.shape[0])
    write_AP_json(path, data=ap_params)

    AP = np.memmap(f"{path}/AP.npy", dtype='float64', mode='w+', shape=(len(cell_types), times.shape[0]) )
    for nid, ct in enumerate(tqdm(cell_types)):
        if ct != 'core':
            cell = cell_models[ct]
            cell.times = times
            cell.act_times = act_times[:,nid]
            cell.run_simulation()
            AP[nid,:] = cell.ap
            if nid % save_freq == 0:
                AP.flush()
#

if __name__ == '__main__':

    case_dir = "/Users/pau/Electrophys/Automata/p2_Bivent"
    ens_dir  = "A_Map_Berruezo_p2_Bivent_IDPacing-39016_6S1-600_1S2-285"
    enscase_fname = f"{case_dir}/{ens_dir}/VentA_Map_Berruezo_p2_Bivent.case"
    mesh_fname = f"/Users/pau/Electrophys/Automata/p2_Bivent/ventricle_Tagged.vtk"

    ap_params = read_AP_json()

    AT_to_AP(enscase_fname=enscase_fname, mesh_fname=mesh_fname, ap_params=ap_params)