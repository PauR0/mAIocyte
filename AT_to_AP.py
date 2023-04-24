#! /usr/bin/env python3

import os, sys
import re
import argparse
import datetime as dt
from multiprocessing import Process


import numpy as np
import pyvista as pv
from tqdm import tqdm

from arrithmic3D_case_reader import A3DENS
from CellSim import CellSim, build_multiple_cells
from param_utils import read_AP_json, write_AP_json
from utils import pad_list_to_array

def load_cell_types_from_mesh(case_dir):

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

    mesh = pv.read(f"{case_dir}/ventricle_Tagged.vtk")
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

def load_act_times(act_times_file):

    path, name_ext = os.path.split(act_times_file)
    name, ext = os.path.splitext(name_ext)

    if ext.lower() == '.case':
        ens = A3DENS()
        ens.read_ens_case_file(act_times_file)
        ens.load_variables(var='Activation_Map(ms)')
        act_times = ens_to_act_times(ens).T

    elif ext == '.txt':
        with open(act_times_file,'r') as f:
            f = f.read()

        f_rep = re.sub(r'FloatList\s+size\=\d+\s', '', f).splitlines()
        lat = [eval(l) for l in tqdm(f_rep, desc="Parsing activation times.")]
        act_times = pad_list_to_array(lat, fillval=np.nan)

    elif ext.lower() == '.npy':
        act_times = np.load(act_times_file)

    return act_times
#

def make_sim_dir(case_dir, act_times_file=None, sim_id=None, f=False):

    if sim_id is None:
        if act_times_file is not None:
            name, _ = os.path.splitext(act_times_file)
            i = name.rfind('_')+1
            sim_id = name[i:]
        else:
            now = dt.datetime.now()
            sim_id = now.strftime('%d_%m_%Y')

    sim_dir = f"{case_dir}/{sim_id}/"

    if os.path.exists(sim_dir) and not f:
        print(f"ERROR: Directory {sim_dir} already exists. To overwrite it try with -f option.")
        return None, None
    elif os.path.exists(sim_dir) and f:
        for f in os.listdir(sim_dir):
            if "process" in f and f.endswith('.npy'):
                os.remove(f"{sim_dir}/{f}")
        return sim_dir, sim_id
    else:
        os.makedirs(sim_dir, exist_ok=True)
        return sim_dir, sim_id
#

def set_up_sim_times(ap_params, last_act_time):
    """
    Function to set up proper simulation times, which
    should be expressed in ms. It follows the following logic:

    If t_ini is negative it is set as the last_act_time - t_ini.
    If t_max is None it is set as the last activation + t_extra.

    Arguments:
    ------------

        ap_params : dict
            The parameters passed to perform the simulation

        last_act_time : float
            The last activation read from the activation times file.

    Returns:
    -----------

        ap_params : dict
            The updated parameter dictionary.
    """

    if ap_params['data']['t_ini'] < 0:
        ap_params['data']['t_ini'] = last_act_time + ap_params['data']['t_ini']

    if ap_params['data']['t_end'] is None:
        ap_params['data']['t_end'] = last_act_time + ap_params['data']['t_extra']

    return ap_params
#

def get_chunk_size(size, n_proc):
    if size % n_proc == 0:
        return size // n_proc
    return size // (n_proc - 1)
#

def run_sim(args):

    n, chunk_size, cell_sims, times, act_times, at_ids, sim_dir = args

    ini, end = n*chunk_size, min(len(cell_sims), (n+1)*chunk_size)
    AP = np.full(shape=(end - ini, times.shape[0]), fill_value=-1e6, dtype='float64')

    text = f"Process #{n} from {ini} to {end}"
    for nid, cs in enumerate(tqdm(cell_sims[ini:end], desc=text, position=n, leave=False)):
        if cs is not None:
            cs.set_time_array(times)
            cs.act_times = act_times[ini+nid, at_ids[ini+nid]]
            cs.run_simulation()
            AP[nid,:] = cs.ap
    np.save(f'{sim_dir}/process_{n}.npy', AP)
    print(f"Process #{n} from {ini} to {end} Finished!")
#

def assemble_parallel_sims(sim_dir, shape, rm_proc=True):

    print("---------------------")
    print(f"Assembling simulation: total shape {shape}")
    AP = np.memmap(f"{sim_dir}/AP.npy", dtype='float64', mode='w+', shape=shape)
    parts = sorted([f for f in os.listdir(sim_dir) if "process" in f and f.endswith('.npy')], key = lambda x: int(x.replace("process_","").replace(".npy", "")))

    print("---------------------")
    total_cells_nan = 0
    ini, end = 0, -1
    pb = tqdm(parts)
    for i, part in enumerate(pb):
        part_ap = np.load(f"{sim_dir}/{part}")
        ids_nan = np.isnan(part_ap)
        end = ini + part_ap.shape[0]
        pb.set_description(f"-Part {i} : {part} from {ini} to {end}")
        if ids_nan.any():
            n_cells_nan = np.unique(np.argwhere(ids_nan)[:,0])
            print(f"WARNING: {ids_nan.sum()} NaNs have been detected at cells: {n_cells_nan}")
            total_cells_nan += n_cells_nan
        AP[ini:end] = part_ap
        AP.flush()
        ini = end
    print("---------------------")
    if total_cells_nan:
        print(f"Warning: {total_cells_nan} cells contined nans.")
    if rm_proc:
        [os.remove(f"{sim_dir}/{part}") for par in parts]


def AT_to_AP(case_dir,
             ap_params,
             act_times_file,
             f=False):

    act_times = load_act_times(act_times_file=act_times_file)
    at_ids = ~ np.isnan(act_times)

    print("-"*20)
    print(f"ACT_TIMES:  shape {act_times.shape}; max {act_times[at_ids].max()}; min {act_times[at_ids].min()}; nans {(~at_ids).sum()}")
    print("-"*20)
    cell_types = load_cell_types_from_mesh(case_dir)

    types_set = np.unique(cell_types)
    print(f"Cell types description:")
    for ct in types_set:
        print(f"\t- Type {ct}: n {(cell_types == ct).sum()}")
    print("-"*20)
    cell_sims = build_multiple_cells(cell_types, types_set=types_set)

    ap_params = set_up_sim_times(ap_params, last_act_time=act_times[at_ids].max())

    t_ini = ap_params['data']['t_ini']
    t_delta = ap_params['data']['t_delta']
    t_end = ap_params['data']['t_end']
    print(f"Action Potentials will be simulated from {t_ini} to {t_end} with a dt {t_delta}")

    N = int((t_end-t_ini)/t_delta)
    times = np.linspace(t_ini, t_end, N)

    now = dt.datetime.now()
    ap_params['metadata']['date'] = now.strftime('%d-%m-%Y')
    ap_params['metadata']['time'] = now.strftime('%H:%M')
    ap_params['data']['shape'] = (len(cell_types), times.shape[0])

    sim_dir, sim_id = make_sim_dir(case_dir=case_dir, act_times_file=act_times_file, sim_id=ap_params['data']['sim_id'], f=f)

    if sim_dir is None:
        return

    if ap_params['data']['sim_id'] is None:
        ap_params['data']['sim_id'] = sim_id

    write_AP_json(sim_dir, data=ap_params)
    np.save(f"{sim_dir}/act_times.npy", act_times)
    chunk_size = get_chunk_size(len(cell_sims), ap_params['data']['n_proc'])


    procs = []
    for i in range(ap_params['data']['n_proc']):
        p = Process(target=run_sim, args=((i, chunk_size, cell_sims, times, act_times, at_ids, sim_dir),))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    assemble_parallel_sims(sim_dir=sim_dir, shape=(len(cell_types), times.shape[0]), rm_proc=False)
#

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Script to generate the Action potetials
                                                    from a simulated case. """)

    parser.add_argument('-f',
                        '--force',
                        dest='f',
                        action='store_true',
                        help=""" Overwrite existing files.""")

    parser.add_argument('-d',
                        '--debug',
                        dest='debug',
                        action='store_true',
                        help="""Plot the available data and print stuff.""")

    parser.add_argument('-a',
                        '--act-times-file',
                        dest='act_times',
                        action='store',
                        default=None,
                        type=str,
                        nargs='?',
                        help="""Path to the file containing the activation
                        times.""")

    parser.add_argument('-p',
                        '--AP-params',
                        dest='ap_params',
                        action='store',
                        default=None,
                        type=str,
                        nargs='?',
                        help="""Path to the file with the AP.""")




    parser.add_argument('case_dir',
                        action='store',
                        type=str,
                        nargs='?',
                        help="""Path to a directory containing the ventricle_Tagged.vtk
                        and the corresponding config files.""")





    args = parser.parse_args()

    if args.case_dir is not None:
        if not os.path.exists(args.case_dir) or not os.path.isdir(args.case_dir):
            print("ERROR: Given case_dir does not exist or is not a valid directory")
            sys.exit()

    if args.act_times is not None:
        if not os.path.exists(args.act_times) or not os.path.isfile(args.act_times):
            print("ERROR: Given act_times file does not exist or is not a file")
            sys.exit()

    if args.ap_params is not None:
        ap_params = read_AP_json(path=args.ap_params, abs_path=True)
    else:
        ap_params = read_AP_json(path=args.case_dir)
    AT_to_AP(case_dir=args.case_dir,
             ap_params=ap_params,
             act_times_file=args.act_times,
             f=args.f)