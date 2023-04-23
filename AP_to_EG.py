#/usr/bin/env python3

import os, sys

import argparse

import datetime as dt

import numpy as np

import pyvista as pv

import matplotlib.pyplot as plt

from tqdm import trange
from scipy.spatial import KDTree

from param_utils import read_AP_json, read_EG_json, write_EG_json


class Electrode:

    def __init__(self, name, loc):

        self.name : str = name
        self.loc  : np.ndarray = loc

        self.d    : np.ndarray = None
        self.EG  : list = None

        self.t_ini : float = None
        self.t_end : float = None
        self.t_delta : float = None
    #

    def __str__(self):

        nt = 0
        if self.EG is not None:
            nt = len(self.EF)

        return F"""Electrode: {self.name}:
                        props:
                            location: {self.loc}
                            Time instants computed: {nt}

                        Time parameters:
                            t_ini = {self.t_ini}
                            t_end = {self.t_end}
                            t_delta = {self.t_delta}
                """
    #

    def compute_distance_vectors(self, mesh, normalize=False):

        self.d = mesh.points - self.loc
        self.d3 = np.power(np.linalg.norm(self.d, axis=1), 2/3)
        self.Mr = (self.d.T * 1 / self.d3).T
    #

    def compute_EG(self, mesh):
        """
        Method to compute the electrogram at a cetain instant.

        Arguments:
        ----------
            mesh : pv.PolyData
                The mesh used to build the distances but with a vector field
                defined at each point, called 'AP_grad' containing the gradient
                of the Action Potential.

        \Phi_e(r_e, t) = - 1/(4\pi\sigma_e) ∫\sigma(r) nablaV_m(r_jT) \gamma_d/r_d^3 dr

        P. Podziemski, P. Kuklik, A. van Hunnik, S. Zeemering, B. Maesen and U. Schotten,
        "Far-field effect in unipolar electrograms revisited: High-density mapping of atrial fibrillation in humans,"
        2015 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC),
        2015, pp. 5680-5683, doi: 10.1109/EMBC.2015.7319681.
        """

        if self.EG is None:
            self.EG = []

        self.EG.append((self.Mr * mesh['AP_grad']).sum())
    #

    def plot(self, ax=None, show=False):

        if ax is None:
            f, ax = plt.subplots()
        if self.EG is not None:
            t = self.t_ini + np.arange(len(self.EG)) * self.t_delta
            ax.plot(t, self.EG)
            ax.set_title(self.name)

        if show:
            plt.show()
    #
#

def load_sim_path(sim_path):

    ap_params = read_AP_json(path=sim_path)

    if os.path.exists(f'{sim_path}/AP.npy'):
        AP = np.memmap(f'{sim_path}/AP.npy', dtype=float, mode='r', shape=tuple(ap_params['data']['shape']))
    else:
        print(f"ERROR: simulation file AP.npy could be found at {sim_path}")

    return AP, ap_params
#

def get_global_id(ap_params, t):
    """
    For a given time t, this function return the projection of the t in
    the discretization scheme of the AP simulation according to:
            gid = int( (t-t_ini) / t_delta )

    If t is below t_ini, then gid = 0
    If t is above t_end, then gid = -1
    """

    if t < ap_params['data']['t_ini']:
        print(f"WARNING: given time {t} is below the minimum time of the simulation ({ap_params['data']['t_ini']})...")
        gid = 0
    elif t > ap_params['data']['t_end']:
        print(f"WARNING: given time {t} is above the maximum time of the simulation ({ap_params['data']['t_end']}) ....")
        gid = -1
    else:
        gid = int((t - ap_params['data']['t_ini']) / ap_params['data']['t_delta'])

    return gid
#

def build_electrodes(eg_params, mesh, torso=None, t_delta=1.0, normalize=False, sort=True):

    if not eg_params['data']['electrodes'] and torso is None:
        print("ERROR: either eg_params must contain a probes directory or a torso\
              or a torso model containing a 'electrodes' data array must be provided....")
        return None

    elecs = {}

    if torso is not None:
        ids = torso['electrodes'] != ''
        for name, loc in zip(torso['electrodes'][ids], torso.points[ids]):
            elecs[name] = Electrode(name=name, loc=loc)
            elecs[name].compute_distance_vectors(mesh=mesh, normalize=normalize)
            elecs[name].t_ini = eg_params['data']['t_ini']
            elecs[name].t_end = eg_params['data']['t_end']
            elecs[name].t_delta = t_delta
    else:
        for name, loc in eg_params['data']['electrodes'].items():
            elecs[name] = Electrode(name=name, loc=loc)
            elecs[name].compute_distance_vectors(mesh=mesh, normalize=normalize)
            elecs[name].t_ini = eg_params['data']['t_ini']
            elecs[name].t_end = eg_params['data']['t_end']
            elecs[name].t_delta = t_delta

    if sort:
        return { k:v for k, v in sorted(elecs.items(), key=lambda item: item[1].name)}

    return elecs
#

def remove_core_from_mesh(mesh):

    ids = mesh['Cell_type'] < 2
    mesh['g_id'] = np.arange(0,mesh.points.shape[0], step=1)
    mesh_filt = mesh.extract_points(ids, adjacent_cells=False)

    return mesh_filt, mesh['g_id'].copy()
#

def find_point_ids(arr1, arr2):
    """
    Use the KDTree module to find the ids of the elements of arr2 in arr1

    Arguments:
    ----------

        arr1 : np.ndarray, (MxN)

        arr2 : np.ndarray, (KxN)


    """

    kdt = KDTree(arr1)
    _, ids = kdt.query(arr2)

    return ids
#

def find_point_ids_sorted(arr1, arr2):
    """
    Find the sorted ids in arr1 of the elements present in both arrays.

    Arguments:
    ----------

        arr1 : np.ndarray, (MxN)

        arr2 : np.ndarray, (KxN)


    """

    if arr1.shape[1] != arr2.shape[1]:
        raise ValueError('arr1 and arr2 have different dimensionality.')

    N = arr1.shape[1]
    int1 = [np.intersect1d(arr1[:,i], arr2[:,i], assume_unique=True,
                return_indices=True)[1] for i in range(N)]

    ids = int1[0]
    for i in range(N-1):
        ids = np.intersect1d(ids, int[i+1], assume_unique=True)

    return ids
#

def save_EG(path, probes, w=False):

    fname = f"{path}/EG.npz"

    if os.path.exists(fname) and not w:
        print(f"WARNING: {fname} already exists, nothing will be saved....")
        return

    save_arrs = {name : probe.EG for name, probe in probes.items()}
    np.savez_compressed(fname, **save_arrs)
#

def set_up_EG_times(ap_params, eg_params):
    """
    Function to set up proper EG times, depending on the existing
    simulation. It is experessed in ms. The function follows the
    logic:

    If t_ini  is negative, it is set as last simulation time minus t_ini
    If t_end is None it is set as the last simulation time.
    If t_ini > t_end prompts an error and returns none.

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


    if eg_params['data']['t_ini'] < 0:
        eg_params['data']['t_ini'] = ap_params['data']['t_end'] + eg_params['data']['t_ini']

    if eg_params['data']['t_end'] is None:
        eg_params['data']['t_end'] = ap_params['data']['t_end']

    if eg_params['data']['t_ini'] > eg_params['data']['t_end']:
        print("ERROR: As set, t_ini is bigger than t_end.....")
        return None

    return eg_params
#



def AP_to_EG(case_path, sim_path, eg_params, mesh=None, torso=None, debug=False, w=False):
    """
    The main function to compute the electrogram from a simulation a ventricle model, and a
    set of electrodes.

    """

    if eg_params is None:
        eg_params = read_EG_json(path=case_path, abs_path=False)

    if mesh is None and os.path.exists(f'{case_path}/ventricle_Tagged.vtk'):
        mesh = pv.read(f'{case_path}/ventricle_Tagged.vtk')
    else:
        print(f"ERROR: mesh not passed nor available at {case_path}/ventricle_Tagged.vtk")
        return
    mesh_filt, _ = remove_core_from_mesh(mesh=mesh)

    if torso is None and os.path.exists(f'{case_path}/torso/torso.vtk'):
        print(f"Loading detected torso at: {case_path}/torso/torso.vtk")
        torso = pv.read(f'{case_path}/torso/torso.vtk')
    elif torso is None:
        print(f"No torso model has been passed, nor could be found at {case_path}. Electrodes will be built from EG_params. ")

    AP, ap_params = load_sim_path(sim_path=sim_path)

    electrodes = build_electrodes(torso=torso, eg_params=eg_params, mesh=mesh_filt, t_delta=ap_params['data']['t_delta'], normalize=False)

    now = dt.datetime.now()
    eg_params['metadata']['date'] = now.strftime('%d-%m-%Y')
    eg_params['metadata']['time'] = now.strftime('%H:%M')

    eg_params = set_up_EG_times(ap_params, eg_params)
    write_EG_json(sim_path, data=eg_params)

    t_ini_ig = get_global_id(ap_params, eg_params['data']['t_ini'])
    t_end_ig = get_global_id(ap_params, eg_params['data']['t_end'])

    for i in trange(t_ini_ig, t_end_ig, desc=f"Computing Electrogram from {eg_params['data']['t_ini']:.2f} to {eg_params['data']['t_end']:.2f}",):
        mesh_filt['AP'] = AP[mesh_filt['g_id'], i]
        mesh_filt = mesh_filt.compute_derivative(scalars='AP', gradient='AP_grad')
        print(f"------------")
        for ename, electrode in electrodes.items():
            electrode.compute_EG(mesh_filt)
            print(f"{ename} : {electrode.EG[-1]}")


        if debug:
            p = pv.Plotter()
            glyph = mesh_filt.glyph(orient='AP_grad', scale=False, geom=pv.Arrow())
            p.add_mesh(glyph, lighting=False)
            #p.add_mesh(mesh, color="grey", ambient=0.6, opacity=0.5, show_edges=False)
            if torso is not None:
                p.add_mesh(torso, color='white', opacity=0.4, show_edges=True)
            p.add_point_labels(np.array([e.loc for _, e in electrodes.items()]), [e.name for _, e in electrodes.items()])
            p.show()

    save_EG(sim_path, electrodes, w=w)

    if debug:
        nr, l = 2, len(electrodes)
        nc = l//nr + l%nr
        el = list(electrodes.keys())
        f, axg = plt.subplots(nrows=nr, ncols=nc)
        for i in range(nr):
            for j in range(nc):
                electrodes[el[i*nc + j*nr]].plot(ax=axg[i][j], show=False)
        plt.show()


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="""Script to compute the EG of a simulation.""",
                                    usage = """ """)

    parser.add_argument('-e',
                        '--eg-params',
                        dest='eg_params',
                        action='store',
                        default=None,
                        type=str,
                        nargs='?',
                        help="""Path to an EG_data.json.""")

    parser.add_argument('-m',
                        '--mesh',
                        dest='mesh_fname',
                        action='store',
                        default=None,
                        type=str,
                        nargs='?',
                        help="""Path to the mesh domain used for the simulation.""")

    parser.add_argument('-d',
                        '--debug',
                        dest='debug',
                        action='store_true',
                        help="""Run in debug mode. Which essentialy is showing
                        some plots...""")

    parser.add_argument('-w',
                        '--overwrite',
                        dest='w',
                        action='store_true',
                        help=""" Overwrite existing files.""")

    parser.add_argument('-t',
                        '--torso-mesh',
                        dest='torso',
                        action='store',
                        default=None,
                        type=str,
                        nargs='?',
                        help="""Path to an already aligned torso model that
                        contains a data array called electrodes.""")




    parser.add_argument('case',
                        action='store',
                        type=str,
                        nargs='?',
                        help="""Path to a case directory. It must contain the
                        ventricle_Tagged.vtk file and additionally the torso
                        directory if not passed using -t argument.""")


    parser.add_argument('sim_path',
                        action='store',
                        default=None,
                        type=str,
                        nargs='?',
                        help="""Path to the directory containing the AP.npy,
                        and an ap_params.json.""")

    args = parser.parse_args()

    if args.case is None:
        print("ERROR: Wrong case path given ....")
        sys.exit()

    if args.sim_path is None:
        print("ERROR: Wrong simulation path given ....")
        sys.exit()

    mesh=None
    if args.mesh_fname is not None:
        mesh = pv.read(args.mesh_fname)

    torso=None
    if args.torso is not None:
        torso = pv.read(args.torso)

    if args.eg_params is not None:
        eg_params = read_EG_json(path=args.eg_params, abs_path=True)
    else:
        eg_params = None

    AP_to_EG(case_path = args.case,
             sim_path  = args.sim_path,
             eg_params = eg_params,
             mesh      = mesh,
             torso     = torso,
             debug     = args.debug,
             w         = args.w)
