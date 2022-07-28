#/usr/bin/env python3

import os, sys

import argparse

import numpy as np

import pyvista as pv

import matplotlib.pyplot as plt

from tqdm import trange
from scipy.spatial import KDTree

from param_utils import read_AP_json, read_EG_json, write_EG_json


class Probe:

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

        return F"""Probe: {self.name}:
                        props:
                            location: {self.loc}
                            Time instants computed: {nt}

                        Time parameters:
                            t_ini = {self.t_ini}
                            t_end = {self.t_end}
                            t_delta = {self.t_delta}
                """

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
                defined at each point, called 'grad_AP' containing the gradient
                of the Action Potential.

        \Phi_e(r_e, t) = - 1/(4\pi\sigma_e) ∫\sigma(r) nablaV_m(r_jT) \gamma_d/r_d^3 dr

        P. Podziemski, P. Kuklik, A. van Hunnik, S. Zeemering, B. Maesen and U. Schotten,
        "Far-field effect in unipolar electrograms revisited: High-density mapping of atrial fibrillation in humans,"
        2015 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC),
        2015, pp. 5680-5683, doi: 10.1109/EMBC.2015.7319681.
        """

        if self.EG is None:
            self.EG = []

        self.EG.append((self.Mr * mesh['grad_AP']).sum())

    def plot(self, ax=None, show=False):

        if ax is None:
            f, ax = plt.subplots()
        if self.EG is not None:
            t = self.t_ini + np.arange(len(self.EG)) * self.t_delta
            ax.plot(t, self.EG)

        if show:
            plt.show()



def load_data_from_path(path, ap_params=None):

    if os.path.exists(f'{path}/ventricle_Tagged.vtk'):
        mesh = pv.read(f'{path}/ventricle_Tagged.vtk')
    else:
        mesh = None


    if os.path.exists(f'{path}/EG_data.json'):
        eg_params = read_EG_json(path=path)
    else:
        eg_params = None

    if os.path.exists(f'{path}/AP_data.json') and ap_params is None:
        ap_params = read_AP_json(path=path)

    if os.path.exists(f'{path}/AP.npy'):
        AP = np.memmap(f'{path}/AP.npy', dtype=float, mode='r', shape=tuple(ap_params['data']['shape']))
    else:
        AP = None

    return mesh, AP, ap_params, eg_params
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

def build_probes(eg_params, mesh, normalize=False):

    probes = {}

    for name, loc in eg_params['data']['probes'].items():
        p = Probe(name=name, loc=loc)
        p.compute_distance_vectors(mesh=mesh, normalize=normalize)

        probes[name] = p
        p.t_ini = eg_params['data']['t_ini']
        p.t_end = eg_params['data']['t_end']
        p.t_delta = eg_params['data']['t_delta']

    return probes
#

def remove_core_from_mesh(mesh):

    ids = mesh['Cell_type'] < 2
    mesh_filt = mesh.extract_points(ids, adjacent_cells=False)

    ids = find_point_ids(arr1=mesh.points, arr2=mesh_filt.points)

    return mesh_filt, ids
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


def AP_to_EG(path, mesh, AP, ap_params, eg_params, debug=False, w=False):

    t_ini_ig = get_global_id(ap_params, eg_params['data']['t_ini'])
    t_end_ig = get_global_id(ap_params, eg_params['data']['t_end'])

    mesh_filt, ids = remove_core_from_mesh(mesh=mesh)

    probes = build_probes(eg_params=eg_params, mesh=mesh_filt, normalize=False)

    for i in trange(t_ini_ig, t_end_ig):
        mesh_filt['AP'] = AP[ids,i]
        mesh_filt = mesh_filt.compute_derivative(scalars='AP', gradient='AP_grad')
        for probe in probes:
            probe.compute_EG(mesh_filt)

        if debug:
            p = pv.Plotter()
            glyph = mesh_filt.glyph(orient='AP_grad', scale=np.ones(mesh_filt.points.shape), geom=pv.Arrow())
            p.add_mesh(glyph, lighting=False)
            #p.add_mesh(mesh, color="grey", ambient=0.6, opacity=0.5, show_edges=False)
            p.show()
            input()

    save_EG(path, probes, w=w)



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

    #parser.add_argument('-a',
    #                    '--ap-params',
    #                    dest='ap_params',
    #                    action='store',
    #                    default=None,
    #                    type=str,
    #                    nargs='?',
    #                    help="""Path to an AP_data.json.""")

    parser.add_argument('-s',
                        '--sim-ap',
                        dest='AP',
                        action='store',
                        default=None,
                        type=str,
                        nargs='?',
                        help="""Path to an AP.npy simulation.""")

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
                        help="""Run in debug mode. Which essentialy is showing some plots...""")

    parser.add_argument('-w',
                        '--overwrite',
                        dest='w',
                        action='store_true',
                        help=""" Overwrite existing files.""")


    parser.add_argument('path',
                        action='store',
                        type=str,
                        nargs='?',
                        help="""Path to a directory where required files may be located. Some files can be missed if they have been passed with an optional arg.""")

    args = parser.parse_args()

    if args.path is None:
        print("ERROR: Wrong path given ....")
        sys.exit()

    mesh, AP, ap_params, eg_params = load_data_from_path(args.path)

    if args.mesh_fname is not None:
        mesh = pv.read(args.mesh_fname)


    if args.eg_params is not None:
        eg_params = read_EG_json(path=args.case_params, abs_path=True)



    for (n, x) in zip(['AP', 'mesh', 'ap_params', 'eg_params'], [AP, mesh, ap_params, eg_params]):
         if x is None:
            print(f"ERROR: {n} argument is None.....")
            sys.exit()

    AP_to_EG(args.path, mesh, AP, ap_params=ap_params, eg_params=eg_params, debug=args.debug, w=args.w)
