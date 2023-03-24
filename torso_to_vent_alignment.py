#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

import pyvista as pv

from param_utils import (read_registrate_json, write_registrate_json,
                         read_landmarks_json, write_landmarks_json)

def compose_mat_trans(t, s, r):
    """
    Compose a transformation matrix from a translation, scale and rotation components.

    Arguments:
    --------

        t : np.ndarray (3,)
            Translation vector.

        s : np.ndarray (3,)
            Scale vector with scale in each coordinate (sx, sy, sz).

        r : np.ndarray (3, 3)
            Rotation matrix.



    Returns:
    --------

        m : np.ndarray (4,4)
            Transformation matrix

    """

    m = np.eye(4)

    m[:-1, -1] = t
    m[:3, :3]  = r
    m[0, :]   *= s[0]
    m[1, :]   *= s[1]
    m[2, :]   *= s[2]

    return m
#

def rigid_transform_3D(target, source):
    """
    Implementation of:
        "Least-Squares Fitting of Two 3-D Point Sets", Arun, K. S. and Huang, T. S. and Blostein, S. D,
        IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume 9 Issue 5, May 1987

    Arguments:
    -----------

        target : np.ndarray (3,N)
            The matrix that will be transformed.
        source : np.ndarray (3,N)
            The matrix that serves as reference.

    Returns:
    ----------
        R : np.ndarray(3,3)
            The rotation matrix
        t : np.ndarray(3,)
            The translation vector


    """

    assert target.shape == source.shape

    num_rows, num_cols = target.shape
    if num_rows != 3:
        raise Exception(f"matrix target is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = source.shape
    if num_rows != 3:
        raise Exception(f"matrix source is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_t = np.mean(target, axis=1)
    centroid_s = np.mean(source, axis=1)

    # ensure centroids are 3x1
    centroid_t = centroid_t.reshape(-1, 1)
    centroid_s = centroid_s.reshape(-1, 1)

    # subtract mean
    tm = target - centroid_t
    sm = source - centroid_s

    H = tm @ np.transpose(sm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        #print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_t + centroid_s

    return R, t.ravel()
#

def transform(pts, t=None, s=None, r=None):
    """
    Apply an affine transformation to a point cloud expressed as (N,3) numpy array.

    Arguments:
    -------------

        pts : np.ndarray (N,3)
            the array of points

        t : np.ndarray (3,)
            Translation vector.

        s : np.ndarray (3,)
            Scale vector with scale in each coordinate (sx, sy, sz).

        r : np.ndarray (3, 3)
            Rotation matrix.

    Returns:
    ---------

        pts_tr : np.ndarray (N,3)
            The transformed points.

    """

    if t is None:
        t=np.zeros((3,))
    if s is None:
        s=np.ones((3,))
    if r is None:
        r=np.eye(3)

    A = compose_mat_trans(t, s, r)
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    pts_tr = unpad(np.dot(A, pad(pts).T).T)

    return pts_tr



class torso_vent_alignment:

    def __init__(self):

        self.target_vent : pv.PolyData = None
        self.source_vent : pv.PolyData = None

        self.target_landmarks : np.ndarray = None
        self.source_landmarks : np.ndarray = None

        self.target_torso : pv.PolyData = None
        self.method   : str = 'rigid_align'

        self.translation_vector : np.ndarray = None
        self.rotation           : Rotation = None
        #self.scale              : np.ndarray = None

        self.vent_path : str = None

        self.params: dict = read_registrate_json()['data']
    #

    def __load_landmarks(self, path):
        """
        Function to load landmarks from json file and parse it into a np.array
        """

        lndmrks = read_landmarks_json(path=path)['data']

        if lndmrks['base'] is None or lndmrks['sept'] is None or lndmrks['apex'] is None:
            print(f"ERROR: Unable to find a propper landmarks.json at {path}....")
            print("""The landmarks.json expected should contain:\n\t"data" : {\n\t\t"base" : [x,y,z],\n\t\t"apex" : [x,y,z],\n\t\t"sept" : [x,y,z]""")
            return None

        return np.array([lndmrks['base'],
                        lndmrks['apex'],
                        lndmrks['sept']])
    #

    def __save_landmarks(self, path, lndmrks_arr):

        lndmrks = read_landmarks_json()

        lndmrks['base'] = lndmrks_arr[0]
        lndmrks['apex'] = lndmrks_arr[1]
        lndmrks['sept'] = lndmrks_arr[2]

        write_landmarks_json(path=path, data=lndmrks)
    #

    def load_torso(self, torso_dir=None):

        if torso_dir is None:
            torso_dir = self.params['torso_dir']

        self.torso = pv.read(f"{torso_dir}/torso.vtk")
        try:
            self.target_vent = pv.read(f"{torso_dir}/ventricle.vtk")
        except FileNotFoundError:
            print("Warning: No target ventricular mesh found.")

        self.target_landmarks = self.__load_landmarks(torso_dir)
    #

    def load_source_data(self, path=None):

        if not path is None:
            self.vent_path = path

        self.source_vent = pv.read(os.path.join(self.vent_path,"ventricle_Tagged.vtk"))
        self.source_landmarks = self.__load_landmarks(self.vent_path)
    #

    def landmark_procrustes(self, update=True):
        """
        This method perform a rigid registration using the centerline in
        the feature vector provided (source_fv), and the target aorta feature vector.

        """

        r, t = rigid_transform_3D(target=self.target_landmarks.T,
                                  source=self.source_landmarks.T)

        if update:
            self.translation_vector = t.flatten
            self.rotation = r

        return r, t
    #

    def compute_registration(self, update=True, apply=True):

        r, t = self.landmark_procrustes(update=update)

        if apply:
            self.apply_transformation(t=t, r=r)
    #

    def apply_transformation(self, t=None, r=None, s=None):

        if t is not None:
            self.translation_vector = t
        if r is not None:
            self.rotation = r


        if self.torso is not None:
            self.torso.points = transform(self.torso.points, t=self.translation_vector, r=self.rotation)
        if self.target_landmarks is not None:
            self.target_landmarks = transform(self.target_landmarks, t=self.translation_vector, r=self.rotation)
        if self.target_vent is not None:
            self.target_vent.points = transform(self.target_vent.points, t=self.translation_vector, r=self.rotation)
    #

    def save_torso(self, path=None, f=False):

        if not path is None:
            self.vent_path = path

        if self.vent_path is None:
            print("ERROR: No path has been provided.....")
            return

        torso_dir = self.vent_path+'/torso_dir'
        if not os.path.exists(torso_dir) or not os.path.isdir(torso_dir):
            os.makedirs(torso_dir)

        torso_fname = f"{torso_dir}/torso.vtk"
        if os.path.exists(torso_fname) and not f:
            print(f"ERROR: The file {torso_fname} already exists and force (f) argument is {f}...")
            print(f"\t\t...nothing will be saved")
        else:
            self.torso.save(f"{torso_dir}/torso.vtk")
            self.__save_landmarks(path=torso_dir, lndmrks_arr=self.target_landmarks)
    #

    def plot(self, show_target_vent=False):

        p = pv.Plotter()

        if self.source_vent is not None:
            p.add_mesh(self.source_vent, opacity=0.8, color='r')
        if self.source_landmarks is not None:
            p.add_mesh(self.source_landmarks, render_points_as_spheres=True, color='y')


        if self.torso is not None:
            p.add_mesh(self.torso, opacity=0.4, color='w', show_edges=True)
        if self.target_vent is not None and show_target_vent:
            p.add_mesh(self.target_vent, opacity=0.8, color='g')
        if self.target_landmarks is not None and show_target_vent:
            p.add_mesh(self.target_landmarks, render_points_as_spheres=True, color='y')

        p.add_axes()
        p.show()
    #

def align_torso_to_vent(vent_dir, torso_dir, debug=False, f=False):

    """
    Function to align a torso model to a patient ventricle.
    To properly work there should exist a directory (torso_dir)
    where a torso.vtk and a landmarks.json can be found. The landmarks.json
    must contain the entries "base", "apex","sept" with the coordinates of
    the landmarks, in the same reference system as the torso, that will be used
    to registrate against the same landmarks found in landmarks.json that must be
    in the same directory where the ventricle model is found.

    Additionally a ventricle.vtk file may be found at torso_dir. There a ventricle model
    in the torso coordinate system can be read for debuging or plotting purposes.

    Arguments:
    ------------

        vent_dir : str
            The path to the ventricle directory. It is assumed to contain the
            ventricle_Tagged.vtk and the landmark.json.

        torso_dir : str
            The path to the torso directory containing the torso.vtk and the
            landmark.json files.

        f : bool
            To force overwritting if a torso_dir already exists at vent_dir

    """

    if not os.path.exists(torso_dir) or not os.path.isdir(torso_dir):
        print(f"ERROR: Wrong torso directory given.. Could not find {torso_dir}")

    if not os.path.exists(torso_dir) or not os.path.isdir(torso_dir):
        print(f"ERROR: Wrong torso directory given.. Could not find {torso_dir}")


    tva = torso_vent_alignment()
    tva.vent_path = vent_dir
    tva.load_source_data()
    tva.load_torso(torso_dir=torso_dir)

    if debug:
        tva.plot(show_target_vent=True)

    tva.compute_registration(apply=True)

    if debug:
        print("-----------------------")
        print("Rotation Matrix:")
        print(tva.rotation)
        print("-----------------------")
        print("Translation Vector:")
        print(tva.rotation)
        print("-----------------------")
        tva.plot(show_target_vent=True)


    tva.save_torso(f=f)

    return tva.torso



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="""Script to perform
                                    an alignment between torso and a ventricle.""",
                                    usage = """ """)


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

    parser.add_argument('vent_dir',
                        action='store',
                        type=str,
                        nargs='?',
                        help="""Path to a directory containing the ventricle_Tagged.vtk
                        and landmark.json files.""")

    parser.add_argument('torso_dir',
                        action='store',
                        type=str,
                        nargs='?',
                        help="""Path to a directory containing the torso.vtk and landmark.json
                        files optionally can contain ventricle.vtk.""")


    args = parser.parse_args()

    if args.vent_dir is not None:
        if not os.path.exists(args.vent_dir) or not os.path.isdir(args.vent_dir):
            print("ERROR: Given vent_dir does not exist or is not a valid directory")
            sys.exit()

    if args.vent_dir is not None:
        if not os.path.exists(args.torso_dir) or not os.path.isdir(args.torso_dir):
            print("ERROR: Given torso_dir does not exist or is not a valid directory")
            sys.exit()

    align_torso_to_vent(args.vent_dir, args.torso_dir, debug=args.debug, f=args.f)