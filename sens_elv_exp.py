#! /usr/bin/env python3

import os

import json

import argparse

import numpy as np

import pandas as pd

from peaks import peakdetect

import matplotlib.pyplot as plt

import pyvista as pv

from utils import check_attrs, check_s2_params


def compute_max_der_and_perc_repolarization(t, ap, perc=0.9, full_output=False, show=False):
    """Compute the times for the points
            1)With maximum derivative before the peak,
            2)Fulfilling a given percentage of repolarization
    """

    max_i = np.argmax(ap)

    #Get max derivative of AP in depolarization fase
    d0 = t[1]-t[0]
    ap_der = np.gradient(ap[:max_i+5], d0)
    max_der = np.argmax(ap_der)
    t_max_der = t[max_der]

    #Then we compute when repolarization reaches x% from max to min
    ap_aux = ap - ap[0]
    ap_aux = np.abs(ap_aux - ap_aux.max() * (1-perc))
    apdq_i =  max_i + np.argmin(ap_aux[max_i:])
    apdq_t =  t[apdq_i]

    if show:
        plt.plot(t, ap, 'k')
        plt.plot([t_max_der, apdq_t], ap[[max_der, apdq_i]], 'r')
        plt.axvspan(t_max_der, apdq_t, facecolor='r', alpha=0.3)
        plt.show()

    if full_output:
        return [t_max_der, ap[max_der]], [apdq_t, ap[apdq_i]]

    return t_max_der, apdq_t
#


def filter_peaks_by_stimuli_pairs_and_value(minpeaks, maxpeaks, mp_a=-np.inf, mp_b=np.inf, Mp_a=-np.inf, Mp_b=np.inf):
    """

    Arguments:
    -----------
        minpeaks : arraylike (npeaks, 2)
            The minpeaks of the signal, understood as the minimum AP before the simulus arrive. The expected shape is a list/array where the elements are [t, AP].

        maxpeaks : arraylike (npeaks, 2)
            The maxpeaks of the signal, understood as the spike generated by the simulus. The expected shape is a list/array where the elements are [t, AP].

        mp_a : float
            Lower extreme for the acceptance interval of the minpeaks

        mp_b : float
            Upper extreme for the acceptance interval of the minpeaks

        Mp_a : float
            Lower extreme for the acceptance interval of the maxpeaks

        Mp_b : float
            Upper extreme for the acceptance interval of the maxpeaks

    Return:
    ---------
        maxpeaks : np.array
            The filtered array of maxpeaks with the shape of the input list

        minpeaks : np.array
            The filtered array of minpeaks with the shape of the input list

    """

    if not isinstance(minpeaks, np.ndarray):
        minpeaks = np.array(minpeaks)
    if not isinstance(maxpeaks, np.ndarray):
        maxpeaks = np.array(maxpeaks)

    #Keep only minpeaks if their AP belong to the interval [mp_a, mp_b]
    minpeaks = minpeaks[ (mp_a <= minpeaks[:,1]) & (minpeaks[:,1] <= mp_b) ]
    mp_remove=[]
    Mpeaks = []

    for i, mp in enumerate(minpeaks[:-1]):
        #Get the id of the temporarily closest maxpeak to mp
        Mpid = np.argmin(np.abs(maxpeaks[:,0] - mp[0]))
        #If its time is lower to mp's, the peak of interest is the following one.
        if maxpeaks[Mpid, 0] < mp[0]:
            Mpid +=1
        if Mp_a <= maxpeaks[Mpid][1] and maxpeaks[Mpid][1] <= Mp_b:
            Mpeaks.append(maxpeaks[Mpid])
        else:
            mp_remove.append(i)

    if mp_remove:
        minpeaks = np.delete(minpeaks, mp_remove, axis=0)

    maxpeaks = np.array(Mpeaks)

    return  maxpeaks, minpeaks
#



class NodeOutput:

    def __init__(self, i=None):

        self.__id : str = None
        if i is not None:
            self.id = i
        self.__AP : np.ndarray = None
        self.__time : np.ndarray = None
        self.__peaks : dict = None

        #S1S2 segmentation
        self.S1S2 : np.ndarray = None
        self.S1S2_gids : np.ndarray = None
        self.S2s : np.ndarray = None

        # Threshold for the peak values
        self.max_minpeak_value = -65
        self.min_maxpeak_value = 0

        #Spatial Coordinates
        self.__loc : np.ndarray = None

    #id
    @property
    def id(self):
        return self.__id
    @id.setter
    def id(self, i):
        self.__id = i
    @id.deleter
    def id(self):
        del self.__id
    #

    #Action Potential
    @property
    def AP(self):
        return self.__AP
    @AP.setter
    def AP(self, ap):
        self.__AP = ap
    @AP.deleter
    def AP(self):
        del self.__AP
    #

    #Time
    @property
    def time(self):
        return self.__time
    @time.setter
    def time(self, t):
        self.__time = t
    @time.deleter
    def time(self):
        del self.__time
    #

    #Peaks time
    @property
    def peaks(self):
        return self.__peaks
    @peaks.setter
    def peaks(self, p):
        self.__peaks = p
    @peaks.deleter
    def peaks(self):
        del self.__peaks
    #

    #Spatial location in R^n
    @property
    def loc(self):
        return self.__loc
    @loc.setter
    def loc(self, l):
        self.__loc = l
    @loc.deleter
    def loc(self):
        del self.__loc
    #

    def detect_peaks(self):

        for attr in ['time', 'AP']:
            if not hasattr(self, attr):
                print(f"Cant load data, {attr} has not been set")
                return

        maxtab, mintab = peakdetect(y_axis=self.AP, x_axis=self.time, lookahead=10, delta=20)


        mintab = [(self.time[0], self.AP[0])] + mintab + [(self.time[-1], self.AP[-1])]
        maxtab, mintab = filter_peaks_by_stimuli_pairs_and_value(mintab, maxtab, mp_b=self.max_minpeak_value, Mp_a=self.min_maxpeak_value)

        self.peaks = {}
        self.peaks['max'] = np.array(maxtab).T
        self.peaks['min'] = np.array(mintab).T
    #

    def plot(self, imin=0, imax=-1, ax=None, show=False):

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.time[imin:imax], self.AP[imin:imax], 'k', label='Action Potential')
        ax.scatter(self.peaks['max'][0,imin:imax], self.peaks['max'][1,imin:imax], c='r', marker='*', label='Max Peaks')
        ax.scatter(self.peaks['min'][0,imin:imax], self.peaks['min'][1,imin:imax], c='b', marker='*', label='Min Peaks')
        ax.set_title(self.id)
        if show:
            plt.show()
    #
#

class SensElvExp:

    def __init__(self):

        self.__path : str = None
        self.__output_path : str = None

        self.__cell_type : str = None

        self.__nodes : dict = None

        self.__debug : bool = False

        self.s1        : int = None
        self.s2_ini    : int = None
        self.s2_end    : int = 250
        self.s2_step   : int = 20
        self.tr_offset : int = 0
        self.s1_per_s2 : int = 9

        self.CV_DI : pd.DataFrame = pd.DataFrame(columns=['S1', 'S2', 'DI_or', 'DI_dest', 'APD_or', 'APD_dest', 'CV', 'cell_type', 'node_or', 'node_dest'])
        self.APD_DI : pd.DataFrame = pd.DataFrame(columns=['S1', 'S2', 'APD', 'DI', 'APD+1', 'cell_type', 'node_id'])
        self.stimuli_segmented : pd.DataFrame = pd.DataFrame(columns=['S1', 'S2', 'DI', 'AP+1', 't_act', 'AP', 'cell_type', 'node_id', "APD", "DI","APD+1"])
    #

    #Experiment's path to directory
    @property
    def path(self):
        return self.__path
    @path.setter
    def path(self, p):
        self.__path = p
    @path.deleter
    def path(self):
        del self.__path
    #

    #Debug Mode
    @property
    def debug(self):
        return self.__debug
    @debug.setter
    def debug(self, d):
        self.__debug = d
    @debug.deleter
    def debug(self):
        del self.__debug
    #

    #Path to ELVIRA simulation output dir.
    @property
    def output_path(self):
        return self.__output_path
    @output_path.setter
    def output_path(self, p):
        self.__output_path = p
    @output_path.deleter
    def output_path(self):
        del self.__output_path
    #

    #Cell type used for the experiment, should be in {'ttepi', 'ttmid' ,'ttendo', 'ttepibz', 'ttmidbz' ,'ttendobz'}.
    @property
    def cell_type(self):
        return self.__cell_type
    @cell_type.setter
    def cell_type(self, ct):
        self.__cell_type = ct
    @cell_type.deleter
    def cell_type(self):
        del self.__cell_type
    #

    #Nodes to evaluate
    @property
    def nodes(self):
        return self.__nodes
    @nodes.setter
    def nodes(self, nd):
        self.__nodes = nd
    @nodes.deleter
    def nodes(self):
        del self.__nodes
    #

    def load_node_info(self):

        if not check_attrs(self, ['path', 'output_path'], 'Cant load data'):
            return

        node_files = [nf for nf in os.listdir(self.output_path) if nf.startswith('node_') and nf.endswith('.npy')]
        read = np.load

        if node_files == []:
            node_files = [nf for nf in os.listdir(self.output_path) if nf.startswith('prc') and nf.endswith('.var')]
            read = np.loadtxt

        data_path = self.path + '/data'
        node_loc = np.loadtxt(data_path+'/nodes.dat', skiprows=2)

        self.domain = node_loc[:, 2:]

        self.nodes = {}
        for nf in node_files:
            data = read( self.output_path+'/'+nf )
            i = nf[ nf.rfind('_',)+1 :-4].lstrip('0')
            if not i in self.nodes.keys():
                self.nodes[i] = NodeOutput(i)
                self.nodes[i].time = data[:, 0]
                self.nodes[i].AP   = data[:, 1]
                self.nodes[i].detect_peaks()
                self.nodes[i].loc = node_loc[int(i)-1, 2:]

        if self.debug:
            p = pv.Plotter()
            p.add_mesh(self.domain, color='w')
            sel_points = np.array([n.loc for _, n in self.nodes.items()])
            p.add_mesh(sel_points, scalars=np.array([n for n in self.nodes.keys()]), render_points_as_spheres=True, point_size=10)
            p.camera_position = 'xy'
            p.camera.azimuth = -180
            p.add_axes()
            p.show()
    #

    def save_nodes_npy(self, rm_var=False):

        if not check_attrs(self, ['path', 'output_path', 'nodes'], 'Cant save data'):
            return

        for name, node in self.nodes.items():

            arr = np.zeros((node.time.shape[0], 2))
            arr[:, 0] = node.time
            arr[:, 1] = node.AP
            fname = f'{self.output_path}/node_{name}.npy'
            np.save(fname, arr)
        if rm_var:
            self.remove_var_files()
    #

    def remove_var_files(self):

        var_files = [nf for nf in os.listdir(self.output_path) if nf.startswith('prc') and nf.endswith('.var')]
        for vf in var_files:
            i = vf[ vf.rfind('_',)+1 :-4].lstrip('0')
            if os.path.exists(f'node_{i}.npy'):
                os.remove(f'{self.output_path}/{vf}')
            else:
                print(f"WARNING: {vf} won't be deleted as its npy version was not found...")
    #

    def extract_S1S2(self, node_id=None, extract_ids=True):

        """

        TODO: Documentation

        """

        if node_id is None:
            for nid in self.nodes:
                self.extract_S1S2(node_id=nid, extract_ids=extract_ids)
            return

        node = self.nodes[node_id]

        S1S2 = []
        S2s = []

        s2_step, cond = check_s2_params(s2_ini=self.s2_ini, s2_end=self.s2_end, s2_step=self.s2_step)

        s2 = self.s2_ini
        t0 = node.peaks['min'][0, np.abs(node.peaks['min'][0] - 0).argmin()]
        t_ini = t0 + (self.s1_per_s2-1) * self.s1
        t_end = t_ini + s2 + min(self.tr_offset * 0.75, 1000)

        end=False
        while cond(s2) and not end:

            #Get the min peak of the S1 before the S2
            mp_S1_i = np.abs(node.peaks['min'][0] - t_ini).argmin()

            if mp_S1_i >= node.peaks['min'][0].shape[0]-2:
                end=True
            else:
                #t_ini is an underestimation of the activation, thus the peak should be after t_ini.
                if node.peaks['min'][0, mp_S1_i] < t_ini:
                    mp_S1_i +=1

                #We store the delay between the estimated activation and the peak
                d = np.abs(node.peaks['min'][0, mp_S1_i] - t_ini)

                mp_S1  = node.peaks['min'][:, mp_S1_i]

                #We check the following min peak as it should be the S2.
                if np.abs(node.peaks['min'][0, mp_S1_i] - node.peaks['min'][0, mp_S1_i+1]) < s2 * 1.2:
                    mp_S2_i = mp_S1_i+1
                    mp_S2  = node.peaks['min'][:, mp_S2_i]

                    #For the ending point we take the minimum between the following min peak's time and 1000 ms after the S2 min peak.
                    arr_aux = np.array([node.peaks['min'][0, mp_S2_i]+1000, node.peaks['min'][0, mp_S2_i+1]])
                    iaux = arr_aux.argmin()
                    if iaux == 0:
                        jaux = np.abs(node.time-arr_aux[iaux]).argmin()
                        END = np.array([node.time[jaux], node.AP[jaux]])
                    else:
                        END = node.peaks['min'][:, mp_S2_i+1]

                    S1S2.append([mp_S1, mp_S2, END])
                    S2s.append(s2)

                    if self.debug:
                        ids = (node.time > t0) & (node.time < min(t_end, END[0]))
                        plt.plot(node.time[ids] - t0, node.AP[ids])
                        plt.axvspan(t0 - t0, mp_S1[0] - t0, facecolor='b', alpha=0.3, label="Stimuli train")
                        plt.axvspan(mp_S1[0]- t0, mp_S2[0] - t0, facecolor='y', alpha=0.3, label="S1")
                        plt.axvspan(mp_S2[0]- t0, END[0]   - t0, facecolor='r', alpha=0.3, label="S2")
                        plt.title(f"Node: {node_id} {self.cell_type} S1 {self.s1}, S2 {s2}")
                        plt.show()
                        input('Do you want to keep going on?(y/Ctrl+c)')
                else:
                    print(f"WARNING :: Node {node_id} cell type {self.cell_type}, s1 {self.s1}, s2 {s2}:\n",
                        f"The S1 happens at {node.peaks['min'][0, mp_S1_i]}, the following minimum peak occurs at {node.peaks['min'][0, mp_S1_i+1]}. \n"
                        f"It exceeds the limit (S2*1.2={s2*1.2:.2f}) that would be at {node.peaks['min'][0, mp_S1_i+1]+s2*1.2:.2f}")

                s2 += s2_step
                t0 += self.s1_per_s2 * self.s1 + s2 + self.tr_offset + d
                t_ini = t0 + (self.s1_per_s2-1) * self.s1
                t_end = t_ini + s2 + min(self.tr_offset * 0.75, 1000)

        node.S1S2 = np.array(S1S2)
        node.S2s = np.array(S2s)

        if extract_ids:
            self.extract_S1S2_global_ids(node_id=node_id)
   #

    def extract_S1S2_global_ids(self, node_id=None):

        if not check_attrs(self, ['nodes'], "Can't extract global ids."):
            return

        if node_id is None:
            for nid in self.nodes:
               self.extract_S1S2_global_ids(node_id=nid)
            return

        node = self.nodes[node_id]

        if not check_attrs(node, ['S1S2'], f"Node {node_id} dont have S1S2 info..."):
            return

        glob_ids = [[np.abs(node.time - p[0]).argmin() for p in s1s2] for s1s2 in node.S1S2]

        node.S1S2_gids = glob_ids
    #

    def compute_APD_DI(self, node_id=None, save_csv=False, w=False):

        if not check_attrs(self, ['nodes', 'cell_type'], "Can't compute APD and DI"):
            return

        if node_id is None:
            for nid in self.nodes.keys():
                self.compute_APD_DI(node_id=nid, save_csv=False)
            if save_csv:
                self.save_APD_DI(w=w)
            return

        node = self.nodes[node_id]

        if not check_attrs(node, ['S2s', 'S1S2_gids'], f"ERROR: Can't compute APD and DI at node {node_id}"):
            return

        data = []
        for s2, [i1,i2,ie] in zip(node.S2s, node.S1S2_gids):

            S1_md_t, S1_apd90_t = compute_max_der_and_perc_repolarization(node.time[i1:i2], node.AP[i1:i2], perc=0.9, show=self.debug)
            S2_md_t, S2_apd90_t = compute_max_der_and_perc_repolarization(node.time[i2:ie], node.AP[i2:ie], perc=0.9, show=self.debug)

            data.append({'S1'        : self.s1,
                         'S2'        : s2,
                         'APD'       : S1_apd90_t - S1_md_t,
                         'DI'        : S2_md_t - S1_apd90_t,
                         'APD+1'     : S2_apd90_t - S2_md_t,
                         'cell_type' : self.cell_type,
                         'node_id'   : node_id })

            if self.debug:
                    print(data[-1])

                    md_i0   = np.abs(node.time - S1_md_t   ).argmin()
                    apd_i0  = np.abs(node.time - S1_apd90_t).argmin()
                    apd0_i0 = i1 + np.argmin(np.abs(node.AP[i1:apd_i0-20] - node.AP[apd_i0]))

                    md_i1   = np.abs(node.time - S2_md_t).argmin()
                    apd_i1  = np.abs(node.time - S2_apd90_t).argmin()
                    apd0_i1 = i2 + np.argmin(np.abs(node.AP[i2:apd_i1-20] - node.AP[apd_i1]))


                    plt.rcParams.update({'font.size': 18})
                    fig = plt.figure()
                    fig.suptitle(f'{self.cell_type} S1 {self.s1} S2 {s2}')

                    ax1 = fig.add_subplot(1, 1, 1)
                    ax1.plot(node.time[i1:ie], node.AP[i1:ie], 'k', linestyle='-', label='AP')
                    ymin, ymax = ax1.get_ylim()

                    ax1.axvspan(node.time[md_i0], node.time[apd_i0], facecolor='b', alpha=0.3, label="APD-1")
                    ax1.plot([node.time[apd0_i0], node.time[apd_i0]], [node.AP[apd_i0]]*2, '--', color='gray', label="APD-1")
                    ax1.text(np.mean([node.time[apd0_i0]-50, node.time[apd_i0]]), node.AP[apd_i0]+5, 'APD$_{t-1}$', fontdict={"usetex": True, 'size': 18} )

                    ax1.axvspan(node.time[apd_i0], node.time[md_i1], facecolor='g', alpha=0.3, label="DI")
                    ax1.plot([node.time[apd_i0], node.time[md_i1]], [min(node.AP[apd_i0], node.AP[md_i1])]*2, '--', color='gray', label="DI")
                    #ax1.text(np.mean([node.time[apd_i0], node.time[md_i1]])-10, min(node.AP[apd_i0], node.AP[md_i1])+5, 'DI', fontdict={"usetex": True,'size': 18} )

                    ax1.axvspan(node.time[md_i1], node.time[apd_i1], facecolor='b', alpha=0.3, label="APD")
                    ax1.plot([node.time[apd0_i1], node.time[apd_i1]], [node.AP[apd_i1]]*2, color='gray', linestyle='--', label="APD")
                    #ax1.text(np.mean([node.time[apd0_i1], node.time[apd_i1]])-50, node.AP[apd_i1]+5, 'APD', fontdict={"usetex": True,'size': 18} )
                    #ax1.legend()
                    plt.show()
                    input('Do you want to keep going on?(y/Ctrl+c)')

        self.APD_DI = self.APD_DI.append(data, ignore_index=True)

        if save_csv:
            self.save_APD_DI(w=w)
    #

    def save_APD_DI(self, w=False):

        save_dir = self.path+'/Rest_Curve_data'
        if not os.path.exists(save_dir) or not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fname = f"{save_dir}/APD_DI_APD+1_{self.s1}_{self.s2_step}.csv"
        if os.path.exists(fname) and not w:
            print(f"WARNING: {fname} already exists and overwrite is set to false, nothing will be saved....")
        else:
            if self.debug:
               print(self.APD_DI.to_markdown())
            self.APD_DI.to_csv(fname)
    #

    def compute_DI_CV(self, node_order=None, save_csv=False, w=False):

        if not check_attrs(self, ['nodes', 'cell_type'], "Can't compute CV and DI"):
            return

        if node_order is None:
            node_order = sorted(list(self.nodes), key=lambda n: self.nodes[n].loc[-1])

        data = []

        for kk, ii in enumerate(node_order[:-1]):

            node = self.nodes[ii]
            for s2, [i1,i2,ie] in zip(node.S2s, node.S1S2_gids):
                or_md_S1_t, or_S1_apd90_t = compute_max_der_and_perc_repolarization(node.time[i1:i2], node.AP[i1:i2], perc=0.9, show=self.debug)
                or_S2_md_t, _    = compute_max_der_and_perc_repolarization(node.time[i2:ie], node.AP[i2:ie], perc=0.9, show=self.debug)

                for jj in node_order[kk+1:]:
                    dest_node = self.nodes[jj]
                    aux = dest_node.S2s == s2
                    if aux.any():
                        s2i = aux.argmax()
                        [j1,j2,je]  = dest_node.S1S2_gids[s2i]
                        dest_md_S1_t, dest_S1_apd90_t = compute_max_der_and_perc_repolarization(dest_node.time[j1:j2], dest_node.AP[j1:j2], perc=0.9, show=self.debug)
                        dest_S2_md_t, _    = compute_max_der_and_perc_repolarization(dest_node.time[j2:je], dest_node.AP[j2:je], perc=0.9, show=self.debug)
                        d_ij = np.linalg.norm(node.loc - dest_node.loc)
                        data.append({'S1'       : self.s1,
                                     'S2'        : s2,
                                     'DI_or'     : or_S2_md_t - or_S1_apd90_t,
                                     'DI_dest'   : dest_S2_md_t - dest_S1_apd90_t,
                                     'APD_or'     : or_S1_apd90_t - or_md_S1_t,
                                     'APD_dest'   : dest_S1_apd90_t - dest_md_S1_t,
                                     'CV'        :  d_ij / (dest_S2_md_t - or_S2_md_t),
                                     'cell_type' : self.cell_type,
                                     'node_or'   : ii,
                                     'node_dest' : jj})

                        if self.debug:
                            print(f"Dist {ii} to  {jj} = {d_ij}")
                            print(data[-1])
                            print('Origen Y:', node.loc[1], 'T act:', or_S2_md_t)
                            print('Dest Y:', dest_node.loc[1], 'T act:', dest_S2_md_t)
                            p = pv.Plotter()
                            p.add_mesh(self.domain, color='white', opacity=0.5)
                            p.add_mesh(node.loc, color='red', render_points_as_spheres=True, point_size=10)
                            p.add_mesh(dest_node.loc, color='green', render_points_as_spheres=True, point_size=10)
                            p.camera_position = 'xy'
                            p.camera.azimuth = -180
                            p.show()

                            or_md_i = np.argwhere(node.time == or_S2_md_t)[0,0]
                            dest_md_i = np.argwhere(dest_node.time == dest_S2_md_t)[0,0]


                            fig = plt.figure()
                            fig.suptitle(f'{self.cell_type} S1 {self.s1} S2 {s2} at or_node {ii} dest_node {jj}')

                            ax1 = fig.add_subplot(1, 1, 1)
                            ax1.plot(node.time[i1:ie],      node.AP[i1:ie],      'red',  linestyle='-.',  label='Origin AP')
                            ax1.plot(dest_node.time[j1:je], dest_node.AP[j1:je], 'green', linestyle=':', label='Dest AP')
                            cv_t = [dest_S2_md_t, or_S2_md_t]
                            cv_ap = [dest_node.AP[dest_md_i], node.AP[or_md_i]]
                            ax1.plot(cv_t, cv_ap, 'k-', label='_nolegend')
                            ymin, ymax = ax1.get_ylim()
                            ax1.axvspan(dest_S2_md_t, or_S2_md_t, facecolor='yellow', alpha=0.3, label='lapse')
                            ax1.legend()
                            plt.show()
                            input('Do you want to keep going on?(y/Ctrl+c)')

        self.CV_DI = self.CV_DI.append(data, ignore_index=True)
        if save_csv:
            self.save_CV_DI(w=w)
    #

    def save_CV_DI(self, w=False):
        save_dir = self.path+'/Rest_Curve_data'
        if not os.path.exists(save_dir) or not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fname = f"{save_dir}/DI_CV_{self.s1}_{self.s2_step}.csv"
        if os.path.exists(fname) and not w:
            print(f"WARNING: {fname} already exists and overwrite is set to false, nothing will be saved....")
        else:
            if self.debug:
                print(self.CV_DI.to_markdown())
            self.CV_DI.to_csv(fname)
    #

    def signal_segmentation(self, node_id=None, w=False):

        if not check_attrs(self, ['nodes', 'cell_type'], "Can't compute signal segmentation"):
            return


        if node_id is None:
            for nid in self.nodes:
                self.signal_segmentation(node_id=nid, w=w)
            return

        node = self.nodes[node_id]

        if not check_attrs(node, ['S2s', 'S1S2_gids'], f"ERROR: Can't make signal segmentation at node {node_id}"):
            return

        save_dir = self.path+'/seg_stimuli'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for s2, [i1,i2,ie] in zip(node.S2s, node.S1S2_gids):

            fname = f"{save_dir}/{self.s1}_{int(s2)}_{self.cell_type}_{node_id}.json"

            if os.path.exists(fname) and not w:
                print(f"WARNING: {fname} already already exists, nothing will be written....")

            else:
                S1_md_t, S1_apd90_t = compute_max_der_and_perc_repolarization(node.time[i1:i2], node.AP[i1:i2], perc=0.9, show=self.debug)
                S2_md_t, S2_apd90_t = compute_max_der_and_perc_repolarization(node.time[i2:ie], node.AP[i2:ie], perc=0.9, show=self.debug)
                # Data to be written
                data={
                    "AP"        : node.AP[i1:i2].tolist(),
                    "AP+1"      : node.AP[i2:ie].tolist(),
                    "APD"       : S1_apd90_t - S1_md_t,
                    "DI"        : S2_md_t - S1_apd90_t,
                    "APD+1"     : S2_apd90_t - S2_md_t,
                    "t_act"     : node.time[i2] - node.time[i1],
                    "t_delta"   : node.time[1] - node.time[0],
                    "node_id"   : node_id,
                    "S1"        : self.s1,
                    "S2"        : s2,
                    "cell_type" : self.cell_type
                }

                with open(fname, "w") as outfile:
                    json.dump(data, outfile, indent=4)
                if self.debug:
                    plt.plot(node.time[i1:i2] - node.time[i1], node.AP[i1:i2], 'r', label='AP')
                    plt.plot(node.time[i2:ie] - node.time[i1], node.AP[i2:ie], 'g', label='AP+1')
                    plt.axvline(node.time[i2] - node.time[i1], color='y', label='t_act')
                    plt.show()
                    input('Do you want to keep going on?(y/Ctrl+c)')
    #


#

def EP_params_from_ELVIRA_simulation(path,
                                     s1,
                                     s2_ini=None,
                                     s2_end=250,
                                     s2_step=-20,
                                     s1_per_s2=9,
                                     tr_offset=0,
                                     comp_APD=False,
                                     comp_DI=False,
                                     seg_sig=False,
                                     cell_type=None,
                                     output_path=None,
                                     debug=False,
                                     rm_var=False,
                                     w=False):

    exp = SensElvExp()
    exp.path = path
    if output_path:
        exp.output_path=output_path
    else:
        exp.output_path = path+f'/output_{s1}_{s2_step}'

    exp.debug = debug

    if s2_ini is None:
        s2_ini = s1

    exp.s1 = s1
    exp.s2_ini  = s2_ini
    exp.s2_end  = s2_end
    exp.s2_step = s2_step
    exp.s1_per_s2 = s1_per_s2
    exp.tr_offset = tr_offset
    exp.cell_type = cell_type

    if comp_APD or comp_DI or seg_sig:
        exp.load_node_info()
        exp.save_nodes_npy(rm_var=rm_var)
        exp.extract_S1S2()
        if comp_APD:
            exp.compute_APD_DI(save_csv=True, w=w)
        if comp_DI:
            exp.compute_DI_CV(save_csv=True, w=w)
        if seg_sig:
            exp.signal_segmentation(w=w)
    else:
        print("WARNING: comp_APD and comp_DI and seg_sig are False... . Nothing will be computed...")

#


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Script to create ELVIRA
                                    cases, using a template case.""",
                                    usage=""" """)

    parser.add_argument('-s',
                        '--s1',
                        dest='s1',
                        action='store',
                        default=1000,
                        type=int,
                        help="""Lapse of time between main consecutive stimuli
                        expressed in ms.""")

    parser.add_argument('--s2-end',
                        dest='s2_end',
                        default=250,
                        action='store',
                        type=float,
                        help="""Last reached value of S2.""")

    parser.add_argument('--s2-ini',
                        dest='s2_ini',
                        type=float,
                        default=None,
                        action='store',
                        help="""Initial S2 for the stimuli train.""")

    parser.add_argument('-p',
                        '--s2-step',
                        dest='s2_step',
                        default=20,
                        action='store',
                        type=int,
                        help="""Lapse of time between the 10th S1 and the S2 stimuli
                        expressed in ms. Default 20""")

    parser.add_argument('--tr-off',
                        dest='tr_off',
                        type=float,
                        default=0,
                        action='store',
                        help="""An extra offset time at the end of each stimuli train.""")

    parser.add_argument('--s1-per-s2',
                        dest='s1_per_s2',
                        type=int,
                        default=9,
                        action='store',
                        help="""Number of S1 stimuli before each S2.""")

    parser.add_argument('-A',
                        '--APD-DI',
                        dest='A',
                        action='store_true',
                        help=""" Compute APD DI data and save it. """)

    parser.add_argument('-C',
                        '--CV-DI',
                        dest='C',
                        action='store_true',
                        help=""" Compute CV DI data and save it. """)

    parser.add_argument('-S',
                        '--signal-seg',
                        dest='sig_seg',
                        action='store_true',
                        help=""" Compute the signal segmentation and save it. """)

    parser.add_argument('-m',
                        '--myo',
                        dest='myo',
                        type=str,
                        help="""Flag to specify if the cellular model, should be in
                        {'CM', 'ttendo', 'ttmid', 'ttepi', 'ttendobz', 'ttmidbz', 'ttepibz' .""")

    parser.add_argument('--output-path',
                        dest='otp_path',
                        type=str,
                        nargs='?',
                        default=None,
                        help=""" The path to output dir, if it does not conform to standard naming.""")

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

    parser.add_argument('-r',
                        '--rm-var',
                        dest='rm_var',
                        action='store_true',
                        help="""Flag to remove var files after saving the corresponding npy file...""")

    parser.add_argument('path',
                        action='store',
                        type=str,
                        nargs='?',
                        help="""Path to an existing case or to a new case to be created.""")



    args = parser.parse_args()

    EP_params_from_ELVIRA_simulation(path=args.path,
                                     s1=args.s1,
                                     s2_ini=args.s2_ini,
                                     s2_end=args.s2_end,
                                     s2_step=args.s2_step,
                                     s1_per_s2=args.s1_per_s2,
                                     tr_offset=args.tr_off,
                                     comp_APD=args.A,
                                     comp_DI=args.C,
                                     seg_sig=args.sig_seg,
                                     cell_type=args.myo,
                                     output_path=args.otp_path,
                                     debug=args.debug,
                                     w=args.w)
