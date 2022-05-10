
import os

import json

import argparse

import numpy as np

import pandas as pd

from fnmatch import fnmatch

from peaks import peakdetect

import matplotlib.pyplot as plt

import pyvista as pv


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
        plt.plot(t, ap)
        plt.plot([t_max_der, apdq_t], ap[[max_der, apdq_i]])
        plt.show()

    if full_output:
        return [t_max_der, ap[max_der]], [apdq_t, ap[apdq_i]]

    return t_max_der, apdq_t
#

def check_attrs(obj, attr_names, err_mesage):

    for attr in attr_names:
            if not hasattr(obj, attr):
                print(f"{err_mesage}, {attr} has not been set")
                return

    return True
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
        self.__peaks : np.ndarray = None
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

    def plot(self, ax=None, show=False):

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.time, self.AP, 'k', label='Action Potential')
        ax.scatter(self.peaks['max'][0], self.peaks['max'][1], c='r', marker='*', label='Max Peaks')
        ax.scatter(self.peaks['min'][0], self.peaks['min'][1], c='b', marker='*', label='Min Peaks')
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

        self.__s1 : int = None
        self.__s2_step : int = None
        self.s1_per_s2 : int = 9
        self.min_s2 : int = 250
        self.DI_CV_df : pd.DataFrame = pd.DataFrame(columns=['S1', 'S2', 'DI_or', 'DI_dest', 'CV', 'cell_type', 'node_or', 'node_dest'])
        self.APD_DI_df : pd.DataFrame = pd.DataFrame(columns=['S1', 'S2', 'APD', 'DI', 'APD+1', 'cell_type', 'node_id'])
        self.stimuli_segmented : pd.DataFrame = pd.DataFrame(columns=['S1', 'S2', 'DI', 'AP+1', 't_act', 'AP', 'cell_type', 'node_id'])
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

    #Cell type used for the experiment, should be in {'ttepi', 'ttmid' ,'ttendo', 'ttepi_bz', 'ttmid_bz' ,'ttendo_bz'}.
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

    #S1
    @property
    def s1(self):
        return self.__s1
    @s1.setter
    def s1(self, s):
        self.__s1 = s
    @s1.deleter
    def s1(self):
        del self.__s1
    #

    #S2 decreasing step
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

    #S2 decreasing step
    @property
    def s2_step(self):
        return self.__s2_step
    @s2_step.setter
    def s2_step(self, st):
        self.__s2_step = st
    @s2_step.deleter
    def s2_step(self):
        del self.__s2_step
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

    def save_nodes_npy(self):

        if not check_attrs(self, ['path', 'output_path', 'nodes'], 'Cant save data'):
            return

        for name, node in self.nodes.items():

            arr = np.zeros((node.time.shape[0], 2))
            arr[:, 0] = node.time
            arr[:, 1] = node.AP
            fname = f'{self.output_path}/node_{name}.npy'
            np.save(fname, arr)
    #

    def get_stimuli_of_interest(self, node_id, t_ini, t_end, s2, show=False):

        """
        TODO:Document propperly

        This function returns the index of 3 minpeaks and 2 maxpeaks of the last two stimuli of a stimuli train
        in the node.time and node.AP array.

        Args:
        ------
            node_id : str
                Key of the node in self.node dict

            t_ini, t_end : float

            s2 : int

        Returns:
        ---------
            mpoi : list[int]
                List of minpeak times

            Mpoi : list[int]
                List of maxpeak times
        """
        node = self.nodes[node_id]

        #Get peaks between t_ini and t_end
        max_peaks_ids = (node.peaks['max'][0] >= t_ini) & (node.peaks['max'][0] < t_end)
        n_peaks = np.sum(max_peaks_ids, dtype=int)

        #75% of the expected maxpeaks
        if n_peaks < self.s1_per_s2*0.75:
            print(f"WARNING: cell type {self.cell_type}, s1 {self.s1}, s2 {s2} at node {node_id}: \n\tBetween {t_ini} and {t_end} node {node_id} has {n_peaks} peaks instead of the {self.s1_per_s2 + 1} expected, APD-DI-APD wont be computed.....")
            return None, None

        #time of the previous minpeak to t_end, this is expected to be S2
        ps2_t = np.max(node.peaks['min'][0][(node.peaks['min'][0] < t_end)])
        #id in time list of ps2 in the full list
        ps2_i = np.argwhere(node.time == ps2_t)[0,0]
        #id in time list of ps2 in the peaks list
        ps2_ii = np.argmax(node.peaks['min'][0][(node.peaks['min'][0] < t_end)])

        #peaks of interest
        s2_Mpi = np.argmin(np.abs(node.peaks['max'][0] - ps2_t)) #The next maxpeak to the S2 minpeak
        if node.peaks['max'][0][s2_Mpi] < ps2_t:
            s2_Mpi += 1

        Mpoi_t = node.peaks['max'][0][[s2_Mpi-1, s2_Mpi]]
        Mpoi_i = [np.argwhere(node.time == t)[0, 0] for t in Mpoi_t]
        mpoi_t = node.peaks['min'][0][[ps2_ii-1, ps2_ii, ps2_ii+1]] #The time of the minpeaks sorrounding the S2
        mpoi_i = [np.argwhere(node.time == t)[0,0] for t in mpoi_t] #The ids of the minpeaks sorrounding the S2

        if show:
            t_ini_i = np.argwhere(node.time == t_ini)[0,0]
            t_end_i = np.argwhere(node.time == t_end)[0,0]
            plt.title(f'{self.cell_type} S1 {self.s1} S2 {s2} at node {node_id}')
            plt.plot(node.time[t_ini_i:t_end_i], node.AP[t_ini_i:t_end_i], 'gray', linestyle='-.')
            plt.plot(node.time[mpoi_i[0]:mpoi_i[-1]], node.AP[mpoi_i[0]:mpoi_i[-1]])
            max_peaks_ids = (node.peaks['max'][0] >= t_ini) & (node.peaks['max'][0] < t_end)
            plt.plot(node.peaks['max'][0][max_peaks_ids], node.peaks['max'][1][max_peaks_ids], 'r*')
            plt.scatter(mpoi_t, node.AP[mpoi_i])
            plt.plot(Mpoi_t, node.AP[Mpoi_i], 'g*')
            plt.show()

        if mpoi_t[2]-mpoi_t[1]> self.s1+s2/2:
            print(f"WARNING: cell type {self.cell_type}, s1 {self.s1}, s2 {s2} at node {node_id}:",
                    f"The last activation was {mpoi_t[2]-mpoi_t[1]}, which is greater than s1+s2/2= {self.s1+s2/2}. APD-DI-APD wont be computed.....")
            return None, None


        return mpoi_i, Mpoi_i
    #

    def compute_APD_DI_df(self, node_id=None, save_csv=False, w=False):

        if not check_attrs(self, ['nodes', 's1', 's2_step', 's1_per_s2', 'min_s2', 'cell_type'], "Can't compute APD and DI"):
            return


        if node_id is None:
            for nid in self.nodes.keys():
                self.compute_APD_DI_df(node_id=nid, save_csv=False)
            if save_csv:
                self.save_APD_DI_df(w=w)
            return

        node = self.nodes[node_id]
        data = []

        s2 = self.s1 - self.s2_step
        t_ini = 0
        t_end =int(t_ini + (self.s1_per_s2 * self.s1) + s2/2)

        end = False

        while not end:
            ps2 = np.argmax(node.peaks['min'][0][(node.peaks['min'][0] < t_end)]).astype(int) #previous minpeak to t_end, this is expected to be S2
            max_ders = []
            apd90s = []
            poi, _ = self.get_stimuli_of_interest(node_id, t_ini=t_ini, t_end=t_end, s2=s2, show=self.debug)
            if poi is not None:
                for i, p in enumerate(poi[:-1]):
                    t = node.time[p:poi[i+1]]
                    ap = node.AP[p:poi[i+1]]
                    md_t, apd90_t = compute_max_der_and_perc_repolarization(t, ap, perc=0.9, show=self.debug)
                    max_ders.append(md_t)
                    apd90s.append(apd90_t)
                data.append({'S1'        : self.s1,
                            'S2'        : s2,
                            'APD'       : apd90s[0] - max_ders[0],
                            'DI'        : md_t - apd90s[0],
                            'APD+1'     : apd90_t - md_t,
                            'cell_type' : self.cell_type,
                            'node_id'   : node_id })

            s2 -= self.s2_step
            if ps2 < len(node.peaks['min'][0])-1:
                t_ini = int(node.peaks['min'][0][ps2+1])
                t_end = int(t_ini + self.s1_per_s2 * self.s1 + s2/2)

            if s2 <= self.min_s2:
                end = True
            if t_end >= np.max(node.time):
                end = True

        self.APD_DI_df = self.APD_DI_df.append(data, ignore_index=True)

        if save_csv:
            self.save_APD_DI_df(w=w)
    #

    def save_APD_DI_df(self, w=False):

        save_dir = self.path+'/Rest_Curve_data'
        if not os.path.exists(save_dir) or not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fname = f"{save_dir}/APD_DI_APD+1_{self.s1}_{self.s2_step}.csv"
        if os.path.exists(fname) and not w:
            print(f"WARNING: {fname} already exists and overwrite is set to false, nothing will be saved....")
        else:
            if self.debug:
               print(self.APD_DI_df.to_markdown())
            self.APD_DI_df.to_csv(fname)
    #

    def compute_DI_CV(self, node_order=None, save_csv=False, w=False):

        if not check_attrs(self, ['nodes', 's1', 's2_step', 's1_per_s2', 'min_s2', 'cell_type'], "Can't compute CV and DI"):
            return

        if node_order is None:
            node_order = sorted([n for n in self.nodes.keys()])

        data = []

        for kk, ii in enumerate(node_order[:-1]):

            node = self.nodes[ii]

            s2 = self.s1 - self.s2_step
            t_ini = 0
            t_end = int(t_ini + (self.s1_per_s2 * self.s1) + s2/2)

            end = False
            while not end:
                ps2 = np.argmax(node.peaks['min'][0][(node.peaks['min'][0] < t_end)]).astype(int) #previous minpeak to t_end, this is expected to be S2
                apd90s = []
                mpoi, Mpoi = self.get_stimuli_of_interest(ii, t_ini=t_ini, t_end=t_end, s2=s2, show=False)
                if mpoi is not None:

                    for i, p in enumerate(mpoi[:-1]):
                        t = node.time[p:mpoi[i+1]]
                        ap = node.AP[p:mpoi[i+1]]
                        or_md_t, apd90_t = compute_max_der_and_perc_repolarization(t, ap, perc=0.9, show=False)
                        apd90s.append(apd90_t)

                    for jj in node_order[kk+1:]:
                        dest_node = self.nodes[jj]
                        d_ij = np.linalg.norm(node.loc - dest_node.loc)
                        dest_apd90s = []
                        dest_mpoi, dest_Mpoi = self.get_stimuli_of_interest(jj, t_ini=t_ini, t_end=t_end, s2=s2, show=False)
                        if dest_mpoi is not None:
                            for i, p in enumerate(dest_mpoi[:-1]):
                                t = dest_node.time[p:dest_mpoi[i+1]]
                                ap = dest_node.AP[p:dest_mpoi[i+1]]
                                dest_md_t, dest_apd90_t = compute_max_der_and_perc_repolarization(t, ap, perc=0.9, show=False)
                                dest_apd90s.append(dest_apd90_t)
                            data.append({'S1'       : self.s1,
                                         'S2'        : s2,
                                         'DI_or'     : or_md_t - apd90s[0],
                                         'DI_dest'   : dest_md_t - dest_apd90s[0],
                                         'CV'        :  d_ij / (dest_md_t - or_md_t),
                                         'cell_type' : self.cell_type,
                                         'node_or'   : ii,
                                         'node_dest' : jj})

                            #if data[-1]['DI_dest']<23:
                            if self.debug:
                                print(f"Dist {ii} to  {jj} = {d_ij}")
                                print(data[-1])
                                print('Origen Y:', node.loc[1], 'T act:', or_md_t)
                                print('Dest Y:', dest_node.loc[1], 'T act:', dest_md_t)
                                p = pv.Plotter()
                                p.add_mesh(self.domain, color='white', opacity=0.5)
                                p.add_mesh(node.loc, color='red', render_points_as_spheres=True, point_size=10)
                                p.add_mesh(dest_node.loc, color='green', render_points_as_spheres=True, point_size=10)
                                p.camera_position = 'xy'
                                p.camera.azimuth = -180
                                p.show()

                                or_t_inii = np.argwhere(node.time == t_ini)[0,0]
                                or_t_endi = np.argwhere(node.time == t_end)[0,0]
                                dest_t_inii = np.argwhere(dest_node.time == t_ini)[0,0]
                                dest_t_endi = np.argwhere(dest_node.time == t_end)[0,0]

                                or_md_i = np.argwhere(node.time == or_md_t)[0,0]
                                dest_md_i = np.argwhere(dest_node.time == dest_md_t)[0,0]


                                fig = plt.figure()
                                fig.suptitle(f'{self.cell_type} S1 {self.s1} S2 {s2} at or_node {ii} dest_node {jj}')

                                ax1 = fig.add_subplot(1, 1, 1)
                                ax1.plot(node.time[or_t_inii:or_t_endi], node.AP[or_t_inii:or_t_endi], 'gray', linestyle='-.', label='_nolegend')
                                ax1.plot(node.time[mpoi[0]:mpoi[-1]], node.AP[mpoi[0]:mpoi[-1]], color='red', label='Origin AP')
                                ax1.plot(dest_node.time[dest_t_inii:dest_t_endi], dest_node.AP[dest_t_inii:dest_t_endi], 'gray', linestyle='-.', label='_nolegend')
                                ax1.plot(dest_node.time[dest_mpoi[0]:dest_mpoi[-1]], dest_node.AP[dest_mpoi[0]:dest_mpoi[-1]], color='green', label='Dest AP')
                                cv_t = [dest_md_t, or_md_t]
                                cv_ap = [dest_node.AP[dest_md_i], node.AP[or_md_i]]
                                ax1.plot(cv_t, cv_ap, 'k-', label='_nolegend')
                                ymin, ymax = ax1.get_ylim()
                                ax1.vlines(cv_t, ymin, ymax, 'gray', '-.', label='_nolegend')
                                ax1.legend()
                                plt.show()


                s2 -= self.s2_step
                if ps2 < len(node.peaks['min'][0])-1:
                    t_ini = int(node.peaks['min'][0][ps2+1])
                    t_end = int(t_ini + self.s1_per_s2 * self.s1 + s2/2)
                if s2 <= self.min_s2:
                    end = True
                if t_end >= np.max(node.time):
                    end = True
            #endwhile

        self.DI_CV_df = self.DI_CV_df.append(data, ignore_index=True)
        if save_csv:
            self.save_DI_CV_df(w=w)
    #

    def save_DI_CV_df(self, w=False):
        save_dir = self.path+'/Rest_Curve_data'
        if not os.path.exists(save_dir) or not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        fname = f"{save_dir}/DI_CV_{self.s1}_{self.s2_step}.csv"
        if os.path.exists(fname) and not w:
            print(f"WARNING: {fname} already exists and overwrite is set to false, nothing will be saved....")
        else:
            if self.debug:
                print(self.DI_CV_df.to_markdown())
            self.DI_CV_df.to_csv(fname)
    #

    def signal_segmentation(self, node_id=None, w=False):

        if not check_attrs(self, ['nodes', 's1', 's2_step', 's1_per_s2', 'min_s2', 'cell_type'], "Can't compute APD and DI"):
            return


        if node_id is None:
            for nid in self.nodes.keys():
                self.signal_segmentation(node_id=nid, w=w)
            return
        node = self.nodes[node_id]


        save_dir = self.path+'/seg_stimuli'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        s2 = self.s1 - self.s2_step
        t_ini = 0
        t_end =int(t_ini + (self.s1_per_s2 * self.s1) + s2/2)

        end = False
        curr_files=os.listdir(save_dir)
        while not end:

            ps2 = np.argmax(node.peaks['min'][0][(node.peaks['min'][0] < t_end)]).astype(int) #previous minpeak to t_end, this is expected to be S2
            fname = f"{save_dir}/{self.s1}_{s2}_{self.cell_type}_{node_id}.json"

            if os.path.exists(fname) and not w:
                print(f"WARNING: {fname} already already exists, nothing will be written....")
            else:
                poi, _ = self.get_stimuli_of_interest(node_id, t_ini=t_ini, t_end=t_end, s2=s2, show=self.debug)
                if poi is not None:
                    # Data to be written
                    data={
                        "AP" : node.AP[poi[0]:poi[1]].tolist(),
                        "AP+1" : node.AP[poi[1]:poi[2]].tolist(),
                        "t_act" : node.time[poi[1]] - node.time[poi[0]],
                        "t_delta" : node.time[1] - node.time[0],
                        "node_id" : node_id,
                        "S1" : self.s1,
                        "S2" : s2,
                        "cell_type" : self.cell_type
                    }

                    with open(fname, "w") as outfile:
                        json.dump(data, outfile, indent=4)

            s2 -= self.s2_step
            if ps2 < len(node.peaks['min'][0])-1:
                t_ini = int(node.peaks['min'][0][ps2+1])
                t_end = int(t_ini + self.s1_per_s2 * self.s1 + s2/2)

            if s2 <= self.min_s2:
                end = True
            if t_end >= np.max(node.time):
                end = True
    #


#

def EP_params_from_ELVIRA_simulation(path, s1, s2_step, cell_type, output_path=None, debug=False, w=False):

    exp = SensElvExp()
    exp.path = path
    if output_path:
        exp.output_path=output_path
    else:
        exp.output_path = path+f'/output_{s1}_{s2_step}'

    exp.debug = debug
    exp.s1 = s1
    exp.s2_step = s2_step
    exp.cell_type = cell_type

    exp.load_node_info()
    exp.save_nodes_npy()
    #exp.compute_APD_DI_df(save_csv=True, w=w)
    #exp.compute_DI_CV(save_csv=True, w=w)
    exp.signal_segmentation(w=w)
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

    parser.add_argument('-p',
                        '--s2-step',
                        dest='s2_step',
                        default=20,
                        action='store',
                        type=int,
                        help="""Lapse of time between the 10th S1 and the S2 stimuli
                        expressed in ms. Default 20""")

    parser.add_argument('-m',
                        '--myo',
                        dest='myo',
                        type=str,
                        help="""Flag to specify if ttepi - ttmid - ttendo.""")

    parser.add_argument('-b',
                        '--border-zone',
                        dest='bz',
                        action='store_true',
                        help="""This flag is used to indicate if cell type should be considered border zone. Default is False.""")

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

    parser.add_argument('path',
                        action='store',
                        type=str,
                        nargs='?',
                        help="""Path to an existing case or to a new case to be created.""")



    args = parser.parse_args()

    cell_type = args.myo
    if args.bz:
        cell_type += '_bz'

    EP_params_from_ELVIRA_simulation(path=args.path, s1=args.s1, s2_step=args.s2_step, cell_type=cell_type, output_path=args.otp_path, debug=args.debug, w=args.w)
