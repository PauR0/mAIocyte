import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree



class RegresorBase:

    def __init__(self, reg=None, X=None,Y=None, train=False):
        pass
    #

    def fit(self, X,Y):
        pass
    #

    def predict_next_curve(self, fs):
        pass
    #

    def predict(self, fs):

        if fs.ndim < 1:
            return np.array([self.predict_next_curve(x) for x in fs])

        return np.array(self.predict_next_curve(fs))
    #

#

class ActPotCurveBase:

    def __init__(self, **kwargs):
        #Time bounds
        self.t_min = 0
        self.t_max = 1000
    #

    def __call__(self, t):
        return self.eval_curve(t)
    #

    def build_curve(self, s):
        pass
    #

    def eval_curve(self, t):
        pass
    #
#

class LUTActPotCurve(ActPotCurveBase):
    """
    Class for 8 point model of action potential curve.
    The 8 points correspond to:

        1,2) First peak
        3,4) local minimum after 1,2)
        5,6) Local maxima after 3,4)
        7,8) APDq the 60% of the repolarization
    """

    def __init__(self, curves_DS, curves_fvs, t_delta):

        super().__init__()
        self.curves_DS = curves_DS
        self.curves_fvs = curves_fvs
        self.current_curve = None
        self.t_delta=t_delta
    #

    def from_dataframe_pickle(self, fname, curve_col = 'AP+1'):

        seg_stim_HD = pd.read_pickle(fname)
        self.curves_DS = np.array(seg_stim_HD[curve_col].tolist())
        self.t_delta = seg_stim_HD.loc[0, 't_delta']
    #

    def build_curve(self, s):
        """
        For the 8 point model, the curve is selected
        from those available in the dataset.

        Arguments:
        ----------
            s : int
                The id of the new curve.

        Returns:
        --------
            current_curve : list[float]
                The values of Action Potential (mV) for 1000 seconds sampled each t_delta.
        """
        i =  np.argwhere( (self.curves_fvs == s).all(axis=1) ).ravel()[0]
        self.current_curve = self.curves_DS[i,:]
        return self.current_curve
    #

    def eval_curve(self, t):
        """
        This function returns the action potential values for given times.
        Action potential values are consulted no computed, they are only available
        each t_delta.
        """
        t[t < self.t_min] = self.t_min
        t[t > self.t_max] = self.t_max
        ids = (t/self.t_delta).astype(int)

        return self.current_curve[ids]
    #
#

class LUTRegresor(RegresorBase):
    """
    TODO:Document this!
    """

    def __init__(self, X=None, Y=None, train=False):
        super().__init__()
        self.full_states = X
        #Currently new states is not used...
        self.new_states  = Y
        self.DI_kdt : KDTree = None
        self.n_di_neighs = 5 #Magic Number!
        if train:
            self.fit()
    #

    def fit(self, X=None, Y=None):

        if X is not None:
            self.full_states = X
        if Y is not None:
            self.new_states = Y

        self.DI_kdt = KDTree(self.full_states[:,0].reshape(-1,1))
    #

    def predict_next_curve(self, fs):
        """
        The state following a given full state (DI, state), is
        computed hierarchically. First the method computes the
        self.n_di_neighs most simmilar DI of the X dataset. Then
        it takes the most simmilar state, i.e. the closest curve
        on the feature space.
        """

        di, s = fs[[0]], fs[1:]

        ii = self.DI_kdt.query([di], k=self.n_di_neighs, return_distance=False).ravel()
        neighs = self.full_states[ii,1:]
        #print(np.array2string(self.full_states[ii], precision=2, floatmode='fixed'))
        closest_state_i = np.linalg.norm(neighs - s, axis=1).argmin()
        new_state = self.new_states[ii[closest_state_i]]
        #print(np.array2string(self.full_states[ii[closest_state_i]], precision=2, floatmode='fixed'))
        return new_state
    #
#


class Cell:

    def __init__(self, state_reg, act_pot_curve, s0=None, **kwargs):

        self.id = None

        self.StateReg    = state_reg
        self.ActPotCurve = act_pot_curve
        self.state       = s0
        self.di          = None
        self.fstate      = None

        self.apd                  = None #ms
        self.apd_percentage       = 0.9  #%
        self.max_depolarization_t = 50   #ms
    #

    def restart(self):
        self.__init__(state_reg=self.StateReg, act_pot_curve=self.ActPotCurve)
    #

    def __str__(self):
        desc_str = f"Cell {self.id}:\n"     +\
                   f"\tDI    = {self.di}\n"  +\
                   f"\tAPD   = {self.apd}\n" +\
                   f"\tState = {self.state}\n"
        return desc_str
    #

    def update_curve_from_state(self):
        self.ActPotCurve.build_curve(self.state)
        self.apd_from_curve()
    #

    def apd_from_curve(self):

        t = np.linspace(self.max_depolarization_t, self.ActPotCurve.t_max, 1000)
        ap = self.ActPotCurve(t)
        apmax = ap.max()
        apmin = ap.min()
        ap_sought = apmin + (apmax - apmin) * (1-self.apd_percentage)
        i = np.argmin(np.abs(ap - ap_sought)).ravel()[0]
        self.apd = t[i]
    #

    def set_state(self, x):
        self.state = x
        self.update_curve_from_state()
    #

    def update_di(self, di):
        self.di = di
        self.fstate = np.hstack([[self.di], self.state])
    #

    def get_next_state(self, di, update=True):

        self.update_di(di)
        new_st = self.StateReg.predict_next_curve(self.fstate)
        if update:
            self.set_state(new_st)

        return new_st
    #

    def eval_curve(self, t):
        return self.ActPotCurve(t)
    #
#
