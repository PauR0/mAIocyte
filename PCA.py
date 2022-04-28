#!/usr/bin/env python3
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.utils import resample


import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class STDPCA:
    """
    Standardized PCA.
    It is a regular PCA, but the coefficients are divided by the variance of the principal direction
    """

    def __init__(self, X=None, y = None, n_dim = None):

        self.X : np.ndarray = None
        self.X_tr : np.ndarray = None
        self.y : np.ndarray = None

        self.n_dim : int = None
        self.mean : np.ndarray = None
        self.pca_mat : np.ndarray = None
        self.pca_var : np.ndarray = None
        self.pca_obj = None

        if X is not None:
            self.fit_X(X)
        if y is not None:
            self.y = y
        if n_dim is not None:
            self.n_dim = n_dim
    #

    def fit_X(self, X=None, transform_X=False):

        if X is not None:
            self.X = X

        self.pca_obj = PCA(svd_solver = 'full')
        self.pca_obj.fit(self.X)
        self.mean = self.pca_obj.mean_
        self.pca_var=self.pca_obj.explained_variance_
        self.pca_mat=self.pca_obj.components_
        if transform_X:
            self.X_tr = self.transfrom(self.X, n_dim=self.pca_obj.n_components_)
    #

    def set_X(self, X, transform_X=False):
        """
        Set the dataset to perform PCA or where PCA has been
        performed if it has been loaded.

        Arguments:
        -------------
            X : array (n_samples, n_features)

        """

        self.X = X
        if transform_X:
            self.X_tr = self.transfrom(X)
    #

    def get_X(self):
        return self.X
    #

    def get_X_tr(self, n_dim=None):

        if n_dim is None and self.n_dim is not None:
            n_dim = self.n_dim

        return self.X_tr[:, :n_dim]
    #

    def set_y(self, y):
        self.y = y
        return
    #

    def get_y(self):
        return self.y
    #

    def set_n_dim(self, n):
        self.n_dim = n
        return
    #

    def get_n_dim(self):
        return self.n_dim
    #

    def transfrom(self, X, n_dim=None):
        """
        Transform dataset to PCA space
        Args:
        -----
            X : array (n_samples, n_features)

            n_dim : int (optional)
                Default to self.n_dim. The desired dimensionality in the PCA space.

        Returns:
        --------
            X_tr : list[array] or array
        """

        if n_dim is None:
            n_dim =  self.n_dim

        X_tr=[]
        for x in X:
            x_tr = self.pca_mat[:n_dim].dot(x)
            c = x_tr / np.sqrt(self.pca_var[:n_dim])
            X_tr.append(c)

        if len(X_tr)==1:
            return np.array(X_tr[0])
        else:
            return np.array(X_tr)
    #

    def inv_transform(self, X):
        """
        Transform dataset from PCA space
        Args:
        -----
            X : array (n_samples, n_coeffs)

        Returns:
        --------
            X_tr : list[array] or array
        """

        if X.shape[1] != self.n_dim:
            n = X.shape[1]
        else:
            n = self.n_dim

        if len(X.shape) < 1:
            return False

        if len(X.shape) == 1:
            X = [X]

        X_tr = []

        for x in X:
            x_tr = x * np.sqrt(self.pca_var[:n]).flatten()
            hd = self.mean + (x_tr * self.pca_mat[:n].T).T.sum(axis=0)
            X_tr.append(hd)

        if len(X_tr)==1:
            return X_tr[0]

        return np.array(X_tr)
    #

    def ppal_dirs(self, n_comp = None, sigmas = None, show = False, x=None):
        """
        Make a plot to ilustrate the first n_comp

        Arguments:
        ------------
            n_comp : int, optional
                Default self.n_dim. Starting from 0 the number of first principal components to show.

            sigmas : iterable of floats, optional
                Default [-1,1]. It can be a single list of floats that will be used for all the component of chosen by
                n_comp or a list containing n_comp lists of sigmas.

            show : bool, optional
                Whether display a n_comp x 1 plot with the pincipal components shape.

            x : array-like (n_features,), optional
                This option is only used if show==True. The independent variable of the plot.

        Returns:
        ---------
            ppal_dirs : array (n_comp, M)
                An array containing the principal components weighted by sigmas.
        """

        if n_comp is None:
            n_comp = self.n_dim

        if sigmas is None:
            sigmas = [-1,1]

        ppal_dirs = []
        for i in range(n_comp):
            for j in sigmas:
                d = np.zeros((1,n_comp))
                d[:,i] = j
                hd = self.inv_transform(d)
                ppal_dirs.append(hd)
        if show:
            _, axes = plt.subplots(n_comp, 1)
            if x is None:
                x = np.arange(0, len(hd))
            for i, ax in enumerate(axes):
                ax.plot(x, ppal_dirs[2*i], color='tab:red', label = f'$\mu {sigmas[0]}M_{{{i}}}$' )
                ax.plot(x, ppal_dirs[2*i+1], color='tab:blue', label = f'$\mu + {sigmas[1]}M_{{{i}}}$')
                ax.plot(x, self.mean, color='tab:purple', label = f'$\mu$')
            plt.legend()
            plt.show()

        return np.array(ppal_dirs)
    #

    def plot_pca_variance(self, n_comp=None, thrs=None):
        """
        Make a plot with the variance explained by each of the principal components

        Arguments:
        --------------

            n_comp : int, optional
                Defaul self.n_dim. The first n_comp principal directions will be plotted.

            thrs : float, optional
                Default None. A float between 0 and 1 representing the percentage of variance desired
                to be displayed as a horizontal line.

        """


        if n_comp is None:
            n_comp = self.n_dim

        plt.style.use('seaborn')
        plt.rcParams.update({'font.size': 16})
        fig, ax1 = plt.subplots()

        color = 'b'
        ax1.set_xticks(range(n_comp))
        ax1.set_xticklabels([f'c{i}' for i in range(n_comp)])
        ax1.set_xlabel('n components')
        ax1.set_ylabel('variance per component', color=color)
        ax1.bar(np.arange(0, n_comp), self.pca_obj.explained_variance_[:n_comp], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'g'
        ax2.set_ylabel('cummulative variance in %', color=color)  # we already handled the x-label with ax1
        var = [np.sum(self.pca_obj.explained_variance_ratio_[:n+1])*100 for n in range(n_comp)]
        ax2.plot(np.arange(0, n_comp), var, 'k-o', mec=color, mfc='k')
        if thrs:
            ax2.axhline(thrs, linestyle='-.', color='gray')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        plt.style.use('default')
    #

    def pairplot(self, show=True, n_dim = None, X_t = None, y_t = None):

        if n_dim is None:
            n_dim = self.n_dim

        X_ld = self.transfrom(self.X)
        if X_t is not None:
            X_tld = self.transfrom(X_t)
            X_ld = np.concatenate( (X_ld, X_tld), axis=0 )

        Y = 0
        if self.y is not None:
            Y = self.y
        if X_t is not None and y_t is not None:
            Y = np.concatenate( (Y, y_t), axis=0 )

        ld_X_df = pd.DataFrame(X_ld, columns = [f'c{i}' for i in range(self.n_dim)])
        ld_X_df['class'] = Y

        sns.pairplot(data=ld_X_df, hue='class', vars=[f'c{i}' for i in range(self.n_dim)])
        if show:
            plt.show()
    #

    def save_pca_data(self, path, abs_path=False):
        """

        TODO:Document

        """
        if abs_path:
            fname = path
        else:
            fname = path + '/pca_info.npy'

        pca_mat = (self.pca_mat.T * self.pca_var).T
        pca_info = np.concatenate([self.mean.reshape(1,-1), pca_mat], axis=0)
        np.save(fname, pca_info)
    #

    def load_pca_data(self, path, abs_path=False):
        """

        TODO:Document

        """

        if abs_path:
            fname = path
        else:
            fname = path + '/pca_info.npy'

        pca_info = np.load(fname)
        self.mean = pca_info[0]
        self.pca_var = np.linalg.norm(pca_info[1:], axis=1)
        self.pca_mat = (pca_info[1:].T / self.pca_var).T
#

#End STDPCA


def make_random_gaussian_coords(coords, n=5):

    m = coords.mean(axis=0)
    cov = np.cov(coords.T)
    coords = np.random.multivariate_normal(m, cov, size=n).T

    return coords
#

def make_random_bootstraped_coords(coords, n=5):

    btstrp_coords = np.array([resample(coords.T[i], n_samples=n) for i in range(coords.shape[1])]).T

    return btstrp_coords
#

def make_random_uniform_coords(coords, n=5):

    low = coords.min(axis=0)
    upp = coords.max(axis=0)

    unif_coords = np.array([np.random.uniform(low=l, high=u, size=n) for l, u in zip(upp, low)]).T

    return unif_coords
#
