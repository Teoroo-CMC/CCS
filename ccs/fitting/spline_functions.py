#------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2021  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
#------------------------------------------------------------------------------#


'''
This module contains functions for spline construction, evaluation and output.
'''


import logging
import numpy as np
import json
import bisect
import copy
import scipy.linalg as linalg
from collections import OrderedDict
from ccs.data.conversion import Bohr__AA, eV__Hartree
from scipy.linalg import block_diag


logger = logging.getLogger(__name__)


class Twobody:
    '''Twobody class that describes properties of an Atom pair.'''

    def __init__(self, name, dismat, Rcut, Swtype='rep',
                 Rmin=None, Resolution=0.1):
        '''Constructs a Twobody object.

        Args:

            name (str): name of the atom pair.
            dismat (dataframe): pairwise  distance matrix.
            nconfigs (int): number of configurations
            Rcut (float): maximum cut off value for spline interval
            Nknots (int): number of knots in the spline interval
            Rmin (float, optional): minimum value of the spline interval
                (default: None).


        '''

        self.name = name
        self.Rcut = Rcut
        self.res = Resolution
        self.Rmin = min
        self.N = int(np.ceil((Rcut - Rmin) / self.res))+1
        self.Rmin = self.Rcut - (self.N-1)*self.res
        self.rn = [i*self.res + self.Rmin for i in range(self.N)]
        self.Swtype = Swtype
        self.dismat = dismat
        self.Nconfs = np.shape(dismat)[0]
        self.C, self.D, self.B, self.A = self.spline_construction()
        self.vv, self.indices = self.get_v()
        self.merge_intervals()
        self.C, self.D, self.B, self.A = self.spline_construction()
        self.vv, self.indices = self.get_v()
        self.const = self.get_const()
        self.curvatures = None
        self.splcoeffs = None
        self.expcoeffs = None

    def merge_intervals(self):
        self.indices.sort()
        self.N = len(self.indices)
        self.rn = [self.rn[i] for i in self.indices]
        print("Merging intervall. N reduced to: ", self.N)

    def get_const(self):
        aa = np.zeros(0)
        g_mono = -1 * np.identity(self.N)
        g_mono[0, 0] = -1*(self.rn[1]-self.rn[0]+self.res)
        g_mono[0, 1] = 2*(self.res)
        for ii in range(1, self.N-1):
            g_mono[ii, ii] = -1*(self.rn[ii+1] - self.rn[ii-1])
            g_mono[ii, ii+1] = 2*(self.rn[ii] - self.rn[ii-1])
        gg = block_diag(g_mono)
        return gg

    def switch_const(self, n_switch):
        g = copy.deepcopy(self.const)
        ii, jj = np.indices(g.shape)
        g[ii > n_switch] = -g[ii > n_switch]
        return g

    def spline_construction(self):
        ''' This function constructs the matrices A, B, C, D.

        '''
        rows = self.N-1
        cols = self.N
        cc = np.zeros((rows, cols), dtype=float)
        np.fill_diagonal(cc, 1, wrap=True)
        cc = np.roll(cc, 1, axis=1)

        dd = np.zeros((rows, cols), dtype=float)
        ii, jj = np.indices(dd.shape)
        dd[ii == jj] = -1
        dd[ii == jj - 1] = 1
        dd = dd

        bb = np.zeros((rows, cols), dtype=float)
        ii, jj = np.indices(bb.shape)
        bb[ii == jj] = -0.5
        bb[ii < jj] = -1
        bb[jj == cols - 1] = -0.5
        bb = np.delete(bb, 0, 0)
        bb = np.vstack((bb, np.zeros(bb.shape[1])))
        bb = bb

        aa = np.zeros((rows, cols), dtype=float)
        tmp = 1 / 3.0
        for row in range(rows - 1, -1, -1):
            aa[row][cols - 1] = tmp
            tmp = tmp + 0.5
            for col in range(cols - 2, -1, -1):
                if row == col:
                    aa[row][col] = 1 / 6.0
                if col > row:
                    aa[row][col] = col - row

        aa = np.delete(aa, 0, 0)
        aa = np.vstack((aa, np.zeros(aa.shape[1])))
        aa = aa

        return cc, dd, bb, aa

    def get_v(self):
        '''
        Constructs the v matrix.

        Returns:

            ndarray: The v matrix for a pair.

        '''

        vv = np.zeros((self.Nconfs, self.N))

        indices = [0]
        for config in range(self.Nconfs):
            distances = [ii for ii in self.dismat[config, :]
                         if self.Rmin <= ii <= self.Rcut]
            uu = 0
            for rr in distances:
                index = bisect.bisect_left(self.rn, rr)
                #index = max(0, index)
                dr = max(self.res, (self.rn[index]-self.rn[index-1]))
                delta = (rr-self.rn[index]) / dr
                indices.append(index)
                aa_ind = self.A[index-1]
                bb_ind = self.B[index-1] * delta
                dd_ind = self.D[index-1] * np.power(delta, 3) / 6.0
                c_d = self.C[index-1] * np.power(delta, 2) / 2.0
                uu = uu + aa_ind + bb_ind + c_d + dd_ind

            vv[config, :] = uu

        logger.debug("\n V matrix :%s", vv)

        return vv, list(set(indices))


class Onebody:
    '''Onebody class that describes properties of an atom.'''

    def __init__(self, name, stomat, epsilon_supported=True, epsilon=0.0):
        '''Constructs a Onebody object.

        Args:

            name (str): name of the atom type.
            epsilon_supported  (bool): flag to tell if epsilon can be determined from the data
            epsilon (float): onebody energy term

        '''
        self.name = name
        self.epsilon_supported = True
        self.epsilon = 0.0
        self.stomat = stomat
