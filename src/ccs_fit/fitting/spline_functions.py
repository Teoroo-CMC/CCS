# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2023  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
# ------------------------------------------------------------------------------#


"""
This module contains functions for spline construction, evaluation and output.
"""


import logging
import numpy as np
import bisect
import copy
import scipy.linalg as linalg
from ccs_fit.data.conversion import Bohr__AA, eV__Hartree
from scipy.linalg import block_diag

logger = logging.getLogger(__name__)


class Twobody:
    """Twobody class that describes properties of an Atom pair."""

    def __init__(
        self,
        name,
        dismat,
        distmat_forces,
        Rcut,
        range_center=None,
        range_width=None,
        search_points=None,
        Swtype="rep",
        Rmin=None,
        Resolution=0.1,
        const_type="mono",
        search_mode="full",
    ):
        """
        Constructs a Twobody object.

        Input
        -----
            name : str
                name of the atom pair.
            dismat : dataframe
                pairwise  distance matrix.
            nconfigs : int
                number of configurations
            Rcut : float
                maximum cut off value for spline interval
            Nknots : int
                number of knots in the spline interval
            Rmin : float
                optional, minimum value of the spline interval, default None
            
        """

        self.name = name
        self.Rmin = Rmin
        self.res = Resolution
        self.N = int(np.ceil((Rcut - Rmin) / self.res)) + 1
        # TO MAXIMIZE NUMERICAL STABILITY INNERMOST POINT IS PLACED IN THE MIDLE OF THE FIRST INTERVAL
        # IN MAIN Rmin IS ADJUSTED THIS WAY, WE THEREFORE BUILD "UPWARDS" FROM Rmin.
        self.N_full = self.N
        self.Rcut = self.Rmin + (self.N - 1) * self.res
        self.rn_full = [(i) * self.res + self.Rmin for i in range(self.N)]
        self.rn = self.rn_full
        if not range_center:
            self.range_center = (Rmin+self.Rcut)/2
        else:
            self.range_center = range_center
        if not range_width:
            self.range_width = (self.Rcut-Rmin)
        else:
            self.range_width = range_width
        if not search_points:
            self.search_points = np.arange(Rmin, self.Rcut, self.res)
        else:
            self.search_points = search_points
        self.Swtype = Swtype
        self.const_type = const_type
        self.search_mode = search_mode
        self.range_center = range_center
        self.range_width = range_width
        self.search_points = search_points
        self.dismat = dismat
        self.Nconfs = np.shape(dismat)[0]
        self.distmat_forces = distmat_forces
        self.Nconfs_forces = np.shape(distmat_forces)[0]
        self.C, self.D, self.B, self.A = self.spline_construction()
        self.vv, self.indices = self.get_v()
        self.const = self.get_const()
        self.fvv_x, self.fvv_y, self.fvv_z, self.indices = self.get_v_forces(
            self.indices)
        self.curvatures = None
        self.splcoeffs = None
        self.expcoeffs = None
        if self.const_type.lower() == 'mono':
            print("    Applying monotonous constraints for pair: ", self.name)
            logger.info(
                f"Applying monotonous constraints for pair: {self.name}")
        if self.const_type.lower() == 'simple':
            print("    Applying simple constraints for pair: ", self.name)
            logger.info(
                f"Applying simple constraints for pair: {self.name}")

    def merge_intervals(self):
        self.indices.sort()
        self.N = len(self.indices)
        self.rn = [self.rn[i] for i in self.indices]
        self.C, self.D, self.B, self.A = self.spline_construction()
        self.vv, _ = self.get_v()
        self.const = self.get_const()
        self.fvv_x, self.fvv_y, self.fvv_z,_ = self.get_v_forces([])
        if self.N_full > self.N:
            print(
                f"    Merging intervals for pair {self.name}; number of intervals reduced from {self.N_full} to {self.N}. ")
        logger.info(
            f"Merging intervals for pair {self.name}; number of intervals reduced from {self.N_full} to {self.N}. ")

    def dissolve_interval(self):
        tmp_curvatures = []
        indices = self.indices
        for i in range(self.N_full - 1 - indices[-1]):
            tmp_curvatures.append(0.0)

        for i in range(self.N - 1, 0, -1):
            l_i = indices[i] - indices[i - 1]
            c_i = self.curvatures[i][0]
            c_i_minus_1 = self.curvatures[i - 1][0]
            d_i = (c_i - c_i_minus_1) / (l_i)
            for j in range(l_i):
                c_tmp = c_i - j * d_i
                tmp_curvatures.append(c_tmp)
        tmp_curvatures.append(*self.curvatures[0])
        tmp_curvatures.reverse()

        self.curvatures = tmp_curvatures
        self.N = self.N_full
        self.rn = self.rn_full
        self.C, self.D, self.B, self.A = self.spline_construction()

    def get_const(self):
        a = -1 * np.identity(self.N)
        g_mono = -1 * np.identity(self.N)
        g_mono[0, 0] = -1
        g_mono[0, 1] = 1
        for ii in range(1, self.N - 1):
            g_mono[ii, ii] = -1
            g_mono[ii, ii+1] = 1
        gg = block_diag(g_mono)
        if self.const_type.lower() == 'mono':
            return gg
        if self.const_type.lower() == 'simple':
            return a
        if self.const_type.lower() == 'none':
            return np.zeros((1,self.N))

    def switch_const(self, n_switch):
        g = copy.deepcopy(self.const)
        ii, jj = np.indices(g.shape)
        g[ii > n_switch] = -g[ii > n_switch]
        return g

    def spline_construction(self):
        """This function constructs the matrices A, B, C, D."""
        dx = np.array([self.rn[i]-self.rn[i-1] for i in range(1, self.N)])
        # dx = np.insert(dx, 0, self.res)

        rows = self.N - 1
        cols = self.N
        cc = np.zeros((rows, cols), dtype=float)
        np.fill_diagonal(cc, 1, wrap=True)
        cc = np.roll(cc, 1, axis=1)

        bb = np.zeros((rows, cols), dtype=float)
        aa = np.zeros((rows, cols), dtype=float)
        dd = np.zeros((rows, cols), dtype=float)
        # dd[rows-1, -1] = -1 / dx[rows-1]
        # dd[rows-1, -2] = +1 / dx[rows-1]

        for i in range(1, rows):
            ii = rows-i-1
            dd[ii+1] = (cc[ii+1] - cc[ii])/dx[ii+1]
            bb[ii] = bb[ii+1] - dx[ii+1]*cc[ii+1] + 0.5*(dx[ii+1]**2)*dd[ii+1]
            aa[ii] = aa[ii+1] - dx[ii+1]*bb[ii+1] + 0.5 * \
                (dx[ii+1]**2)*cc[ii+1] - (1/6.)*(dx[ii+1]**3)*dd[ii+1]

        return cc, dd, bb, aa

    def get_v(self):
        """
        Constructs the v matrix.

        Returns
        -------
            ndarray : matrix
                The v matrix for a pair.

        """

        vv = np.zeros((self.Nconfs, self.N))

        indices = [0]
        for config in range(self.Nconfs):
            distances = [
                ii for ii in self.dismat[config, :] if self.Rmin <= ii <= self.Rcut
            ]
            uu = 0
            for rr in distances:
                index = bisect.bisect_left(self.rn, rr)
                delta = (rr - self.rn[index])  # / self.res
                indices.append(index)
                # INDEX IS SHIFTED IN A,B,C,D
                # THIS IS BECAUSE OF THE A,B,C, and D matrix being (N-1)xN
                index = index - 1
                aa_ind = self.A[index]
                bb_ind = self.B[index] * delta
                dd_ind = self.D[index] * np.power(delta, 3) / 6.0
                c_d = self.C[index] * np.power(delta, 2) / 2.0
                uu = uu + aa_ind + bb_ind + c_d + dd_ind

            vv[config, :] = uu

        return vv, list(set(indices))

    def get_v_forces(self, indices):
        """
        Constructs the v matrix.

        Returns:

            ndarray: The v matrix for a pair.

        """
        vv_x = np.zeros((self.Nconfs_forces, self.N))
        vv_y = np.zeros((self.Nconfs_forces, self.N))
        vv_z = np.zeros((self.Nconfs_forces, self.N))

        for config in range(self.Nconfs_forces):
            uu_x = 0
            uu_y = 0
            uu_z = 0
            for rv in self.distmat_forces[config, :]:
                rr = np.linalg.norm(rv)
                if rr > 0 and rr < self.Rcut:
                    index = bisect.bisect_left(self.rn, rr)
                    delta = (rr - self.rn[index])
                    indices.append(index)
                    # INDEX IS SHIFTED IN A,B,C,D
                    # THIS IS BECAUSE OF THE A,B,C, and D matrix being (N-1)xN
                    # NOTE THAT FORCE IS NOT NEGATIVE OF DERIVATIVE SINCE
                    # WE ARE "MOVING" FROM END OF INTERVAL INWARDS!!!
                    index = index - 1
                    bb_ind = self.B[index]
                    dd_ind = self.D[index] * np.power(delta, 2) / 2.0
                    c_d = self.C[index] * delta
                    c_force = bb_ind + c_d + dd_ind
                    uu_x = uu_x + c_force * rv[0] / rr
                    uu_y = uu_y + c_force * rv[1] / rr
                    uu_z = uu_z + c_force * rv[2] / rr

            vv_x[config, :] = uu_x
            vv_y[config, :] = uu_y
            vv_z[config, :] = uu_z
        return vv_x, vv_y, vv_z, list(set(indices))

    def get_spline_coeffs(self):
        a_values = np.dot(self.A, self.curvatures)
        b_values = np.dot(self.B, self.curvatures)
        c_values = np.dot(self.C, self.curvatures)
        d_values = np.dot(self.D, self.curvatures)
        r_values = self.rn

        spl = []
        for i in range(self.N-1):
            a_r = a_values[i]
            b_r = b_values[i]
            c_r = c_values[i]
            d_r = d_values[i]
            dr = (r_values[i+1]-r_values[i])
            a_l = a_r - b_r*dr + (c_r/2.)*dr**2 - (d_r/6.)*dr**3
            b_l = b_r - c_r*dr + (3*d_r/6.)*dr**2
            c_l = c_r - d_r*dr
            d_l = d_r
            c_l = (1/2.)*c_l
            d_l = (1/6.)*d_l
            spl.append([float(a_l), float(b_l), float(c_l), float(d_l)])
        spl = np.array(spl)

        self.splcoeffs = spl

    def get_expcoeffs(self):
        """Calculates coefficients of exponential function.

        Args:

            aa (float):
            bb (float):
            cc (float):
            r0 (float):

        Returns:

            alpha (float):
            beta (float):
            gamma (float):

        """
        aa = self.splcoeffs[0, 0]
        bb = self.splcoeffs[0, 1]
        cc = self.splcoeffs[0, 2]
        r0 = self.rn[0]

        alpha = -cc / bb
        beta = alpha * r0 + np.log(cc / alpha**2)
        gamma = aa - np.exp(-alpha * r0 + beta)

        self.expcoeffs = [alpha, beta, gamma]


class Onebody:
    """Onebody class that describes properties of an atom."""

    def __init__(self, name, stomat, epsilon_supported=True, epsilon=0.0):
        """Constructs a Onebody object.

        Args:

            name (str): name of the atom type.
            epsilon_supported  (bool): flag to tell if epsilon can be determined from the data
            epsilon (float): onebody energy term

        """
        self.name = name
        self.epsilon_supported = True
        self.epsilon = 0.0
        self.stomat = stomat
