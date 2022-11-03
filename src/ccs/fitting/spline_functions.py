# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2021  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
# ------------------------------------------------------------------------------#


"""
This module contains functions for spline construction, evaluation and output.
"""


from ctypes import c_int
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
    """Twobody class that describes properties of an Atom pair."""

    def __init__(
        self,
        name,
        dismat,
        distmat_forces,
        Rcut,
        Swtype="rep",
        Rmin=None,
        Resolution=0.1,
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
        self.Rcut = Rcut
        self.res = Resolution
        self.N = int(np.ceil((Rcut - Rmin) / self.res)) + 1
        self.N_full = self.N
        self.Rmin = self.Rcut - (self.N - 1) * self.res
        self.rn_full = [i * self.res + self.Rmin for i in range(self.N)]
        self.rn = self.rn_full
        self.Swtype = Swtype
        self.dismat = dismat
        self.Nconfs = np.shape(dismat)[0]
        self.distmat_forces = distmat_forces
        self.Nconfs_forces = np.shape(distmat_forces)[0]
        self.C, self.D, self.B, self.A = self.spline_construction()
        self.vv, self.indices = self.get_v()
        self.const = self.get_const()
        self.fvv_x, self.fvv_y, self.fvv_z = self.get_v_forces()
        self.curvatures = None
        self.splcoeffs = None
        self.expcoeffs = None

    def merge_intervals(self):
        self.indices.sort()
        self.N = len(self.indices)
        self.rn = [self.rn[i] for i in self.indices]
        self.C, self.D, self.B, self.A = self.spline_construction()
        self.vv, _ = self.get_v()
        self.const = self.get_const()
        self.fvv_x, self.fvv_y, self.fvv_z = self.get_v_forces()
        print("Merging intervall. N reduced to: ", self.N)

    def dissolve_interval(self):
        tmp_curvatures = []
        indices = self.indices
        for i in range(self.N_full - 1 - indices[-1]):
            tmp_curvatures.append(0.0)

        for i in range(self.N - 1, 0, -1):
            l_i = indices[i] - indices[i - 1]
            if i > 1:
                l_i_minus_1 = indices[i - 1] - indices[i - 2]
            else:
                l_i_minus_1 = 1
            c_i = self.curvatures[i][0] / l_i
            c_i_minus_1 = self.curvatures[i - 1][0] / l_i_minus_1
            d_i = (c_i - c_i_minus_1) / l_i
            for j in range(l_i):
                c_tmp = c_i - j * d_i
                tmp_curvatures.append(c_tmp)
        tmp_curvatures.append(*self.curvatures[0])
        tmp_curvatures.reverse()

        for i in range(len(indices)):
            print(indices[i], *self.curvatures[i])
        for i in range(len(tmp_curvatures)):
            print(i, tmp_curvatures[i])

        self.curvatures = tmp_curvatures
        self.N = self.N_full
        self.rn = self.rn_full
        self.C, self.D, self.B, self.A = self.spline_construction()

    def get_const(self):
        aa = np.zeros(0)
        g_mono = -1 * np.identity(self.N)
        g_mono[0, 0] = -1 * (self.rn[1] - self.rn[0] + self.res)
        g_mono[0, 1] = 2 * (self.res)
        for ii in range(1, self.N - 1):
            g_mono[ii, ii] = -1 * (self.rn[ii + 1] - self.rn[ii - 1])
            g_mono[ii, ii + 1] = 2 * (self.rn[ii] - self.rn[ii - 1])
        gg = block_diag(g_mono)
        return gg

    def switch_const(self, n_switch):
        g = copy.deepcopy(self.const)
        ii, jj = np.indices(g.shape)
        g[ii > n_switch] = -g[ii > n_switch]
        return g

    def spline_construction(self):
        """This function constructs the matrices A, B, C, D."""
        rows = self.N - 1
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
                # index = max(0, index)
                dr = max(self.res, (self.rn[index] - self.rn[index - 1]))
                if dr > (self.res + 0.02):
                    print("---", dr, self.rn[index], index)
                delta = (rr - self.rn[index]) / dr
                indices.append(index)
                index = index - 1
                aa_ind = self.A[index]
                bb_ind = self.B[index] * delta
                dd_ind = self.D[index] * np.power(delta, 3) / 6.0
                c_d = self.C[index] * np.power(delta, 2) / 2.0
                uu = uu + aa_ind + bb_ind + c_d + dd_ind

            vv[config, :] = uu

        return vv, list(set(indices))

    def get_v_forces(self):
        """
        Constructs the v matrix.

        Returns:

            ndarray: The v matrix for a pair.

        """
        vv_x = np.zeros((self.Nconfs_forces, self.N))
        vv_y = np.zeros((self.Nconfs_forces, self.N))
        vv_z = np.zeros((self.Nconfs_forces, self.N))

        indices = [0]
        for config in range(self.Nconfs_forces):
            uu_x = 0
            uu_y = 0
            uu_z = 0
            for rv in self.distmat_forces[config, :]:
                rr = np.linalg.norm(rv)
                if rr > 0 and rr < self.Rcut:
                    index = bisect.bisect_left(self.rn, rr)
                    dr = max(self.res, (self.rn[index] - self.rn[index - 1]))
                    delta = (rr - self.rn[index]) / dr
                    indices.append(index)
                    bb_ind = -self.B[index - 1]
                    dd_ind = -self.D[index - 1] * np.power(delta, 2) / 3.0
                    c_d = -self.C[index - 1] * delta
                    c_force = bb_ind + c_d + dd_ind
                    uu_x = uu_x + c_force * rv[0] / rr
                    uu_y = uu_y + c_force * rv[1] / rr
                    uu_z = uu_z + c_force * rv[2] / rr

            vv_x[config, :] = uu_x
            vv_y[config, :] = uu_y
            vv_z[config, :] = uu_z

        return vv_x, vv_y, vv_z

    def get_spline_coeffs(self):
        """
        Spline coefficients for a spline with given 1st derivatives at its ends.
        The process turns the (internal) right-aligned spline table to a more common
        left-aligned form.

            Returns:

                np.transpose(mtx): spline coefficients in matrix-form

        """
        assert self.N == self.N_full, "Intervals still merged, dissolve them!"
        a_values = np.dot(self.A, self.curvatures)
        b_values = np.dot(self.B, self.curvatures)
        c_values = np.dot(self.C, self.curvatures)
        d_values = np.dot(self.D, self.curvatures)

        # Add extra point at the left of the interval.
        # Note, we are still working with a right-aligned spline...
        dr = -1.0
        x_values = np.array(self.rn)
        # x_values = np.append(x_values, 0.0)

        y_values = a_values
        y_0 = a_values[0] + dr * (
            b_values[0] + dr * (0.5 * c_values[0] + dr * d_values[0] / 6.0)
        )
        y_values = np.insert(y_values, 0, y_0)
        # y_values = np.append(y_values, 0.0)

        left_deriv = (
            b_values[0] + dr * (c_values[0] + dr * d_values[0] / 2.0)
        ) / self.res  #
        right_deriv = 0.0

        # The algebraic excersise starts here...
        nn = len(x_values)
        kk = x_values[1:] - x_values[:-1]
        mu = kk[:-1] / (kk[:-1] + kk[1:])
        mun = 1.0
        alpha = 1.0 - mu
        alpha0 = 1.0
        mu = np.hstack((mu, [mun]))
        alpha = np.hstack(([alpha0], alpha))
        dd = (
            6.0
            / (kk[:-1] + kk[1:])
            * (
                (y_values[2:] - y_values[1:-1]) / kk[1:]
                - (y_values[1:-1] - y_values[0:-2]) / kk[:-1]
            )
        )
        d0 = 6.0 / kk[0] * ((y_values[1] - y_values[0]) / kk[0] - left_deriv)
        dn = 6.0 / kk[-1] * (right_deriv - (y_values[-1] - y_values[-2]) / kk[-1])
        dd = np.hstack((d0, dd, dn))
        mtx = 2.0 * np.identity(nn, dtype=float)
        for ii in range(nn - 1):
            mtx[ii, ii + 1] = alpha[ii]
            mtx[ii + 1, ii] = mu[ii]
        mm = linalg.solve(mtx, dd)
        c0 = y_values[:-1]
        c1 = (y_values[1:] - y_values[:-1]) / kk - (2.0 * mm[:-1] + mm[1:]) / 6.0 * kk
        c2 = mm[:-1] / 2.0
        c3 = (mm[1:] - mm[:-1]) / (6.0 * kk)
        mtx = np.array([c0, c1, c2, c3])

        self.splcoeffs = np.transpose(mtx)

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
        # NOTE WE SHOULD NOTE INCLUDE INNERMOST POINT WHICH IS IL-DEFINED!
        # OLD CODE
        # aa = self.splcoeffs[0, 0]
        # bb = self.splcoeffs[0, 1]
        # cc = self.splcoeffs[0, 2]
        # r0 = self.rn[0]
        # NEW CODE
        aa = self.splcoeffs[1, 0]
        bb = self.splcoeffs[1, 1]
        cc = self.splcoeffs[1, 2]
        r0 = self.rn[1]

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
