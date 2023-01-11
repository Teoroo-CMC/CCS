# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                           #
#  Copyright (C) 2019 - 2023  CCS developers group                              #
#                                                                               #
#  See the LICENSE file for terms of usage and distribution.                    #
#                                                                               #
#  The code below is not well-tested, so use at your own disclosure.            #
#                                                                               #
# ------------------------------------------------------------------------------#

import bisect
import numpy as np
from cvxopt import matrix, solvers
from scipy.linalg import block_diag


class CCS_regressor:
    def __init__(self, N=100, xmin=0, xmax=1, dx=None, sw=False, eps=False):
        self.N = N
        self.sw = sw
        self.xmin = xmin
        self.xmax = xmax
        if not dx:
            self.dx = np.ones((N, 1))
            self.dx = (self.xmax - self.xmin) * self.dx / np.sum(self.dx)
        self.eps = eps
        self.C, self.D, self.B, self.A = self.spline_construction(self.N)
        print("WARNING: THE CCS_REGRESSOR HAS NOT BEEN TESTED, USE AT YOUR OWN DISCLOSURE")

    def merge_intervals(self, x):
        dx = self.dx
        xmin = self.xmin
        xns = np.array([float(sum(dx[0:i])) + xmin for i in range(len(dx))])
        indices = [self.N]
        for i in range(len(x)):
            index = bisect.bisect_left(xns, x[i])
            indices.append(index)

        indices = list(set(indices))
        indices.sort()

        N_new = len(indices)
        dx_new = np.zeros((N_new, 1))

        i_last = 0
        cnt = 0
        for i in indices:
            dx_new[cnt] = np.sum(dx[i_last:i])
            cnt += 1
            i_last = i

        self.N = N_new
        self.dx = dx_new
        self.C, self.D, self.B, self.A = self.spline_construction(self.N)
        print("Merging interval. N reduced to: ", self.N)

    def rubber_band(self):
        pass

    @staticmethod
    def solver(pp, qq, gg, hh, aa, bb, maxiter=300, tol=(1e-10, 1e-10, 1e-10)):
        """The solver for the objective.

        Args:

            pp (matrix): P matrix as per standard Quadratic Programming(QP)
                notation.
            qq (matrix): q matrix as per standard QP notation.
            gg (matrix): G matrix as per standard QP notation.
            hh (matrix): h matrix as per standard QP notation
            aa (matrix): A matrix as per standard QP notation.
            bb (matrix): b matrix as per standard QP notation
            maxiter (int, optional): maximum iteration steps (default: 300).
            tol (tuple, optional): tolerance value of the solution
                (default: (1e-10, 1e-10, 1e-10)).

        Returns:

            sol (dict): dictionary containing solution details

        """

        solvers.options["show_progress"] = False
        solvers.options["maxiters"] = maxiter
        solvers.options["feastol"] = tol[0]
        solvers.options["abstol"] = tol[1]
        solvers.options["reltol"] = tol[2]

        if aa:
            sol = solvers.qp(pp, qq, gg, hh, aa, bb)
        else:
            sol = solvers.qp(pp, qq, gg, hh)

        return sol

    def fit(self, x, y):
        self.merge_intervals(x)
        self.mm, self.incices = self.model(x)
        n_switch = self.N
        pp = matrix(np.transpose(self.mm).dot(self.mm))
        qq = -1 * matrix(np.transpose(self.mm).dot(y))
        gg, aa = self.const(n_switch)
        hh = np.zeros(gg.shape[0])
        bb = np.zeros(aa.shape[0])
        self.sol = self.solver(pp, qq, matrix(gg), matrix(hh), matrix(aa), matrix(bb))

    def const(self, n_switch):
        aa = np.zeros(0)
        g_mono = -1 * np.identity(self.N)
        for ii in range(self.N - 1):
            # g_mono[ii, ii] = - (1/self.dx[ii+1]+1/self.dx[ii])
            # g_mono[ii, ii+1] = 2/self.dx[ii]
            g_mono[ii, ii] = -(self.dx[ii + 1] + self.dx[ii])
            g_mono[ii, ii + 1] = 2 * self.dx[ii]
        # g_mono[ii > n_switch] = -g_mono[ii > n_switch]
        gg = block_diag(g_mono, 0)
        return gg, aa

    def predict(self, x):
        mm, indices = self.model(x)
        y = mm.dot(np.array(self.sol["x"]))
        return y

    def spline_construction(self, N):
        """This function constructs the matrices A, B, C, D.

        Args:
            N : Number of knots

        Returns:

            cc, dd, bb, aa (matrices): constructed matrices

        """
        rows = N - 1
        cols = N

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

    def model(self, x):
        """Constructs the v matrix.

        Args:
            self

        Returns:

            ndarray: The v matrix for a pair.

        """
        aa = self.A
        bb = self.B
        cc = self.C
        dd = self.D
        size = len(x)
        dx = self.dx
        xmin = self.xmin
        xns = np.array([float(sum(dx[0:i])) + xmin for i in range(len(dx))])

        vv = np.zeros((size, self.N))
        uu = np.zeros((self.N, 1)).flatten()
        indices = []
        for i in range(size):

            index = bisect.bisect_left(xns, x[i])
            index = max(0, index)

            if (index < self.N) & (index > 0):
                delta = (x[i] - xns[index]) / dx[index - 1]
                indices.append(index)
                aa_ind = aa[index - 1]
                bb_ind = bb[index - 1] * delta
                dd_ind = dd[index - 1] * np.power(delta, 3) / 6.0
                c_d = cc[index - 1] * np.power(delta, 2) / 2.0
                uu = aa_ind + bb_ind + c_d + dd_ind

            vv[i, :] = uu
        vv = np.hstack((vv, np.ones((size, 1))))

        return vv, set(indices)
