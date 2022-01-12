import logging
import itertools
import pickle
import json
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from numpy.core.numeric import indices
from scipy.linalg import block_diag


class CCS_regressor:
    def __init__(self, N=100, xmin=0, xmax=1, sw=False, eps=False):
        self.N = N
        self.sw = sw
        self.xmin = xmin
        self.xmax = xmax
        self.dx = (self.xmax-self.xmin)/self.N
        self.eps = eps
        self.C, self.D, self.B, self.A = self.spline_construction(
            self.N, self.dx)
        self.interval = np.linspace(self.xmin, self.xmax, self.N, dtype=float)

    @staticmethod
    def solver(pp, qq, gg, hh, aa, bb, maxiter=300, tol=(1e-10, 1e-10, 1e-10)):
        '''The solver for the objective.

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

        '''

        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = maxiter
        solvers.options['feastol'] = tol[0]
        solvers.options['abstol'] = tol[1]
        solvers.options['reltol'] = tol[2]

        if aa:
            sol = solvers.qp(pp, qq, gg, hh, aa, bb)
        else:
            sol = solvers.qp(pp, qq, gg, hh)

        return sol

    def fit(self, x, y):
        self.mm, self.incices = self.model(x)
        n_switch = self.N
        pp = matrix(np.transpose(self.mm).dot(self.mm))
        qq = -1 * matrix(np.transpose(self.mm).dot(y))
        gg, aa = self.const(n_switch)
        hh = np.zeros(gg.shape[0])
        bb = np.zeros(aa.shape[0])
        self.sol = self.solver(pp, qq, matrix(gg), matrix(hh), matrix(aa),
                               matrix(bb))

    def const(self, n_switch):
        aa = np.zeros(0)
        g_mono = -1 * np.identity(self.N)
        ii, jj = np.indices(g_mono.shape)
        g_mono[ii == jj - 1] = 1
        g_mono[ii > n_switch] = -g_mono[ii > n_switch]
        gg = block_diag(g_mono, 0)
        return gg, aa

    def predict(self, x):
        mm, indices = self.model(x)
        y = mm.dot(np.array(self.sol['x']))
        return y

    def spline_construction(self, N, dx):
        ''' This function constructs the matrices A, B, C, D.

        Args:
            N : Number of knots
            dx (list): grid spaceing

        Returns:

            cc, dd, bb, aa (matrices): constructed matrices

        '''
        rows = N-1
        cols = N

        cc = np.zeros((rows, cols), dtype=float)
        np.fill_diagonal(cc, 1, wrap=True)
        cc = np.roll(cc, 1, axis=1)
        dd = np.zeros((rows, cols), dtype=float)
        ii, jj = np.indices(dd.shape)
        dd[ii == jj] = -1
        dd[ii == jj - 1] = 1
        dd = dd / dx

        bb = np.zeros((rows, cols), dtype=float)
        ii, jj = np.indices(bb.shape)
        bb[ii == jj] = -0.5
        bb[ii < jj] = -1
        bb[jj == cols - 1] = -0.5
        bb = np.delete(bb, 0, 0)
        bb = np.vstack((bb, np.zeros(bb.shape[1])))
        bb = bb * dx

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
        aa = aa * dx * dx

        return cc, dd, bb, aa

    def model(self, x):
        '''Constructs the v matrix.

        Args:
            self

        Returns:

            ndarray: The v matrix for a pair.

        '''
        aa = self.A
        bb = self.B
        cc = self.C
        dd = self.D
        size = len(x)
        dx = self.dx
        xmin = self.xmin

        vv = np.zeros((size, self.N))
        uu = np.zeros((self.N, 1)).flatten()
        indices = []
        for i in range(size):
            index = int(np.ceil(np.around(((x[i] - xmin) / dx), decimals=5)))
            if(index < self.N) & (index > 0):
                indices.append(index)
                delta = x[i] - self.interval[index]
                aa_ind = aa[index - 1]
                bb_ind = bb[index - 1] * delta
                dd_ind = dd[index - 1] * np.power(delta, 3) / 6.0
                c_d = cc[index - 1] * np.power(delta, 2) / 2.0
                uu = aa_ind + bb_ind + c_d + dd_ind

            vv[i, :] = uu
        vv = np.hstack((vv, np.ones((size, 1))))

        return vv, set(indices)
