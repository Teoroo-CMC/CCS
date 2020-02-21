''' This module contains functions for spline construction, evaluation and output.'''
import logging

import numpy as np
import scipy.linalg as linalg

logger = logging.getLogger(__name__)


def spline_construction(rows, cols, dx):
    """ Form Spline coefficient matrices 

    Args:
        rows -- int
        The number of rows of the matrix
        cols -- int 
        The number of columns of the matrix 
        dx   -- int
        The size of the interval

    Returns:
        A   -- Matrix with a coefficients for spline
        B   -- Matrix with b coefficients for spline
        C   -- Matrix with c coefficients for spline
        D   -- Matrix with d coefficients for spline

    """

    C = np.zeros((rows, cols), dtype=float)
    np.fill_diagonal(C, 1, wrap=True)
    C = np.roll(C, 1, axis=1)

    D = np.zeros((rows, cols), dtype=float)
    i, j = np.indices(D.shape)
    D[i == j] = -1
    D[i == j - 1] = 1
    D = D / dx

    B = np.zeros((rows, cols), dtype=float)
    i, j = np.indices(B.shape)
    B[i == j] = -0.5
    B[i < j] = -1
    B[j == cols - 1] = -0.5
    B = np.delete(B, 0, 0)
    B = np.vstack((B, np.zeros(B.shape[1])))
    B = B * dx

    A = np.zeros((rows, cols), dtype=float)
    tmp = 1 / 3.0
    for row in range(rows - 1, -1, -1):
        A[row][cols - 1] = tmp
        tmp = tmp + 0.5
        for col in range(cols - 2, -1, -1):
            if row == col:
                A[row][col] = 1 / 6.0
            if col > row:
                A[row][col] = col - row

    A = np.delete(A, 0, 0)
    A = np.vstack((A, np.zeros(A.shape[1])))
    A = A * dx * dx

    return C, D, B, A


def spline_eval012(a, b, c, d, r, Rcut, Rmin, dx, x):
    ''' This function returns cubic spline value given certain distances '''

    if r == Rmin:
        index = 1
    else:
        index = int(np.ceil((r - Rmin) / dx))

    if index >= 1:
        dr = r - x[index]
        f0 = a[index-1] + dr * \
            (b[index-1] + dr*(0.5*c[index-1] + (d[index-1]*dr/3.0)))
        f1 = b[index - 1] + dr * (c[index - 1] + (0.5 * d[index - 1] * dr))
        f2 = c[index - 1] + d[index - 1] * dr
        print('value of f0' + str(f0))
        return f0, f1, f2
    else:
        raise ValueError(' r < Rmin')


def spline_energy_model(Rcut, Rmin, df, cols, dx, size, x):
    C, D, B, A = spline_construction(cols - 1, cols, dx)
    logger.debug(" Number of configuration for v matrix: %s", size)
    logger.debug("\n A matrix is: \n %s \n Spline interval = %s", A, x)
    v = np.zeros((size, cols))
    indices = []
    for config in range(size):
        distances = [i for i in df[config, :] if i <= Rcut and i >= Rmin]
        u = 0
        for r in distances:
            index = int(np.ceil(np.around(((r - Rmin) / dx), decimals=5)))
            indices.append(index)
            delta = r - x[index]
            logger.debug("\n In config %s\t distance r = %s\tindex=%s\tbin=%s",
                         config, r, index, x[index])
            a = A[index - 1]
            b = B[index - 1] * delta
            d = D[index - 1] * np.power(delta, 3) / 6.0
            c_d = C[index - 1] * np.power(delta, 2) / 2.0
            u = u + a + b + c_d + d

        v[config, :] = u
    logger.debug("\n V matrix :%s", v)
    return v


def write_splinecoeffs(twb, coeffs, fname='splines.out', exp_head=False):
    coeffs_format = ' '.join(['{:6.3f}'] * 2 + ['{:15.8E}'] * 4) + '\n'
    with open(fname, 'w') as fout:
        fout.write('Spline table\n')
        for index in range(len(twb.interval)-1):
            r_start = twb.interval[index]
            r_stop = twb.interval[index+1]
            fout.write(coeffs_format.format(r_start, r_stop, *coeffs[index]))


def write_error(mdl_eng, ref_eng, mse, fname='error.out'):
    header = "{:<15}{:<15}{:<15}".format("Reference", "Predicted", "Error")
    error = abs(ref_eng - mdl_eng)
    maxerror = max(abs(error))
    footer = "MSE = {:2.5E}\nMaxerror = {:2.5E}".format(mse, maxerror)
    np.savetxt(fname,
               np.transpose([ref_eng, mdl_eng, error]),
               header=header,
               footer=footer,
               fmt="%-15.5f")


class Twobody():
    ''' Class representing two body 
     Attributes:
    Name -- str 
        The name of the atomic pair (eg: Zn,O,ZnO).
    Rcut -- float   
        The cutoff distance for splines.
    Rmin --  float
        The distance were splines begin.
    dx   -- float
        The gridsize for the splines.
    dismat -- numpy matrix
        The pairwise distances.
    V    -- Numpy Matrix
        The spline energy model matrix( refer to paper)
    c    -- numpy array
        The curvatures for the splines


    '''

    def __init__(self,
                 name,
                 Dismat,
                 Nconfigs,
                 Rcut=None,
                 Rmin=None,
                 Nknots=None,
                 Nswitch=None):
        self.name = name
        self.Rcut = Rcut
        self.Rmin = Rmin
        self.Nknots = Nknots
        self.Nswitch = Nswitch
        self.Dismat = Dismat
        self.Nconfigs = Nconfigs
        self.dx = (self.Rcut - self.Rmin) / self.Nknots
        self.cols = self.Nknots + 1
        self.interval = np.linspace(self.Rmin,
                                    self.Rcut,
                                    self.cols,
                                    dtype=float)
        self.C, self.D, self.B, self.A = spline_construction(
            self.cols - 1, self.cols, self.dx)
        self.v = self.get_v()

    def get_v(self):
        return spline_energy_model(self.Rcut, self.Rmin, self.Dismat,
                                   self.cols, self.dx, self.Nconfigs,
                                   self.interval)
