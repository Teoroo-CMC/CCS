"""
This module contains functions for spline construction, evaluation and output.
"""
import logging

import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def spline_construction(rows, cols, dx):
    """ This function constructs the matrices A, B, C, D.
    Args:
        rows (int): The row dimension for matrix
        cols (int): The column dimension of the matrix
        dx (list): grid space((Rcut-Rmin)/N)
    
    Returns:
        A,B,C,D matrices
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
    """Returns cubic spline value, first and second derivative at a point r. 
    Args:
        a (ndarray): a coefficients of the spline.
        b (ndarray): b coefficients of the spline.
        c (ndarray): c coefficients of the spline.
        d (ndarray): d coefficients of the spline.
        r (float):   Point to evaluate the spline function.
        Rcut (float): The max value cut off for spline interval.
        Rmin (float): The min value cut off for spline interval.
        dx (float):  Grid space.
        x (interval): Spline interval.
    
    Raises:
        ValueError: If the point to be evaluated is below cut-off
    
    Returns:
        float: The values for spline function value, first and second derivative
    """

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
#        print( type(f0))
        return float(f0), float(f1), float(f2)
    else:
        raise ValueError(' r < Rmin')
    
def get_spline_coeffs(xx, yy, deriv0, deriv1):
    'Spline coefficients for a spline with given 1st derivatives at its ends.'

    nn = len(xx)
    kk = xx[1:] - xx[:-1]
    mu = kk[:-1] / (kk[:-1] + kk[1:])
    mun = 1.0
    alpha = 1.0 - mu
    alpha0 = 1.0
    mu = np.hstack((mu, [ mun ]))
    alpha = np.hstack(([ alpha0 ], alpha))
    dd = 6.0 / (kk[:-1] + kk[1:]) * ((yy[2:] - yy[1:-1]) / kk[1:] 
                                     - (yy[1:-1] - yy[0:-2]) / kk[:-1])
    d0 = 6.0 / kk[0] * ((yy[1] - yy[0]) / kk[0] - deriv0)
    dn = 6.0 / kk[-1] * (deriv1 - (yy[-1] - yy[-2]) / kk[-1])
    dd = np.hstack((d0, dd, dn))
    mtx = 2.0 * np.identity(nn, dtype=float)
    for ii in range(nn - 1):
        mtx[ii,ii+1] = alpha[ii]
        mtx[ii+1,ii] = mu[ii]
    mm = linalg.solve(mtx, dd)
    c0 = yy[:-1]
    c1 = (yy[1:] - yy[:-1]) / kk - (2.0 * mm[:-1] + mm[1:]) / 6.0 * kk
    c2 = mm[:-1] / 2.0
    c3 = (mm[1:] - mm[:-1]) / (6.0 * kk)
    mtx = np.array([ c0, c1, c2, c3 ])
    return np.transpose(mtx)

def spline_energy_model(Rcut, Rmin, df, cols, dx, size, x):
    """ Constructs the v matrix 
    
    Args:
        Rcut (float): The max value cut off for spline interval.
        Rmin (float): The min value cut off for spline interval.
        df (ndarray): The paiwise distance matrix.
        cols (int):  Number of unknown parameters.
        dx (float): Grid size.
        size (int): Number of configuration.
        x (list): Spline interval.
    
    Returns:
        ndarray: The v matrix for a pair.
    """
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
    """This function writes the spline output
    
    Args:
        twb (Twobody): Twobody class object.
        coeffs (ndarray): Array containing spline coefficients.
        fname (str, optional): Filename to output the spline coefficients. Defaults to 'splines.out'.
        exp_head (bool, optional): To fit an exponential function at shorter atomic distances. Defaults to False.
    """
    coeffs_format = ' '.join(['{:6.3f}'] * 2 + ['{:15.8E}'] * 4) + '\n'
    with open(fname, 'w') as fout:
        fout.write('Spline table\n')
        for index in range(len(twb.interval)-1):
            r_start = twb.interval[index]
            r_stop = twb.interval[index+1]
            fout.write(coeffs_format.format(r_start, r_stop, *coeffs[index]))


def write_error(mdl_eng, ref_eng, mse, fname='error.out'):
    """ Prints the errors in a file
    
    Args:
        mdl_eng (ndarray): Energy prediction values from splines.
        ref_eng (ndarray): Reference energy values.
        mse (float): Mean square error.
        fname (str, optional): Output filename.. Defaults to 'error.out'.
    """
    header = "{:<15}{:<15}{:<15}".format("Reference", "Predicted", "Error")
    error = abs(ref_eng - mdl_eng)
    maxerror = max(abs(error))
    footer = "MSE = {:2.5E}\nMaxerror = {:2.5E}".format(mse, maxerror)
    np.savetxt(fname,
               np.transpose([ref_eng, mdl_eng, error]),
               header=header,
               footer=footer,
               fmt="%-15.5f")
def get_expcoeffs(aa, bb, cc , r0):
    alpha = -cc / bb
    beta = alpha * r0 + np.log(cc / alpha ** 2)
    gamma = aa - np.exp(-alpha * r0 + beta)
    return alpha, beta, gamma


def get_exp_values(coeffs, rr):
    aa, bb, cc = coeffs
    return np.exp(-aa * rr + bb) + cc

def write_as_nxy(fname, datadesc, vectors, column_names):
    header_parts = []
    for ii, colname in enumerate(column_names):
        header_parts.append('Column {}: {}'.format(ii + 1, colname))
    header = '\n'.join(header_parts)
    data = np.array(vectors).transpose()
    np.savetxt(fname, data, header=header)
    print_io_log(fname, datadesc)


def print_io_log(fname, fcontent):
    print("{} -> '{}'".format(fcontent, fname))


def write_splinerep(fname, expcoeffs, splcoeffs, rr, rcut):
    delta=0.01
    fp = open(fname,'w')
    fp.write('Spline\n')
    fp.write('{:d} {:4.3f}\n'.format(len(rr),rcut+delta))
    print(*expcoeffs)
    fp.write('{:15.8E} {:15.8E} {:15.8E}\n'.format(*expcoeffs))
    
    splcoeffs_format = ' '.join(['{:6.3f}'] * 2 + ['{:15.8E}'] * 4) + '\n'
    for ir in range(len(rr) - 1):
        rcur = rr[ir]
        rnext = rr[ir + 1]
        fp.write(splcoeffs_format.format(rcur, rnext, *splcoeffs[ir]))
    poly5coeffs_format = ' '.join(['{:6.3f}'] * 2 + ['{:15.8E}'] * 6) + '\n'
    fp.write(poly5coeffs_format.format(rr[-1],rr[-1]+delta, 0.0, 0.0, 0.0,0.0,0.0,0.0))
    fp.close()
    print_io_log(fname, 'Repulsive in Spline format')

def append_spline (fin, fspl, fout):
    """Take electronic part from fin add fspl and write to fout."""
    with open(fspl, 'r') as f:
        spline = f.readlines()
    with open(fin, 'r') as f:
        skf = f.readlines()
    newskf = []
    for line in skf:
        if line == SPLINETAG:
            break
        else:
            newskf.append(line)
    with open(fout, 'w') as f:
        f.writelines(newskf)
        f.write('\n')         # do we need this line?
        f.writelines(spline)  # assuming fspl already has SPLINETAG

class Twobody():
    """ Twobody class describes properties of an Atom pair"""
    
    def __init__(self,
                 name,
                 Dismat,
                 Nconfigs,
                 Rcut,
                 Nknots,
                 Rmin=None,                 
                 Nswitch=None):
        """ Constructs an Two body object
       
        Args:
            name (str): Name of the atom pair.
            Dismat (dataframe): Pairwise  distance matrix.
            Nconfigs (int): Number of configurations
            Rcut (float): Maximum cut off value for spline interval
            Nknots (int): Number of knots in the spline interval
            Rmin (float, optional): Minimum value of the spline interval. Defaults to None.
            Nswitch (int, optional): The switching point for the spline. Defaults to None.
        """
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
        """ Function for spline matrix
        
        Returns:
            ndarray: v matrix
        """
        return spline_energy_model(self.Rcut, self.Rmin, self.Dismat,
                                   self.cols, self.dx, self.Nconfigs,
                                   self.interval)
