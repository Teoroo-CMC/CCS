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
import scipy.linalg as linalg

logger = logging.getLogger(__name__)


def spline_construction(rows, cols, dx):
    ''' This function constructs the matrices A, B, C, D.

    Args:

        rows (int): The row dimension for matrix
        cols (int): The column dimension of the matrix
        dx (list): grid space ((rcut - rmin) / N)

    Returns:

        cc, dd, bb, aa (matrices): constructed matrices

    '''

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


def spline_eval012(aa, bb, cc, dd, rr, rmin, dx, xx):
    '''Returns cubic spline value, first and second derivative at a point r.

    Args:

        aa (ndarray): a coefficients of the spline.
        bb (ndarray): b coefficients of the spline.
        cc (ndarray): c coefficients of the spline.
        dd (ndarray): d coefficients of the spline.
        rr (float): point to evaluate the spline function.
        rmin (float): the min value cut off for spline interval.
        dx (float):  grid space.
        xx (interval): spline interval.

    Raises:

        ValueError: if the point to be evaluated is below cut-off

    Returns:

        floats: values for spline function value, first and second derivative

    '''

    if rr == rmin:
        index = 1
    else:
        index = int(np.ceil((rr - rmin) / dx))

    if index >= 1:
        dr = rr - xx[index]
        f0 = aa[index - 1] + dr \
            * (bb[index - 1] + dr \
               * (0.5 * cc[index - 1] + (dd[index - 1] * dr / 6.0))
        )
        f1 = bb[index - 1] + dr * (cc[index - 1] + (0.5 * dd[index - 1] * dr))
        f2 = cc[index - 1] + dd[index - 1] * dr

        return float(f0), float(f1), float(f2)

    raise ValueError('r < Rmin')


def get_spline_coeffs(xx, yy, deriv0, deriv1):
    '''Spline coefficients for a spline with given 1st derivatives at its ends.

    Args:

        xx ():
        yy ():
        deriv0 ():
        deriv1 ():

    Returns:

        np.transpose(mtx): spline coefficients

    '''

    nn = len(xx)
    kk = xx[1:] - xx[:-1]
    mu = kk[:-1] / (kk[:-1] + kk[1:])
    mun = 1.0
    alpha = 1.0 - mu
    alpha0 = 1.0
    mu = np.hstack((mu, [mun]))
    alpha = np.hstack(([alpha0], alpha))
    dd = (6.0 / (kk[:-1] + kk[1:])
          * ((yy[2:] - yy[1:-1]) / kk[1:] - (yy[1:-1] - yy[0:-2]) / kk[:-1]))
    d0 = 6.0 / kk[0] * ((yy[1] - yy[0]) / kk[0] - deriv0)
    dn = 6.0 / kk[-1] * (deriv1 - (yy[-1] - yy[-2]) / kk[-1])
    dd = np.hstack((d0, dd, dn))
    mtx = 2.0 * np.identity(nn, dtype=float)
    for ii in range(nn - 1):
        mtx[ii, ii + 1] = alpha[ii]
        mtx[ii + 1, ii] = mu[ii]
    mm = linalg.solve(mtx, dd)
    c0 = yy[:-1]
    c1 = (yy[1:] - yy[:-1]) / kk - (2.0 * mm[:-1] + mm[1:]) / 6.0 * kk
    c2 = mm[:-1] / 2.0
    c3 = (mm[1:] - mm[:-1]) / (6.0 * kk)
    mtx = np.array([c0, c1, c2, c3])

    return np.transpose(mtx)


def spline_energy_model(rcut, rmin, df, cols, dx, size, xx):
    '''Constructs the v matrix.

    Args:

        rcut (float): max value cutoff for spline interval.
        rmin (float): min value cutoff for spline interval.
        df (ndarray): paiwise distance matrix.
        cols (int):  number of unknown parameters.
        dx (float): grid size.
        size (int): number of configuration.
        xx (list): spline interval.

    Returns:

        ndarray: The v matrix for a pair.

    '''

    cc, dd, bb, aa = spline_construction(cols - 1, cols, dx)
    logger.debug(" Number of configuration for v matrix: %s", size)
    logger.debug("\n A matrix is: \n %s \n Spline interval = %s", aa, xx)
    vv = np.zeros((size, cols))
    indices = []
    for config in range(size):
        distances = [ii for ii in df[config, :] if rmin <= ii <= rcut]
        uu = 0
        for rr in distances:
            index = int(np.ceil(np.around(((rr - rmin) / dx), decimals=5)))
            indices.append(index)
            delta = rr - xx[index]
            logger.debug("\n In config %s\t distance r = %s\tindex=%s\tbin=%s",
                         config, rr, index, xx[index])
            aa_ind = aa[index - 1]
            bb_ind = bb[index - 1] * delta
            dd_ind = dd[index - 1] * np.power(delta, 3) / 6.0
            c_d = cc[index - 1] * np.power(delta, 2) / 2.0
            uu = uu + aa_ind + bb_ind + c_d + dd_ind

        vv[config, :] = uu

    logger.debug("\n V matrix :%s", vv)

    return vv, set(indices)


def write_splinecoeffs(twb, coeffs, fname='splines.out'):
    '''Writes the spline output.

    Args:

        twb (Twobody): Twobody class object.
        coeffs (ndarray): Array containing spline coefficients.
        fname (str, optional): Filename to output the spline coefficients
            (default: 'splines.out').

    '''

    coeffs_format = ' '.join(['{:6.3f}'] * 2 + ['{:15.8E}'] * 4) + '\n'

    with open(fname, 'w') as fout:
        fout.write('Spline table\n')
        for index in range(len(twb.interval) - 1):
            r_start = twb.interval[index]
            r_stop = twb.interval[index + 1]
            fout.write(coeffs_format.format(r_start, r_stop, *coeffs[index]))


def write_error(mdl_eng, ref_eng, mse, fname='error.out'):
    '''Prints the errors in a file.

    Args:

        mdl_eng (ndarray): Energy prediction values from splines.
        ref_eng (ndarray): Reference energy values.
        mse (float): Mean square error.
        fname (str, optional): Output filename (default: 'error.out').

    '''

    header = '{:<15}{:<15}{:<15}'.format('Reference', 'Predicted', 'Error')
    error = abs(ref_eng - mdl_eng)
    maxerror = max(abs(error))
    footer = 'MSE = {:2.5E}\nMaxerror = {:2.5E}'.format(mse, maxerror)
    np.savetxt(fname, np.transpose([ref_eng, mdl_eng, error]), header=header,
               footer=footer, fmt='%-15.5f')


def get_expcoeffs(aa, bb, cc, r0):
    '''Calculates coefficients of exponential function.

    Args:

        aa (float):
        bb (float):
        cc (float):
        r0 (float):

    Returns:

        alpha (float):
        beta (float):
        gamma (float):

    '''

    alpha = -cc / bb
    beta = alpha * r0 + np.log(cc / alpha ** 2)
    gamma = aa - np.exp(-alpha * r0 + beta)

    return alpha, beta, gamma


def get_exp_values(coeffs, rr):
    '''Calculates exponential function.

    Args:

        coeffs (tupel): contains coefficients of exponential function
        rr (float): values to calculate exponential for

    Returns:

        exp_vals (1darray): exponential values

    '''

    aa, bb, cc = coeffs
    exp_vals = np.exp(-aa * rr + bb) + cc

    return exp_vals


def write_as_nxy(fname, datadesc, vectors, column_names):
    '''Writes to file in nxy format.

    Args:

        fname (str): filename
        datadesc ():
        vectors ():
        column_names ():

    '''

    header_parts = []
    for ii, colname in enumerate(column_names):
        header_parts.append('Column {}: {}'.format(ii + 1, colname))
    header = '\n'.join(header_parts)
    data = np.array(vectors).transpose()
    np.savetxt(fname, data, header=header)
    print_io_log(fname, datadesc)


def print_io_log(fname, fcontent):
    '''Prints some IO information.

    Args:

        fname (str): filename
        fcontent (str): descriptor of content

    '''

    print("{} -> '{}'".format(fcontent, fname))


def write_splinerep(fname, expcoeffs, splcoeffs, rr, rcut):
    '''Calculates coefficients of exponential function.

    Args:

        fname (str): filename
        expcoeffs ():
        splcoeffs ():
        rr ():
        rcut ():

    '''

    delta = 0.01

    with open(fname, 'w') as fp:
        fp.write('Spline\n')
        fp.write('{:d} {:4.3f}\n'.format(len(rr), rcut + delta))
        fp.write('{:15.8E} {:15.8E} {:15.8E}\n'.format(*expcoeffs))

        splcoeffs_format = ' '.join(['{:6.3f}'] * 2 + ['{:15.8E}'] * 4) + '\n'
        for ir in range(len(rr) - 1):
            rcur = rr[ir]
            rnext = rr[ir + 1]
            fp.write(splcoeffs_format.format(rcur, rnext, *splcoeffs[ir]))
        poly5coeffs_format = ' '.join(['{:6.3f}'] * 2 + ['{:15.8E}'] * 6) + '\n'
        fp.write(poly5coeffs_format.
                 format(rr[-1], rr[-1] + delta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    print_io_log(fname, 'Repulsive in Spline format')


def append_spline(fin, fspl, fout):
    '''Take electronic part from fin add fspl and write to fout.

    Args:

        fin (str): input filename
        fspl (str): spline filename
        fout (str): output filename

    '''

    newskf = []

    with open(fspl, 'r') as fid:
        spline = fid.readlines()

    with open(fin, 'r') as fid:
        skf = fid.readlines()

    for line in skf:
        if line == SPLINETAG:
            break
        else:
            newskf.append(line)

    with open(fout, 'w') as fid:
        fid.writelines(newskf)
        fid.write('\n')
        fid.writelines(spline)


def spline_mask(rcut, rmin, df, cols, dx, size):
    '''Constructs the mask matrix.

    Args:

        rcut (float): max value cut off for spline interval.
        rmin (float): min value cut off for spline interval.
        df (ndarray): paiwise distance matrix.
        cols (int):  number of unknown parameters.
        dx (float): grid size.
        size (int): number of configuration.

    Returns:

        mask (ndarray): mask matrix for a pair.

    '''

    mask = np.zeros([cols, 1])
    for config in range(size):
        distances = [ii for ii in df[config, :] if rmin <= ii <= rcut]
        for rr in distances:
            index = int(np.ceil(np.around(((rr - rmin) / dx), decimals=5))) - 1
            mask[index] = 1

    return mask


class Twobody:
    '''Twobody class that describes properties of an Atom pair.'''


    def __init__(self, name, dismat, nconfigs, Rcut, Nknots, Swtype='rep',
                 Rmin=None, nswitch=None):
        '''Constructs a Twobody object.

        Args:

            name (str): name of the atom pair.
            dismat (dataframe): pairwise  distance matrix.
            nconfigs (int): number of configurations
            Rcut (float): maximum cut off value for spline interval
            Nknots (int): number of knots in the spline interval
            Rmin (float, optional): minimum value of the spline interval
                (default: None).
            nswitch (int, optional): switching point for the spline
                (default: None).

        '''

        self.name = name
        self.Rcut = Rcut
        self.Rmin = Rmin
        self.Nknots = Nknots
        self.nswitch = nswitch
        self.Swtype = Swtype
        self.dismat = dismat
        self.nconfigs = nconfigs
        self.dx = (self.Rcut - self.Rmin) / self.Nknots
        self.cols = self.Nknots + 1
        self.interval = np.linspace(self.Rmin, self.Rcut, self.cols,
                                    dtype=float)
        self.cc, self.dd, self.bb, self.aa = \
            spline_construction(self.cols - 1, self.cols, self.dx)
        self.vv, self.indices = self.get_v()
        self.mask = self.get_mask()


    def get_v(self):
        '''Returns spline matrix.

        Returns:

            ndarray: v matrix

        '''

        return spline_energy_model(self.Rcut, self.Rmin, self.dismat, self.cols,
                                   self.dx, self.nconfigs, self.interval)


    def get_mask(self):
        ''' Returns spline mask.

        Returns:

            ndarray: mask vector

        '''

        return spline_mask(self.Rcut, self.Rmin, self.dismat, self.cols,
                           self.dx, self.nconfigs)
