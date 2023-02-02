# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                           #
#  Copyright (C) 2019 - 2023  CCS developers group                              #
#                                                                               #
#  See the LICENSE file for terms of usage and distribution.                    #
# ------------------------------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import epsilon_0

def Buckingham(r, A, B, C):
    return A*np.exp(-B*r) - C/(r**6)

def Buckingham_Coulomb(r, q1, q2, A, B, C):
    return A*np.exp(-B*r) - C/(r**6) + q1*q2/(4*np.pi*epsilon_0*r)

def Lennard_Jones(r, eps, sigma):
    sig_r6 = (sigma/r)**6
    return 4*eps(sig_r6**2 - sig_r6)

def Morse(r, De, a, re):
    return De*(1-np.exp(-a(r-re)))**2
    