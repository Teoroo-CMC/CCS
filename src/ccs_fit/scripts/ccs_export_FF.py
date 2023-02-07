import sys
import os
import numpy as np
import itertools as it
from collections import OrderedDict, defaultdict
from numpy import linalg as LA
import json
from ase.units import Bohr, Hartree
from scipy.optimize import curve_fit
from scipy.constants import epsilon_0
import matplotlib.pyplot as plt

def Buckingham(r, A, B, C):
    return A*np.exp(-B*r) - C/(r**6)

# def Buckingham_Coulomb(r, q1, q2, A, B, C):
#     return A*np.exp(-B*r) - C/(r**6) + q1*q2/(4*np.pi*epsilon_0*r)

def Lennard_Jones(r, eps, sigma):
    sig_r6 = (sigma/r)**6
    return 4*eps*(sig_r6**2 - sig_r6)

def Morse(r, De, a, re, C):
    return De*(1-np.exp(-a*(r-re)))**2 + C 

def _write(elem1, elem2, CCS_params, f_Buck, f_LJ, f_Mor, exp=True):
    elem1 = elem1
    elem2 = elem2
    no_pair = False
    try:
        pair = elem1 + "-" + elem2
        rcut = CCS_params["Two_body"][pair]["r_cut"]
    except:
        try:
            pair = elem2 + "-" + elem1
            rcut = CCS_params["Two_body"][pair]["r_cut"]
        except:
            rcut = 0.0
            no_pair = True
    if no_pair:
        pass
    else:
        Rmin = CCS_params["Two_body"][pair]["r_min"] 
        Rcut = CCS_params["Two_body"][pair]["r_cut"] 
        a = CCS_params["Two_body"][pair]["spl_a"]
        b = CCS_params["Two_body"][pair]["spl_b"]
        c = CCS_params["Two_body"][pair]["spl_c"]
        d = CCS_params["Two_body"][pair]["spl_d"]
        aa = CCS_params["Two_body"][pair]["exp_a"]
        bb = CCS_params["Two_body"][pair]["exp_b"]
        cc = CCS_params["Two_body"][pair]["exp_c"]
        exp = False
        x = CCS_params["Two_body"][pair]["r"]
        dx = CCS_params["Two_body"][pair]["dr"]

        r = np.linspace(Rmin, Rcut, 1000)
        spl_to_fit = []

        for cur_r in r:
            if cur_r >= Rmin and cur_r < Rcut:
                index = int(np.floor((cur_r - Rmin) / dx))
                dr = cur_r - x[index]
                f0 = a[index] + dr * (
                b[index] + dr * (c[index] + (d[index] * dr))
                )
                spl_to_fit.append(f0)
            elif cur_r < Rmin:
                val = np.exp(-aa * cur_r + bb) + cc
                spl_to_fit.append(val)
            elif cur_r >= Rcut:
                spl_to_fit.append(0)
    
        def Lennard_Jones(r, eps, sigma):
            sig_r6 = (sigma/r)**6
            return 4*eps*(sig_r6**2 - sig_r6)

        def Morse(r, De, a, re, C):
            return De*(1-np.exp(-a*(r-re)))**2 + C 

        popt_Buck, pcov = curve_fit(Buckingham, r, spl_to_fit, maxfev=5000)
        popt_Mor, pcov = curve_fit(Morse, r, spl_to_fit, p0=[2,1,1,0], bounds=[[0, 0, 0, -np.inf],[np.inf, np.inf, np.inf, np.inf]], maxfev=5000)
        popt_LJ, pcov = curve_fit(Lennard_Jones, r, spl_to_fit, p0=[1,1], maxfev=5000)
        print("Buckingham fit (not optimised) for element pair {}-{};     V(r) = {:.2f}*exp(-{:.2f}*r) -({:.2f})/r^6.".format(elem1, elem2, popt_Buck[0], popt_Buck[1], popt_Buck[2]))
        print("Lennard Jones fit (not optimised) for element pair {}-{};  V(r) = 4*{:.2f}*(({:.2f}/r)^12 - ({:.2f}/r)^6)".format(elem1, elem2, popt_LJ[0], popt_LJ[1], popt_LJ[1]))
        print("Morse fit (not optimised) for element pair {}-{};          V(r) = {:.2f}*(1-np.exp(-{:.2f}*(r-{:.2f})))^2 + {:.2f}".format(elem1, elem2, popt_Mor[0], popt_Mor[1], popt_Mor[2], popt_Mor[3]))


        print("{:^8s} {:^8s} {:20.10f} {:20.10f} {:20.10f}".format(elem1, elem2, popt_Buck[0], popt_Buck[1], popt_Buck[2]), file=f_Buck)
        print("{:^8s} {:^8s} {:20.10f} {:20.10f}".format(elem1, elem2, popt_LJ[0], popt_LJ[1]), file=f_LJ)
        print("{:^8s} {:^8s} {:20.10f} {:20.10f} {:20.10f} {:20.10f}".format(elem1, elem2, popt_Mor[0], popt_Mor[1], popt_Mor[2], popt_Mor[3]), file=f_Mor)


        plt.plot(r, spl_to_fit, 'r', label='CCS')
        plt.plot(r, Buckingham(r, popt_Buck[0], popt_Buck[1], popt_Buck[2]), 'b', label = 'Buck')
        plt.plot(r, Morse(r, popt_Mor[0], popt_Mor[1], popt_Mor[2], popt_Mor[3]), color='orange', label='Morse')
        plt.plot(r, Lennard_Jones(r, popt_LJ[0], popt_LJ[1]), color='magenta', label='LJ')
        plt.xlabel("Interatomic distance (Å)")
        plt.ylabel("Potential energy (eV)")
        plt.legend()
        plt.show()

def write_FF(CCS_params_file):

    with open(CCS_params_file, "r") as f:
        CCS_params = json.load(f)

    f_Buck = open("Buckingham.dat", "w")
    f_LJ = open("Lennard_Jones.dat", "w")
    f_Mor = open("Morse.dat", "w")

    print("{:^8s} {:^8s} {:^20s} {:^20s} {:^20s}\n".format("Element", "Element", "A", "B", "C"), file=f_Buck)
    print("{:^8s} {:^8s} {:^20s} {:^20s}\n".format("Element", "Element", "ε", "σ"), file=f_LJ)
    print("{:^8s} {:^8s} {:^20s} {:^20s} {:^20s} {:^20s}\n".format("Element", "Element", "D_e", "a", "r_e", "C"), file=f_Mor)
    
    for pair in CCS_params["Two_body"]:
        elem = pair.split("-")
        _write(elem[0], elem[1], CCS_params, exp=True, f_Buck=f_Buck, f_LJ=f_LJ, f_Mor=f_Mor)


def main():

    size = os.get_terminal_size()
    c = size.columns
    txt = "-"*c
    print("")
    print(txt)

    try:
        import art
        txt = art.text2art('Exporting Force Field parameters for Buckingham, Morse and Lennard Jones potentials.')
        print(txt)
    except:
        print('Exporting Force Field parameters for Buckingham, Morse and Lennard Jones potentials.')

    try:
        CCS_params_file = sys.argv[1]
    except:
        print("Please provide CCS params-file as first argument.")
        exit()

    size = os.get_terminal_size()
    c = size.columns
    txt = "-"*c
    print(txt)
    print("")

    write_FF(CCS_params_file)


if __name__ == "__main__":
    main()
