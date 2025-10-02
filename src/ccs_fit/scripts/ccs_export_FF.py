import sys
import os
import numpy as np
import json
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from ccs_fit.ase_calculator.ccs_ase_calculator import spline_table
from ccs_fit.scripts.helper import terminal_header


def Buckingham(r, A, B, C):
    return A * np.exp(-B * r) - C / (r**6)


def Lennard_Jones(r, eps, sigma):
    sig_r6 = (sigma / r) ** 6
    return 4 * eps * (sig_r6**2 - sig_r6)


def Morse(r, De, a, re):
    return De * ((1 - np.exp(-a * (r - re))) ** 2 - 1)


def Pedone(r, De, a, re, C):
    return De * ((1 - np.exp(-a * (r - re))) ** 2 - 1) + C / r**12


def _write(elem1, elem2, CCS_params, f_Buck, f_LJ, f_Mor, f_Ped):
    elem1 = elem1
    elem2 = elem2
    no_pair = False
    try:
        pair = elem1 + "-" + elem2
    except:
        try:
            pair = elem2 + "-" + elem1
        except:
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

        try:
            popt_Buck = curve_fit(Buckingham, r, spl_to_fit, maxfev=5000)
            print(
                "Buckingham fit (not optimised) for element pair {}-{};     V(r) = {:.2f}*exp(-{:.2f}*r) -({:.2f})/r^6.".format(
                    elem1, elem2, popt_Buck[0], popt_Buck[1], popt_Buck[2]
                )
            )
            print(
                "{:^8s} {:^8s} {:20.10f} {:20.10f} {:20.10f}".format(
                    elem1, elem2, popt_Buck[0], popt_Buck[1], popt_Buck[2]
                ),
                file=f_Buck,
            )
            plt.plot(
                r,
                Buckingham(r, popt_Buck[0], popt_Buck[1], popt_Buck[2]),
                "b",
                label="Buck",
            )
        except:
            print("Buckingham potential not found within max iteration bound.")

        try:
            popt_LJ = curve_fit(
                Lennard_Jones, r, spl_to_fit, p0=[1, 1], maxfev=5000
            )
            print(
                "Lennard Jones fit (not optimised) for element pair {}-{};  V(r) = 4*{:.2f}*(({:.2f}/r)^12 - ({:.2f}/r)^6)".format(
                    elem1, elem2, popt_LJ[0], popt_LJ[1], popt_LJ[1]
                )
            )
            print(
                "{:^8s} {:^8s} {:20.10f} {:20.10f}".format(
                    elem1, elem2, popt_LJ[0], popt_LJ[1]
                ),
                file=f_LJ,
            )
            plt.plot(
                r,
                Lennard_Jones(r, popt_LJ[0], popt_LJ[1]),
                color="magenta",
                label="LJ",
            )
        except:
            print(
                "Lennard-Jones potential not found within max iteration bound."
            )

        try:
            popt_Mor = curve_fit(
                Morse, r, spl_to_fit, p0=[2, 1, 1], maxfev=5000
            )
            print(
                "Morse fit (not optimised) for element pair {}-{};          V(r) = {:.2f}*((1-np.exp(-{:.2f}*(r-{:.2f})))^2 - 1)".format(
                    elem1, elem2, popt_Mor[0], popt_Mor[1], popt_Mor[2]
                )
            )
            print(
                "{:^8s} {:^8s} {:20.10f} {:20.10f} {:20.10f}".format(
                    elem1, elem2, popt_Mor[0], popt_Mor[1], popt_Mor[2]
                ),
                file=f_Mor,
            )
            plt.plot(
                r,
                Morse(r, popt_Mor[0], popt_Mor[1], popt_Mor[2]),
                color="orange",
                label="Morse",
            )
        except:
            print("Morse potential not found within max iteration bound.")

        try:
            popt_Ped = curve_fit(
                Pedone,
                r,
                spl_to_fit,
                p0=[popt_Mor[0], popt_Mor[1], popt_Mor[2], 1],
                maxfev=5000,
            )
            print(
                "Pedone fit (not optimised) for element pair {}-{};         V(r) = {:.2f}*((1-np.exp(-{:.2f}*(r-{:.2f})))^2 - 1) + {:.2f}/r^12".format(
                    elem1,
                    elem2,
                    popt_Ped[0],
                    popt_Ped[1],
                    popt_Ped[2],
                    popt_Ped[3],
                )
            )
            print(
                "{:^8s} {:^8s} {:20.10f} {:20.10f} {:20.10f} {:20.10f}".format(
                    elem1,
                    elem2,
                    popt_Ped[0],
                    popt_Ped[1],
                    popt_Ped[2],
                    popt_Ped[3],
                ),
                file=f_Ped,
            )
            plt.plot(
                r,
                Pedone(r, popt_Ped[0], popt_Ped[1], popt_Ped[2], popt_Ped[3]),
                color="green",
                label="Pedone",
            )
        except:
            print("Pedone potential not found within max iteration bound.")

        plt.plot(r, spl_to_fit, "r--", label="CCS")
        plt.xlabel("Interatomic distance (Å)")
        plt.ylabel("Potential energy (eV)")
        plt.legend()
        plt.show()


def write_LAMMPS(jsonfile, form="uf3",prefix="CCS",include_head=False):
    json_file = open(jsonfile)
    CCS_params = json.load(json_file)
    tags = {}
    filename = "CCS." + form
    if form == 'table':

        import math
        dr=0.005
        with open(filename, "w") as f:
            rmin=np.inf
            rcut=0.0
            for pair in CCS_params["Two_body"].keys():
                if CCS_params["Two_body"][pair]["r_min"] < rmin:
                    rmin=CCS_params["Two_body"][pair]["r_min"]
                if CCS_params["Two_body"][pair]["r_cut"] > rcut:
                    rcut=CCS_params["Two_body"][pair]["r_cut"]
            if include_head:
                rmin=0.005
            rmin=math.floor(rmin / dr) * dr
            rcut=math.ceil(rcut / dr) * dr
            for pair in CCS_params["Two_body"].keys():
                elem1, elem2 = pair.split("-")
                tb = spline_table(elem1, elem2, CCS_params)
                r = np.arange(rmin, rcut+dr , dr) #SHOULD IT BE LIKE THIS ?
                sr_scl=0.0
                if "Range_sep" in CCS_params:
                    if CCS_params["Range_sep"] == "Stitch":
                        sr_scl=14.39964547842567*CCS_params["Charges"][elem1]*CCS_params["Charges"][elem2]  
                f.write("\n {}".format(pair))
                f.write("\n N {} \n".format(len(r)))


                for index, elem in enumerate(r):
                    if elem >= CCS_params["Two_body"][pair]["r_cut"]:
                        energy = 0.0
                        force = 0.0
                    else:
                        energy = tb.eval_energy(elem) - sr_scl * (1 / elem)
                        force = tb.eval_force(elem) - sr_scl * (1 / (elem ** 2))

                    f.write("\n {} {:.12f} {} {}".format(index + 1, elem, energy, force))



        print("# Specifications for lammps input.")
        element_list={}
        for i,elem in enumerate( CCS_params["One_body"] ):
            print(f"group {elem}  type {i+1}")
            element_list[elem]=i+1
        if "Charges"  in CCS_params: 
            print("")
            for q in  CCS_params["Charges"]:
                print("set type {} charge {}".format(element_list[q],  CCS_params["Charges"][q]))
            print("")
            print("kspace_style ewald 1e-12")
            print("")
            print("pair_style   hybrid/overlay coul/long {:.6f} table spline {} ewald".format(rcut,len(r)))
            print("pair_coeff   *    *    coul/long")
            for pair in CCS_params["Two_body"].keys():
                elem1, elem2 = pair.split("-")
                print("pair_coeff    {} {} table {} {} {:.6f}".format(element_list[elem1],element_list[elem2],filename,pair,rcut  ))
        else:
            print("")
            print("pair_style table spline")
            for pair in CCS_params["Two_body"].keys():
                elem1, elem2 = pair.split("-")
                print("pair_coeff    {} {} {} {} {:.6f}".format(element_list[elem1],element_list[elem2],filename,pair,rcut  ))


    if form == 'uf3':
        from scipy.interpolate import make_interp_spline
        with open(filename, "w") as f:
            for pair in CCS_params["Two_body"].keys():
                f.write("#UF3 POT UNITS: metal DATE: today AUTHOR: Me CITATION: please\n")
                elem1, elem2 = pair.split("-")
                sr_scl=0.0
                if "Range_sep" in CCS_params:
                    if CCS_params["Range_sep"] == "Stitch":
                        sr_scl=14.39964547842567*CCS_params["Charges"][elem1]*CCS_params["Charges"][elem2]  
                tb = spline_table(elem1, elem2, CCS_params)
                rmin=CCS_params["Two_body"][pair]["r_min"]
                rmax=CCS_params["Two_body"][pair]["r_cut"]
                dr=CCS_params["Two_body"][pair]["dr"]
                r = np.arange(rmin,rmax+dr , dr)
                c_pts = [0.5*tb.eval_energy(elem)+0.5*sr_scl/elem for elem in r] #0.5 to compensate for double counting
                c_pts = np.array(c_pts)
                bsplines=make_interp_spline(r, c_pts, k=3,bc_type=([(1,-0.5*tb.eval_force(r[0]) -0.5*sr_scl/(r[0]**2)  )],[(1,-0.5*tb.eval_force(r[-1]) -0.5*sr_scl/(r[-1]**2))]))
                r=bsplines.t
                coefs=bsplines.c
                f.write("2B {} {} 0 3 nk\n".format(elem1,elem2))
                f.write("{:.6f} {} \n".format(r[-1],len(r)))
                [f.write("{:.6f} ".format(r_val)) for r_val in r]
                f.write("\n")
                f.write("{} \n".format(len(coefs)))
                [f.write(" {:.12e}".format(co)) for co in coefs]
                f.write("\n#\n")
        print("# Specifications for lammps input.")
        element_list={}
        for i,elem in enumerate( CCS_params["One_body"] ):
            print(f"group {elem}  type {i+1}")
            element_list[elem]=i+1
        reversed_element_list = {v: k for k, v in element_list.items()}
        if "Charges" in  CCS_params: 
            print("")
            for q in  CCS_params["Charges"]:
                print("set type {} charge {}".format(element_list[q],  CCS_params["Charges"][q]))
            print("")
            print("kspace_style ewald 1e-12")
            print("")
            print("pair_style   hybrid/overlay coul/long 12.0 uf3 2")
            print("pair_coeff   * * coul/long")
            print("pair_coeff   * * uf3 {} ".format(filename),end="")
            for i in range(len(element_list)):
                print(" {}".format(reversed_element_list[i+1]),end="")
            print("")
        else:
            print("")
            print("pair_style uf3 2 ")
            print("pair_coeff   * * {} ".format(filename),end="")
            for i in range(len(element_list)):
                print(" {}".format(reversed_element_list[i+1]),end="")
            print("")




def write_GULP(jsonfile, form='GULP',include_head=False):
    """
    spline <cubic> <reverse> <intra/inter> <bond/x12/x13/x14/mol/o14/g14> <kcal/kjmol> <type_of_bond>
    atom1 atom2 <shift> <rmin> rmax <1*flag>
    energy_1 distance_1
    energy_2 distance_2
    """
    json_file = open(jsonfile)
    CCS_params = json.load(json_file)
    tags = {}
    filename = "CCS." + form
    with open(filename, "w") as f:
        dr=0.005
        for pair in CCS_params["Two_body"].keys():
            elem1, elem2 = pair.split("-")
            tb = spline_table(elem1, elem2, CCS_params)
            rmin = CCS_params["Two_body"][pair]["r_min"]
            if include_head:
                rmin=0.0
            r = np.arange(rmin, tb.Rcut + 2*dr, dr)
            tags[pair] = dict(
                {"Rmin": rmin, "Rcut": tb.Rcut, "dr": dr, "N": len(r)}
            )
            f.write("\n spline reverse")
            f.write("\n {} {} 0 {} {}".format(elem1, elem2, rmin, tb.Rcut + 2*dr))
            [
                f.write("\n {} {}".format(elem, tb.eval_energy(elem)))
                for elem in r
            ]

    return tags


def ccs_export_FF(CCS_params=None,form="lammps_table", include_head=False):

    if form == "lammps_table":
        write_LAMMPS(CCS_params,form="table",include_head=include_head)

    if form == "lammps_uf3":
        write_LAMMPS(CCS_params,form="uf3")

    if form == "gulp_table":
        write_GULP(CCS_params,form="GULP",include_head=include_head)

    if form == "analytic":
        with open(CCS_params, "r") as f:
            CCS_params = json.load(f)
        f_Buck = open("Buckingham.dat", "w")
        f_LJ = open("Lennard_Jones.dat", "w")
        f_Mor = open("Morse.dat", "w")
        f_Ped = open("Pedone.dat", "w")

        print(
            "{:^8s} {:^8s} {:^20s} {:^20s} {:^20s}\n".format(
                "Element", "Element", "A", "B", "C"
            ),
            file=f_Buck,
        )
        print(
            "{:^8s} {:^8s} {:^20s} {:^20s}\n".format(
                "Element", "Element", "epsilon", "sigma"
            ),
            file=f_LJ,
        )
        print(
            "{:^8s} {:^8s} {:^20s} {:^20s} {:^20s}\n".format(
                "Element", "Element", "D_e", "a", "r_e"
            ),
            file=f_Mor,
        )
        print(
            "{:^8s} {:^8s} {:^20s} {:^20s} {:^20s} {:^20s}\n".format(
                "Element", "Element", "D_e", "a", "r_e", "C"
            ),
            file=f_Ped,
        )

        for pair in CCS_params["Two_body"]:
            elem = pair.split("-")
            _write(
                elem[0],
                elem[1],
                CCS_params,
                f_Buck=f_Buck,
                f_LJ=f_LJ,
                f_Mor=f_Mor,
                f_Ped=f_Ped,
            )


def main():
    import argparse

    terminal_header("C3S : export FF params")

    parser = argparse.ArgumentParser(description="C3S exporting tool")
    parser.add_argument(
        "-f",
        "--form",
        type=str,
        metavar="",
        default="lammps_table",
        help="Format. Availble option: lammps_uf3, lammps_table, gulp_table, analytic",
    )
    parser.add_argument(
        "-head",
        "--include_head",
        type=bool,
        metavar="",
        default=False,
        help="Include exponential head",
    )
    parser.add_argument(
        "-p",
        "--CCS_params",
        type=str,
        metavar="",
        default="CCS_params.json",
        help="Parameter file. Default CCS_params.json",
    )


    args = parser.parse_args()

    ccs_export_FF(**vars(args))



if __name__ == "__main__":
    main()
