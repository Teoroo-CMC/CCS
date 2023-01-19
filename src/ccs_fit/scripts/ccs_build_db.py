import ase.db as db
import os
import re
from ase.io import Trajectory, read, write

# from ase.calculators.neighborlist import *
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from sympy import E
from tqdm import tqdm
import copy
import sys
from ase.units import Bohr, Hartree


def ccs_build_db(
    mode=None,
    DFT_DB=None,
    DFTB_DB=None,
    file_list=None,
    greedy=False,
    greed_threshold=0.0001,
    overwrite=False,
    verbose=False,
):

    AUtoEvA = Hartree / Bohr

    if os.path.isfile(DFT_DB):
        if overwrite:
            os.remove(DFT_DB)
        else:
            print(
                "DFT database already exists. Please delete the file or use another file name."
            )
            sys.exit()
    DFT_DB = db.connect(DFT_DB)

    if mode == "DFTB":
        if os.path.isfile(DFTB_DB):
            print(
                "DFTB database already exists. Please delete the file or use another file name."
            )
            exit()
        DFTB_DB = db.connect(DFTB_DB)

    try:
        f = open(file_list, "r")
    except FileNotFoundError:
        print("training-set list not found.")
    L = len(f.readlines())
    f.close()
    f = open(file_list, "r")

    counter = 0
    for lns in tqdm(f, total=L, desc="    Building data-bases",):
        counter += 1
        lns = lns.split()
        DFT_FOLDER = lns[0]

        structure_DFT = read(DFT_FOLDER, index=-1)
        EDFT = structure_DFT.get_potential_energy()
        DFT_DB.write(structure_DFT, PBE=True, key=counter)

        # EXTRACT ALL REASONABLE STEPS?
        converged_indices = []
        Natoms = len(structure_DFT)
        if greedy and (mode != "DFTB"):
            try:
                f2 = open(DFT_FOLDER, "r")
                outcar = f2.read()
                f2.close
            except:
                print("Only implemeted for VASP OUTCAR-files.")

            NELM = int(re.findall("NELM   \=(.+?)\;", outcar)[0])
            indices = re.findall("Iteration(.+?)\(", outcar)
            all_energies = re.findall("entropy\=(.+?)e", outcar)
            all_energies = [float(x) / Natoms for x in all_energies]
            indices = [int(x) - 1 for x in indices]
            dE = re.findall("2. order\) :(.+?)\(", outcar)
            uindices = set(indices)
            converged_indices = []
            previous_E = EDFT
            for i in uindices:
                N_SCF = [(el == i) for el in indices]
                if (sum(N_SCF) < NELM) & (
                    abs(all_energies[i] - previous_E) > greed_threshold
                ):
                    converged_indices.append(i)
                    previous_E = all_energies[i]

            for i in converged_indices:
                counter += 1
                structure_DFT = read(DFT_FOLDER, index=i)
                EDFT = structure_DFT.get_potential_energy()
                DFT_DB.write(structure_DFT, PBE=True, key=counter)

        if mode == "DFTB":
            DFTB_FOLDER = lns[1]
            # READ DFTB
            structure_DFTB = copy.deepcopy(structure_DFT)
            f2 = open(DFTB_FOLDER, "r")
            time_to_read = False
            cnt = 0
            while True:
                next_line = f2.readline()

                if time_to_read and cnt < 1:
                    EDFTB = float(next_line) * Hartree
                    cnt += 1

                if "mermin_energy" in next_line:
                    time_to_read = True

                if not next_line:
                    break
            f2.close

            # READ DFTB FORCES
            Natoms = structure_DFTB.get_global_number_of_atoms()
            time_to_read = False
            DFTB_forces = np.zeros([Natoms, 3])
            while True:
                next_line = f2.readline()

                if time_to_read and acnt < Natoms - 1:
                    af = next_line.split()
                    DFTB_forces[acnt, 0] = float(af[0]) * AUtoEvA
                    DFTB_forces[acnt, 1] = float(af[1]) * AUtoEvA
                    DFTB_forces[acnt, 2] = float(af[2]) * AUtoEvA
                    acnt += 1

                if "forces" in next_line:
                    time_to_read = True
                    acnt = 0

                if not next_line:
                    break
            f2.close

            calculator = SinglePointCalculator(
                structure_DFTB, energy=EDFTB, free_energy=EDFTB, forces=DFTB_forces
            )

            structure_DFTB.calc = calculator
            structure_DFTB.get_potential_energy()
            structure_DFTB.get_forces()
            DFTB_DB.write(structure_DFTB, DFTB=True, key=counter)

    f.close()


def main():
    import argparse

    try:
        size = os.get_terminal_size()
        c = size.columns
        txt = "-"*c
        print("")
        print(txt)
        import art
        txt = art.text2art('CCS:Build DB')
        print(txt)
    except:
        pass

    parser = argparse.ArgumentParser(description='CCS fetching tool')
    parser.add_argument("-m", "--mode",         type=str, metavar="",
                        default='CCS',  help="Mode. Availble option: CCS, DFTB")
    parser.add_argument("-d", "--DFT_DB", type=str, metavar="",
                        default='DFT.db',  help="Name of DFT reference data-base")
    parser.add_argument("-dd", "--DFTB_DB", type=str, metavar="",
                        default=None,  help="Name of DFTB reference data-base")
    parser.add_argument("-g", "--greedy", type=bool,  metavar="",
                        default=False, help="Extract geometry optmization steps from OUTCAR")
    parser.add_argument("-gt", "--greed_threshold", type=float, metavar="",
                        default=0.0001, help="minimum energy difference between steps extracted using option -g")
    parser.add_argument("-v", "--verbose",
                        action="store_true", help="Verbose output")

    args = parser.parse_args()

    ccs_build_db(**vars(args))

    print("    USAGE:  ccs_build_db MODE [...] ")
    print(" ")
    print("    The following modes and inputs are supported:")
    print("")
    print("        CCS:  file_list(string) DFT.db(string) greedy(bool)")
    print("        DFTB: file_list(string) DFT.db(string) DFTB.db(string)")
    print(" ")

    assert sys.argv[1] in ["CCS", "CCS+Q", "DFTB"], "Mode not supported."
    
    mode = sys.argv[1]
    file_list = sys.argv[2]
    DFT_data = sys.argv[3]
    print("    Mode: ", mode)
    if mode == "CCS" or mode == "CCS+Q":
        greedy = bool(sys.argv[4])
        print("    DFT data base: ", DFT_data)
        print("    Greedy mode: ", greedy)
        print("")

        ccs_build_db(mode, DFT_DB=DFT_data, file_list=file_list, greedy=greedy)
    if mode == "DFTB":
        DFTB_data = sys.argv[4]
        print("    DFT data base: ", DFT_data)
        print("    DFTB data base: ", DFTB_data)
        print("")

        ccs_build_db(mode, DFT_DB=DFT_data,
                     DFTB_DB=DFTB_data, file_list=file_list)

    try:
        size = os.get_terminal_size()
        c = size.columns
        txt = "-"*c
        print(txt)
        print("")
    except:
        pass


if __name__ == "__main__":
    main()
