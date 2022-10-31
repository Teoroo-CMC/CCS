import ase.db as db
import re
from ase.io import Trajectory, read, write
from ase.calculators.neighborlist import *
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from sympy import E
from tqdm import tqdm
import copy
import sys
from ase.units import Bohr, Hartree

try:
    from fortformat import Fnetdata
except:
    pass


def ccs_build_db(
    mode=None,
    DFT_DB=None,
    DFTB_DB=None,
    file_list=None,
    Fortnet=False,
    greedy=False,
    greed_threshold=0.0001,
):

    AUtoEvA = Hartree / Bohr

    DFT_DB = db.connect(DFT_DB)
    if mode == "DFTB":
        DFTB_DB = db.connect(DFTB_DB)

    f = open(file_list, "r")
    L = len(f.readlines())
    f.close()
    f = open(file_list, "r")

    fnet_energies = np.empty((L, 1))
    fnet_strucs = []
    fnet_features = []

    counter = 0
    for lns in tqdm(f, total=L):
        counter += 1
        lns = lns.split()
        DFT_FOLDER = lns[0]

        structure_DFT = read(DFT_FOLDER + "/OUTCAR", index=-1)
        EDFT = structure_DFT.get_potential_energy()
        DFT_DB.write(structure_DFT, PBE=True, key=counter)

        # EXTRACT ALL REASONABLE STEPS?
        converged_indices = []
        Natoms = len(structure_DFT)
        if greedy and (mode != "DFTB"):
            f2 = open(DFT_FOLDER + "/OUTCAR", "r")
            outcar = f2.read()
            f2.close
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

            for i in tqdm(converged_indices):
                counter += 1
                structure_DFT = read(DFT_FOLDER + "/OUTCAR", index=i)
                EDFT = structure_DFT.get_potential_energy()
                DFT_DB.write(structure_DFT, PBE=True, key=counter)

        if mode == "DFTB":
            DFTB_FOLDER = lns[1]
            # READ DFTB
            structure_DFTB = copy.deepcopy(structure_DFT)
            f2 = open(DFTB_FOLDER + "/detailed.out", "r")
            EDFTBstr = f2.read()
            EDFTBstr = re.search("Total Electronic energy:(.+?)eV", EDFTBstr).group(1)
            EDFTBstr_arr = EDFTBstr.split()
            EDFTB = float(EDFTBstr_arr[2])
            f2.close

            # READ DFTB FORCES
            f2 = open(DFTB_FOLDER + "/detailed.out", "r")
            time_to_read = False
            DFTB_forces = np.zeros([structure_DFTB.get_global_number_of_atoms(), 3])
            while True:
                next_line = f2.readline()

                if next_line == " \n":
                    time_to_read = False

                if time_to_read:
                    af = next_line.split()
                    DFTB_forces[acnt, 0] = float(af[0]) * AUtoEvA
                    DFTB_forces[acnt, 1] = float(af[1]) * AUtoEvA
                    DFTB_forces[acnt, 2] = float(af[2]) * AUtoEvA
                    acnt += 1

                if "Total Forces" in next_line:
                    time_to_read = True
                    acnt = 0

                if not next_line:
                    break
            f2.close

            # READ MULLIKEN POPULATIONS
            f2 = open(DFTB_FOLDER + "/detailed.out", "r")
            time_to_read = False
            DFTB_MULLIKEN = np.zeros([structure_DFTB.get_global_number_of_atoms(), 4])
            while True:
                next_line = f2.readline()
                if next_line == " \n":
                    time_to_read = False

                if time_to_read:
                    af = next_line.split()
                    at = int(af[0]) - 1
                    orb = int(af[2])
                    DFTB_MULLIKEN[at][orb] += float(af[3])

                if " Atom Sh.   l       Population" in next_line:
                    time_to_read = True

                if not next_line:
                    break

            # READ NET CHARGES
            f2 = open(DFTB_FOLDER + "/detailed.out", "r")
            time_to_read = False
            DFTB_CHARGES = np.zeros([structure_DFTB.get_global_number_of_atoms(), 1])
            while True:
                next_line = f2.readline()
                if next_line == " \n":
                    time_to_read = False

                if time_to_read:
                    af = next_line.split()
                    at = int(af[0]) - 1
                    DFTB_CHARGES[at, 0] = float(af[1])

                if " Atom           Charge" in next_line:
                    time_to_read = True

                if not next_line:
                    break

            f2.close

            calculator = SinglePointCalculator(
                structure_DFTB, energy=EDFTB, free_energy=EDFTB, forces=DFTB_forces
            )

            structure_DFTB.calc = calculator
            structure_DFTB.get_potential_energy()
            structure_DFTB.get_forces()
            DFTB_DB.write(
                structure_DFTB,
                DFTB=True,
                key=counter,
                data={"fnet_charges": DFTB_CHARGES, "fnet_mull": DFTB_MULLIKEN},
            )
            fnet_strucs.append(structure_DFT)
            fnet_features.append(DFTB_MULLIKEN)
            fnet_energies[counter - 1, 0] = EDFT - EDFTB

    f.close()
    if Fortnet == True:
        fnetdata = Fnetdata(
            atoms=fnet_strucs,
            targets=np.asarray(fnet_energies),
            features=fnet_features,
            atomic=False,
        )
        fnetdata.dump("fnetdata.hdf5")


def main():
    print("--- USAGE:  ccs_build_db MODE [...] --- ")
    print(" ")
    print("       The following modes and inputs are supported:")
    print("       CCS:  file_list(string) DFT.db(string) greedy(bool)")
    print("       DFTB: file_list(string) DFT.db(string) DFTB.db(string) Fortnet(bool)")
    print(" ")

    mode = sys.argv[1]
    file_list = sys.argv[2]
    DFT_data = sys.argv[3]
    print("    Mode: ", mode)
    if mode == "CCS":
        greedy = bool(sys.argv[4])
        print("    DFT data base: ", DFT_data)
        print("    Greedy mode: ", greedy)
        print("")
        print("-------------------------------------------------")
        ccs_build_db(mode, DFT_DB=DFT_data, file_list=file_list, greedy=greedy)
    if mode == "DFTB":
        DFTB_data = sys.argv[4]
        Fortnet = bool(sys.argv[5])
        print("    DFT data base: ", DFT_data)
        print("    DFTB data base: ", DFTB_data)
        print("    Create Fortnet data: ", Fortnet)
        print("")
        print("-------------------------------------------------")
        ccs_build_db(
            mode,
            DFT_DB=DFT_data,
            DFTB_DB=DFTB_data,
            file_list=file_list,
            Fortnet=Fortnet,
        )


if __name__ == "__main__":
    main()
