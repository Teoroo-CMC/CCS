import sys
import os
import numpy as np
import ase.db as db
import json
import random
from ase.calculators.singlepoint import SinglePointCalculator
from tqdm import tqdm
import time
from ase.calculators.mixing import LinearCombinationCalculator

from ccs_fit.ase_calculator.ccs_ase_calculator import CCS


def ccs_validate(
    mode=None,
    CCS_params=None,
    Ns='all',
    DFT_DB="DFT.db",
    CCS_DB="CCS_validate.db",
    DFTB_DB=None,
    charge_dict=None,
    charge_scaling=False,
    include_forces=False,
    verbose=False,
):
    """
    Function to verify database generation.
    ---------------------------------------

    Input
    -----
        mode : str
            String describing which mode is used, supported modes are:
                CCS:   CCS_params_file(string) NumberOfSamples(int) DFT.db(string)")
                CCS+Q: CCS_params_file(string) NumberOfSamples(int) DFT.db(string) charge_dict(string) charge_scaling(bool)")
                DFTB:  CCS_params_file(string) NumberOfSamples(int) DFT.db(string) DFTB.db(string)")

    Returns
    -------
        What does it return Jolla?

    Example usage
    -------------
        ccs_validate MODE [...]
    """
    if os.path.isfile(CCS_DB):
        os.remove(CCS_DB)

    DFT_DB = db.connect(DFT_DB)
    CCS_DB = db.connect(CCS_DB)

    if mode == "DFTB":
        DFTB_DB = db.connect(DFTB_DB)

    if isinstance(charge_dict, str):
        charge_dict = json.loads(charge_dict)

    if isinstance(CCS_params, str):
        with open(CCS_params, "r") as f:
            CCS_params = json.load(f)

    if charge_dict is None:
        charge = False
    else:
        charge = True

    f = open("CCS_validate.dat", "w")
    print("{:^13s} {:^13s} {:^13s} {:^13s} {:^13s}".format("#Reference", "Predicted", "Error", "No_of_atoms", "structure_no"), file=f)

    if include_forces:
        f_force = open("CCS_validate_forces.dat", "w")
        print("{:^13s} {:^13s} {:^13s} {:^13s}".format("#Reference", "Predicted", "Error", "structure_no"), file=f_force)

    CCS_calc = CCS(CCS_params=CCS_params, charge=charge,
               q=charge_dict, charge_scaling=charge_scaling)

    calc = LinearCombinationCalculator([CCS_calc], [1])

    if Ns == 'all':
        Ns = -1  # CONVERT TO INTEGER INPUT FORMAT

    if Ns > 0:
        Ns = int(Ns)
        mask = [a <= Ns for a in range(len(DFT_DB))]
        random.shuffle(mask)
    else:
        mask = len(DFT_DB) * [True]

    counter = 0
    for row in tqdm(DFT_DB.select(), total=len(DFT_DB), colour="#800000"):
        if mask[counter]:
            structure = row.toatoms()
            EDFT = structure.get_potential_energy()
            if include_forces:
                FREF = structure.get_forces()
            EREF = EDFT
            structure.calc = calc
            ECCS = structure.get_potential_energy()
            if include_forces:
                FCCS = structure.get_forces()
            if mode == "DFTB":
                key = row.key
                EDFTB = DFTB_DB.get("key=" + str(key)).energy
                EREF = EDFT - EDFTB
                if include_forces:
                    FDFTB= DFTB_DB.get("key=" + str(key)).forces
                    FREF = [FREF[i]-FDFTB[i] for i in range(len(FREF))]
                sp_calculator = SinglePointCalculator(
                    structure, energy=EDFTB + ECCS)
                structure.calc = sp_calculator
                structure.get_potential_energy()
                
            print('{:13.8f} {:13.8f} {:13.8f} {:13d} {:13d}'.format(
                    EREF, ECCS, np.abs(EREF - ECCS),
                    len(structure), counter), file=f)
            if include_forces:
                FREF = [item for sublist in FREF for item in sublist]
                FCCS = [item for sublist in FCCS for item in sublist]
                for force_id, force_ref in enumerate(FREF):
                    print('{:13.8f} {:13.8f} {:13.8f} {:13d}'.format(
                        force_ref, FCCS[force_id], np.abs(force_ref - FCCS[force_id]), counter), file=f_force)

        #     try:
        #         CCS_DB.write(structure, key=key)
        #         print("key found")
        #     except:
        #         print(counter)
        #         print(structure.get_potential_energy())
        #         print(structure.get_forces())
        #         print(dir(structure))
        #         # print(structure.results["energy"])
        #         print(" ")
        #         CCS_DB.write(structure)
            CCS_DB.write(structure)

        counter += 1


def main():
    import argparse

    try:
        size = os.get_terminal_size()
        c = size.columns
        txt = "-"*c
        print("")
        print(txt)
        import art
        txt = art.text2art('CCS:Validate')
        print(txt)
    except:
        pass

    parser = argparse.ArgumentParser(description='CCS fetching tool')
    parser.add_argument("-m", "--mode",         type=str, metavar="",
                        default='CCS',  help="Mode. Availble option: CCS, CCS+Q, DFTB")
    parser.add_argument("-d", "--DFT_DB", type=str, metavar="",
                        default='DFT.db',  help="Name of DFT reference data-base")
    parser.add_argument("-dc", "--CCS_DB", type=str, metavar="",
                        default='CCS_validate.db',  help="Name of data-base to store results")
    parser.add_argument("-p", "--CCS_params", type=str, metavar="",
                        default='CCS_params.json',  help="CCS_params.json file")
    parser.add_argument("-n", "--Ns",  type=int,  metavar="",
                        default=-1,  help="Number of structures to include")
    parser.add_argument("-v", "--verbose",
                        action="store_true", help="Verbose output")
    parser.add_argument("-chg", "--charge_dict",      type=json.loads, metavar="",
                        help="Specify atomic charges in json format, e.g.: \n \'{ \"Zn\" : 2.0 , \"O\" : -2.0 }\'  ")
    parser.add_argument("-chg_s", "--charge_scaling",
                        type=bool, metavar="", default=False)
    parser.add_argument("-f", "--include_forces", type=bool, metavar="",
                        default=False, help="Validation of the reproduced forces w.r.t. those they were fitted on.")

    args = parser.parse_args()

    ccs_validate(**vars(args))


if __name__ == "__main__":
    main()
