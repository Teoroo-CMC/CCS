import sys
import os
import numpy as np
import ase.db as db
import json
import random
from ase.calculators.singlepoint import SinglePointCalculator
from tqdm import tqdm

from ccs_fit.ase_calculator.ccs_ase_calculator import CCS


def ccs_validate(
    mode=None,
    CCS_params=None,
    Ns="all",
    DFT_DB=None,
    CCS_DB="CCS_validate.db",
    DFTB_DB=None,
    charge=False,
    q=None,
    charge_scaling=False,
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

    f = open("CCS_validate.dat", "w")
    print("#Reference      Predicted      Error      No_of_atoms structure_no", file=f)

    calc = CCS(CCS_params=CCS_params, charge=charge,
               q=q, charge_scaling=charge_scaling)

    if Ns != "all":
        Ns = int(Ns)
        mask = [a <= Ns for a in range(len(DFT_DB))]
        random.shuffle(mask)
    else:
        mask = len(DFT_DB) * [True]

    counter = -1
    for row in tqdm(DFT_DB.select(), total=len(DFT_DB), colour="#800000"):
        counter += 1
        if mask[counter]:
            structure = row.toatoms()
            EDFT = structure.get_total_energy()
            EREF = EDFT
            structure.calc = calc
            ECCS = structure.get_potential_energy()
            if mode == "DFTB":
                key = row.key
                EDFTB = DFTB_DB.get("key=" + str(key)).energy
                EREF = EDFT - EDFTB
                sp_calculator = SinglePointCalculator(
                    structure, energy=EDFTB + ECCS)
                structure.calc = sp_calculator
                structure.get_potential_energy()

            print(EREF, ECCS, np.abs(EREF - ECCS),
                  len(structure), counter, file=f)
            try:
                CCS_DB.write(structure, key=key)
            except:
                CCS_DB.write(structure)


def main():
    size = os.get_terminal_size()
    c = size.columns
    txt = "-"*c
    print("")
    print(txt)

    try:
        import art
        txt = art.text2art('CCS:Validate')
        print(txt)
    except:
        pass

    print("    USAGE:  ccs_validate MODE [...] ")
    print(" ")
    print("    The following modes and inputs are supported:")
    print("")
    print("        CCS:   CCS_params_file(string) NumberOfSamples(int) DFT.db(string)")
    print(
        "       CCS+Q: CCS_params_file(string) NumberOfSamples(int) DFT.db(string) charge_dict(string) charge_scaling(bool)"
    )
    print(
        "        DFTB:  CCS_params_file(string) NumberOfSamples(int) DFT.db(string) DFTB.db(string)"
    )
    print("")

    try:
        assert sys.argv[1] in ["CCS", "CCS+Q", "DFTB"], "Mode not supproted."
    except:
        exit()

    mode = sys.argv[1]
    CCS_params_file = sys.argv[2]
    Ns = sys.argv[3]
    DFT_data = sys.argv[4]
    with open(CCS_params_file, "r") as f:
        CCS_params = json.load(f)

    print("    Mode: ", mode)
    if mode == "CCS":
        print("    Number of samples: ", Ns)
        print("    DFT reference data base: ", DFT_data)
        print("")

        ccs_validate(mode=mode, CCS_params=CCS_params, Ns=Ns, DFT_DB=DFT_data)
    if mode == "DFTB":
        DFTB_data = sys.argv[5]
        print("    Number of samples: ", Ns)
        print("    DFT reference data base: ", DFT_data)
        print("    DFTB reference data base: ", DFTB_data)
        print("")

        ccs_validate(
            mode=mode, CCS_params=CCS_params, Ns=Ns, DFT_DB=DFT_data, DFTB_DB=DFTB_data
        )
    if mode == "CCS+Q":
        print(
            "        NOTE: charge_dict should use double quotes to enclose property nanes. Example:"
        )
        print('        \'{"Zn":2.0,"O" : -2.0 } \'')
        charge_dict = sys.argv[5]
        charge_scaling = sys.argv[6]
        if charge_scaling == "True":
            charge_scaling = True
        if charge_scaling == "False":
            charge_scaling = False
        print("    Number of samples: ", Ns)
        print("    DFT reference data base: ", DFT_data)
        print("    Charge dictionary: ", charge_dict)
        print("    Charge scaling: ", charge_scaling)
        print("")

        charge_dict = json.loads(charge_dict)
        ccs_validate(
            mode=mode,
            CCS_params=CCS_params,
            Ns=Ns,
            DFT_DB=DFT_data,
            charge=True,
            q=charge_dict,
            charge_scaling=charge_scaling,
        )


if __name__ == "__main__":
    main()
