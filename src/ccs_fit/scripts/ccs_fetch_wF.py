#!/usr/bin/python3.6
import math
import sys
import json
import itertools as it
from collections import OrderedDict, defaultdict
from matplotlib.pyplot import pink
import numpy as np
from ase import Atoms
from ase import io
import ase.db as db
from sympy import true
from tqdm import tqdm
import itertools
import random
import os


def pair_dist(atoms, R_c, ch1, ch2, counter):
    """
        This function returns pairwise distances between two types of atoms within a certain cuttoff

    Input
    -----
        R_c : float
            Cut off distance(6. Å)
        ch1 : str
            Atom species 1
        ch2 : str
            Atoms species 2

    Returns
    -------
        A list of distances
    """
    try:
        cell = atoms.get_cell()
        n_repeat = R_c * np.linalg.norm(np.linalg.inv(cell), axis=0)
        n_repeat = np.ceil(n_repeat).astype(int)
        offsets = [
            *itertools.product(*[np.arange(-n, n + 1) for n in n_repeat])]
    except:
        cell = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        offsets = [[0, 0, 0]]

    mask1 = [atom == ch1 for atom in atoms.get_chemical_symbols()]
    mask2 = [atom == ch2 for atom in atoms.get_chemical_symbols()]
    pos1 = atoms[mask1].positions
    index1 = np.arange(0, len(atoms))[mask1]
    atoms_2 = atoms[mask2]
    Natoms_2 = len(atoms_2)

    pos2 = []
    for offset in offsets:
        pos2.append((atoms_2.positions + offset @ cell))

    pos2 = np.array(pos2)
    pos2 = np.reshape(pos2, (-1, 3))
    r_distance = []
    forces = OrderedDict()
    for p1, id in zip(pos1, index1):
        tmp = pos2 - p1
        norm_dist = np.linalg.norm(tmp, axis=1)
        dist_mask = norm_dist < R_c
        r_distance.extend(norm_dist[dist_mask].tolist())
        forces["F" + str(counter) + "_" + str(id)
               ] = np.asarray(tmp[dist_mask]).tolist()

    if ch1 == ch2:
        r_distance.sort()
        r_distance = r_distance[::2]

    return r_distance, forces


def ccs_fetch(
    mode=None, DFT_DB=None, R_c=6.0, Ns=-1, DFTB_DB=None, charge_dict=None,include_forces=False,verbose=False):
    """
    Function to read files and output structures.json

    Input
    -----
        args : list
            list of filenames
        R_c : float
            optional: Distance cut-off. Defaults to 7.0.

    Returns
    -------
        structures.json : JSON file
            Collection of structures in .json format.

    Example
    -------
        To be added.
    """
    DFT_DB = db.connect(DFT_DB)
    print(mode)
    
    if mode == "CCS":
        REF_DB = DFT_DB

    if mode == "CCS+Q":
        from pymatgen.core import Lattice, Structure
        from pymatgen.analysis import ewald

        REF_DB = DFT_DB

    if mode == "DFTB":
        REF_DB = db.connect(DFTB_DB)

    if Ns > 0:
        mask = [a <= Ns for a in range(len(REF_DB))]
        random.shuffle(mask)
    else:
        mask = len(REF_DB) * [True]

    species = []
    counter = -1
    c = OrderedDict()
    d = OrderedDict()
    cf = OrderedDict()
    for row in tqdm(REF_DB.select(), total=len(DFT_DB), desc="    Fetching data", colour="#008080"):
        counter = counter + 1
        if mask[counter]:
            struct = row.toatoms()
            ce = OrderedDict()
            FREF = row.forces
            EREF = row.energy
            ce["energy_dft"] = EREF
            if mode == "DFTB":
                key = str(row.key)
                EDFT = DFT_DB.get("key=" + key).energy
                FDFT = DFT_DB.get("key=" + key).forces
                ce["energy_dft"] = EDFT
                ce["energy_dftb"] = EREF
            dict_species = defaultdict(int)
            struct.charges = []
            for elem in struct.get_chemical_symbols():
                dict_species[elem] += 1
                if mode == "CCS+Q":
                    struct.charges.append(charge_dict[elem])
            atom_pair = it.combinations_with_replacement(
                dict_species.keys(), 2)
            if mode == "CCS+Q":
                lattice = Lattice(struct.get_cell())
                coords = struct.get_scaled_positions()
                ew_struct = Structure(
                    lattice,
                    struct.get_chemical_symbols(),
                    coords,
                    site_properties={"charge": struct.charges},
                )
                Ew = ewald.EwaldSummation(ew_struct,compute_forces=True)
                ES_energy = Ew.total_energy
                ES_forces = Ew.forces
                ce["ewald"] = ES_energy

            for i in range(len(struct)):
                if mode == "CCS":
                    cf["F" + str(counter) + "_" + str(i)
                       ] = {"force_dft": list(FREF[i, :])}
                if mode == "DFTB":
                    cf["F" + str(counter) + "_" + str(i)
                       ] = {"force_dft": list(FDFT[i, :]), "force_dftb": list(FREF[i, :])}
                if mode == "CCS+Q":
                    cf["F" + str(counter) + "_" + str(i)
                       ] = {"force_dft": list(FREF[i, :]), "force_ewald": list(ES_forces[i, :])}
  
            ce["atoms"] = dict_species
            for (x, y) in atom_pair:
                pair_distances, forces = pair_dist(struct, R_c, x, y, counter)
                ce[str(x) + "-" + str(y)] = pair_distances
                for i in range(len(struct)):
                    try:
                        cf["F" + str(counter) + "_" + str(i)][
                            str(x) + "-" + str(y)
                        ] = forces["F" + str(counter) + "_" + str(i)]
                    except:
                        pass
                #FORCES SHOULD BE DOUBLE COUNTED!
                if(x != y):
                    pair_distances, forces = pair_dist(struct, R_c, y, x, counter)                
                    for i in range(len(struct)):
                        try:
                            cf["F" + str(counter) + "_" + str(i)][
                                str(x) + "-" + str(y)
                            ] = forces["F" + str(counter) + "_" + str(i)]
                        except:
                            pass
                    
            d["S" + str(counter + 1)] = ce
    st = OrderedDict()
    st["energies"] = d
    if include_forces:
        st['forces'] = cf
    with open("structures.json", "w") as f:
        json.dump(st, f, indent=8)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CCS fetching tool')
    parser.add_argument("-m", "--mode",         type=str, metavar="",
                        default='CCS',  help="Mode. Availble option: CCS, CCS+Q, DFTB")
    parser.add_argument("-d", "--DFT_DB", type=str, metavar="",
                        default='DFT.db',  help="Name of DFT reference data-base")
    parser.add_argument("-dd", "--DFTB_DB", type=str, metavar="",
                        default=None,  help="Name of DFTB reference data-base")
    parser.add_argument("-r", "--R_c",    type=float, metavar="",
                        default=6.0,  help="Cut-off radius")
    parser.add_argument("-n", "--Ns",  type=int,  metavar="",
                        default=-1,  help="Number of structures to include")
    parser.add_argument("-v", "--verbose",
                        action="store_true", help="Verbose output")
    parser.add_argument("-chg", "--charge_dict",      type=json.loads, metavar="",
                        help="Specify atomic charges in json format, e.g.: \n \'{ \"Zn\" : 2.0 , \"O\" : -2.0 }\'  ")
    parser.add_argument("-f", "--include_forces",  action="store_true",
                       help='Include forces.')

    args = parser.parse_args()

    ccs_fetch(**vars(args))

    try:
        size = os.get_terminal_size()
        c = size.columns
        txt = "-"*c
        print("")
        print(txt)
        import art
        txt = art.text2art('CCS:Fetch')
        print(txt)
    except:
        pass


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
