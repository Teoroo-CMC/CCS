#!/usr/bin/python3.6
import math
import sys
import json
import itertools as it
from collections import OrderedDict, defaultdict
import numpy as np
from ase import Atoms
from ase import io
import ase.db as db
from sympy import true
from tqdm import tqdm
import itertools
import random


def pair_dist(atoms, R_c, ch1, ch2, counter):
    ''' This function returns pairwise distances between two types of atoms within a certain cuttoff
    Args:
        R_c (float): Cut off distance(6. Å)
        ch1 (str): Atom species 1
        ch2 (str): Atoms species 2

    Returns:
        A list of distances
    '''
    cell = atoms.get_cell()
    n_repeat = R_c * np.linalg.norm(np.linalg.inv(cell), axis=0)
    n_repeat = np.ceil(n_repeat).astype(int)

    mask1 = [atom == ch1 for atom in atoms.get_chemical_symbols()]
    mask2 = [atom == ch2 for atom in atoms.get_chemical_symbols()]
    pos1 = atoms[mask1].positions
    index1 = np.arange(0, len(atoms))[mask1]
    atoms_2 = atoms[mask2]
    Natoms_2 = len(atoms_2)

    offsets = [*itertools.product(*[np.arange(-n, n+1)
                                  for n in n_repeat])]

    pos2 = []
    i = -1
    for offset in offsets:
        i += 1

        pos2.append((atoms_2.positions+offset@cell))

    pos2 = np.array(pos2)
    pos2 = np.reshape(pos2, (-1, 3))
    r_distance = []
    forces = OrderedDict()
    for p1, id in zip(pos1, index1):
        tmp = pos2-p1
        norm_dist = np.linalg.norm(tmp, axis=1)
        dist_mask = norm_dist < R_c
        r_distance.extend(norm_dist[dist_mask].tolist())
        forces["F"+str(counter)+"_"+str(id)
               ] = np.asarray(tmp[dist_mask]).tolist()

    if(ch1 == ch2):
        r_distance.sort()
        r_distance = r_distance[::2]

    return r_distance, forces


def main():
    """  Function to read files and output structures.json

    Args:
        args(list): list of filenames
        R_c (float, optional): Distance cut-off. Defaults to 7.0.
    """

    print("--- USAGE:  DB2TRAIN.py MODE Rc NumberOfSamples DFT.db [...] --- ")
    print(" ")

    mode = sys.argv[1]
    print("    MODE", mode)
    if(mode == "CCS"):
        R_c = float(sys.argv[2])

        print("    R_c set to: ", R_c)
        DFT_data = sys.argv[4]
        print("    DFT reference data base: ", DFT_data)
        print("-------------------------------------------------")

    DFT_DB = db.connect(DFT_data)
    if(sys.argv[3] == "all"):
        pass
    else:
        Ns = int(sys.argv[3])

    mask = [a <= Ns for a in range(len(DFT_DB))]
    random.shuffle(mask)

    species = []
    counter = -1
    c = OrderedDict()
    d = OrderedDict()
    for row in tqdm(DFT_DB.select()):
        counter = counter + 1
        if(mask[counter]):
            FDFT = row.forces
            EDFT = row.energy
            struct = row.toatoms()

            dict_species = defaultdict(int)
            for elem in struct.get_chemical_symbols():
                dict_species[elem] += 1
            atom_pair = it.combinations_with_replacement(
                dict_species.keys(), 2)
            cf = OrderedDict()
            for i in range(len(struct)):
                cf["F"+str(counter)+"_"+str(i)
                   ] = {'force_dft':  list(FDFT[i, :])}
            ce = OrderedDict()
            ce['energy_dft'] = EDFT
            ce['atoms'] = dict_species
            for (x, y) in atom_pair:
                pair_distances, forces = pair_dist(struct, R_c, x, y, counter)
                ce[str(x)+'-'+str(y)] = pair_distances
                for i in range(len(struct)):
                    try:
                        cf["F"+str(counter)+"_"+str(i)][str(x)+'-'+str(y)
                                                        ] = forces["F"+str(counter)+"_"+str(i)]
                    except:
                        pass
            d['S'+str(counter+1)] = ce
    st = OrderedDict()
    st['energies'] = d
    #st['forces'] = cf
    with open('structures.json', 'w') as f:
        json.dump(st, f, indent=8)


main()
