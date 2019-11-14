import numpy as np
import sys
import json
import itertools as it
from collections import OrderedDict,defaultdict
from ase import Atoms
from ase import io
from ase.calculators.neighborlist  import *  # This should be changed


def pair_dist(atoms,R_c, ch1, ch2):
    ''' This function returns pairwise distances between two types of atoms within a certain cuttoff
    Args:
        R_c (float): Cut off distance(3.5 Å)
        ch1 (str): Atom species 1
        ch2 (str): Atoms species 2

    Returns:
        A list of distances in bohr
    '''
    if ch1==ch2:

        nl = NeighborList(atoms.get_number_of_atoms() *
                        [R_c], self_interaction=False, bothways=False)
    else:
        nl = NeighborList(atoms.get_number_of_atoms() *
                        [R_c], self_interaction=False, bothways=True)


    nl.update(atoms)
    distances = []
    for j in range(atoms.get_number_of_atoms()):
        if (atoms.get_chemical_symbols()[j] == ch1):
            indices, offsets = nl.get_neighbors(j)
            for i, offset in zip(indices, offsets):
                if(atoms.get_chemical_symbols()[i] == ch2):  # ONLY TO OXYGEN
                    distances.append((np.linalg.norm(
                        atoms.positions[i] + np.dot(offset, atoms.get_cell()) - atoms.positions[j])))
    
    distances.sort()
    r_distances = [round(elem*1.88973, 6)
                   for elem in distances]  # distances in bohr
    return r_distances

def main(R_c,*args):
    c = OrderedDict()
    d = OrderedDict()
    species= []
    for counter,filename in enumerate(args):
        struct = io.read(filename)        
        dict_species = defaultdict(int)
        for elem in struct.get_chemical_symbols():
            dict_species[elem]+=1    
        atom_pair=it.combinations_with_replacement(dict_species.keys(),2)
        c['Atoms']= dict_species
        for (x,y) in atom_pair:
            pair_distances = pair_dist(struct,R_c,x,y)
            c[str(x)+str(y)]= pair_distances
        d['Structure '+str(counter+1)] = c
    with open('structures.json','w') as f:
        json.dump(d,f,indent=8)
                    
                
            

    
main(3.5,*sys.argv[1:])
