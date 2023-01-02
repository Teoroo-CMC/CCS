#------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2023  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
#------------------------------------------------------------------------------#

'''
Common Ewald summation tools used by the CCS project.
'''


from pymatgen import Lattice, Structure
from pymatgen.analysis import ewald


def ewald_summation(atoms):
    '''Calculates the Ewald summation of multiple structures.

    Args:

        atoms (list): list of ASE Atoms objects

    '''

    lattice = Lattice(atoms.get_cell())
    coords = atoms.get_scaled_positions()
    struct = Structure(lattice, atoms.get_chemical_symbols(), coords)
    struct.add_oxidation_state_by_guess()
    ew = ewald.EwaldSummation(struct, compute_forces=False)

    return ew.total_energy
