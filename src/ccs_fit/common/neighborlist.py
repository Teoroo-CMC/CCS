# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2023  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
# ------------------------------------------------------------------------------#

"""
Common neighbor list routines used by the CCS project.
"""


import numpy as np
from ase.neighborlist import NeighborList

from ccs_fit.data.conversion import AA__Bohr


def pair_dist(atoms, rcut, ch1, ch2):
    """Calculates the pairwise distances between two types of atoms within a
       certain cuttoff.

    Args:

        atoms (list): list of ASE Atoms objects
        rcut (float): neighbor list cutoff in Angstrom
        ch1 (str): atom species 1
        ch2 (str): atoms species 2

    Returns:

        dists_rounded (list): list of distances in Bohr, i.e. atomic units

    """

    if ch1 == ch2:
        bothways = False
    else:
        bothways = True

    nl = NeighborList(
        atoms.get_global_number_of_atoms() * [rcut],
        self_interaction=False,
        bothways=bothways,
    )
    nl.update(atoms)

    distances = []

    for jj in range(atoms.get_global_number_of_atoms()):
        if atoms.get_chemical_symbols()[jj] == ch1:
            indices, offsets = nl.get_neighbors(jj)
            for ii, offset in zip(indices, offsets):
                if atoms.get_chemical_symbols()[ii] == ch2:
                    distances.append(
                        (
                            np.linalg.norm(
                                atoms.positions[ii]
                                + np.dot(offset, atoms.get_cell())
                                - atoms.positions[jj]
                            )
                        )
                    )

    distances.sort()

    # convert distances from Angstrom to Bohr and round
    dists_rounded = [round(elem * AA__Bohr, 6) for elem in distances]

    return dists_rounded
