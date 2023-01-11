# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2023  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
# ------------------------------------------------------------------------------#

import logging
import numpy as np
import itertools as it
from collections import OrderedDict, defaultdict
from numpy import linalg as LA
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import full_3x3_to_voigt_6_stress

try:
    from pymatgen.core import Lattice, Structure
    from pymatgen.analysis import ewald
except:
    pass

logging.basicConfig(filename="ccs.spl", level=logging.DEBUG)
logg = logging.getLogger(__name__)


class spline_table:
    def __init__(self, elem1, elem2, CCS_params, exp=True):
        self.elem1 = elem1
        self.elem2 = elem2
        self.no_pair = False
        try:
            pair = elem1 + "-" + elem2
            self.rcut = CCS_params["Two_body"][pair]["r_cut"]
        except:
            try:
                pair = elem2 + "-" + elem1
                self.rcut = CCS_params["Two_body"][pair]["r_cut"]
            except:
                self.rcut = 0.0
                self.no_pair = True
        if self.no_pair:
            self.a = [0.0]
            self.b = [0.0]
            self.c = [0.0]
            self.d = [0.0]
            self.aa = 0.0
            self.bb = 0.0
            self.cc = 0.0
            self.Rmin = 0.0
            self.Rcut = 0.0
            self.dx = 1.0
            self.x = [0.0]
            self.exp = False
        else:
            self.Rmin = CCS_params["Two_body"][pair]["r_min"]
            self.Rcut = CCS_params["Two_body"][pair]["r_cut"]
            self.a = CCS_params["Two_body"][pair]["spl_a"]
            self.b = CCS_params["Two_body"][pair]["spl_b"]
            self.c = CCS_params["Two_body"][pair]["spl_c"]
            self.d = CCS_params["Two_body"][pair]["spl_d"]
            self.aa = CCS_params["Two_body"][pair]["exp_a"]
            self.bb = CCS_params["Two_body"][pair]["exp_b"]
            self.cc = CCS_params["Two_body"][pair]["exp_c"]
            self.exp = False
            self.x = CCS_params["Two_body"][pair]["r"]
            self.dx = CCS_params["Two_body"][pair]["dr"]

    def eval_energy(self, r):
        index = int(np.floor((r - self.Rmin) / self.dx))

        if r >= self.Rmin and r <= self.rcut:
            dr = r - self.x[index]
            f0 = self.a[index] + dr * (
                self.b[index] + dr * (self.c[index] + (self.d[index] * dr))
            )
            return float(f0)
        elif r < self.Rmin:
            val = np.exp(-self.aa * r + self.bb) + self.cc
            return val
        else:
            val = 0.0
            return val

    def eval_force(self, r):

        index = int(np.floor((r - self.Rmin) / self.dx))

        if r >= self.Rmin and r <= self.rcut:
            dr = r - self.x[index]
            f1 = self.b[index] + dr * (2 * self.c[index] + (3 * self.d[index] * dr))
            return f1
        elif r < self.Rmin:
            val = -self.aa * np.exp(-self.aa * r + self.bb)
            return val
        else:
            val = 0.0
            return val


def ew(atoms, q):

    #   structure = AseAtomsAdaptor.get_structure(atoms)
    atoms.charges = []
    for a in atoms.get_chemical_symbols():
        atoms.charges.append(q[a])
    lattice = Lattice(atoms.get_cell())
    coords = atoms.get_scaled_positions()
    struct = Structure(
        lattice,
        atoms.get_chemical_symbols(),
        coords,
        site_properties={"charge": atoms.charges},
    )
    Ew = ewald.EwaldSummation(struct, compute_forces=True)
    return Ew


class CCS(Calculator):
    """
    CCS calculator

    Curvature constrained splines calculator compatible with the ASE
    format.

    Parameters
    ----------
    input_file : XXX
        To be added, Jolla.

    Returns
    -------
    What does it return actually, Jolla?


    Examples
    --------
    >>> To be added, Jolla.
    """

    implemented_properties = {"stress", "energy", "forces"}

    def __init__(
        self, CCS_params=None, charge=None, q=None, charge_scaling=False, **kwargs
    ):
        self.rc = 7.0  # SET THIS MAX OF ANY PAIR
        self.exp = None
        self.charge = charge
        self.species = None
        self.pair = None
        self.q = q
        self.CCS_params = CCS_params
        self.eps = CCS_params["One_body"]
        if charge_scaling:
            for key in self.q:
                self.q[key] *= self.CCS_params["Charge scaling factor"]

        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.species = list(set(self.atoms.get_chemical_symbols()))
        self.pair = dict()
        for a, b in it.product(self.species, self.species):
            self.pair[a + b] = spline_table(a, b, self.CCS_params)

        if self.atoms.number_of_lattice_vectors == 3:
            cell = atoms.get_cell()
            n_repeat = self.rc * np.linalg.norm(np.linalg.inv(cell), axis=0)
            n_repeat = np.ceil(n_repeat).astype(int)
            offsets = [*it.product(*[np.arange(-n, n + 1) for n in n_repeat])]

        natoms = len(self.atoms)
        dict_species = defaultdict(int)
        for elem in self.atoms.get_chemical_symbols():
            dict_species[elem] += 1

        energy = 0.0
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))

        # ONE-BODY ENERGY
        elems = it.combinations_with_replacement(dict_species.keys(), 1)
        for elem in elems:
            try:
                energy += self.eps[elem[0]] * dict_species[elem[0]]
            except:
                pass

        # PAIR-WISE ENERGY AND FORCE
        for x, y in it.product(self.species, self.species):
            xy_distances = []
            mask1 = [atom == x for atom in self.atoms.get_chemical_symbols()]
            mask2 = [atom == y for atom in self.atoms.get_chemical_symbols()]
            pos1 = self.atoms[mask1].positions
            index1 = np.arange(0, len(self.atoms))[mask1]
            atoms_2 = self.atoms[mask2]
            if self.atoms.number_of_lattice_vectors == 3:
                pos2 = []
                for offset in offsets:
                    pos2.append((atoms_2.positions + offset @ cell))
            else:
                pos2 = list(atoms_2.positions)
            pos2 = np.array(pos2)
            pos2 = np.reshape(pos2, (-1, 3))
            for p1, id in zip(pos1, index1):
                dist = pos2 - p1
                norm_dist = np.linalg.norm(dist, axis=1)
                dist_mask = (norm_dist < self.rc) & (norm_dist > 0)
                xy_distances.extend(norm_dist[dist_mask].tolist())
                # Sometimes there are no distances to append

                try:
                    forces[id, :] += np.sum(
                        (
                            dist[dist_mask].T
                            * list(
                                map(self.pair[x + y].eval_force, norm_dist[dist_mask])
                            )
                            / norm_dist[dist_mask]
                        ).T,
                        axis=0,
                    )
                except:
                    pass

            energy += 0.5 * sum(map(self.pair[x + y].eval_energy, xy_distances))

        if self.charge:
            ewa = ew(self.atoms, self.q)
            energy = energy + ewa.total_energy
            forces = forces + ewa.forces

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces

        if self.atoms.number_of_lattice_vectors == 3:
            stresses = full_3x3_to_voigt_6_stress(stresses)
            self.results["stress"] = stresses.sum(axis=0) / self.atoms.get_volume()
            self.results["stresses"] = stresses / self.atoms.get_volume()

        # natoms = len(self.atoms)
        # if 'numbers' in system_changes:
        #     self.nl = NeighborList(
        #         [self.rc / 2] * natoms, bothways=True, self_interaction=False)

        # self.nl.update(self.atoms)

        # positions = self.atoms.positions
        # cell = self.atoms.cell

        # for at in range(natoms):
        #     elem1 = self.atoms.get_chemical_symbols()[at]
        #     if self.eps is not None:
        #         try:
        #             energy_eps = energy_eps + self.eps[elem1]
        #         except:
        #             pass
        #     indices, offsets = self.nl.get_neighbors(at)
        #     f_tot = np.zeros((1, 3))
        #     s_tot = np.zeros((3, 3))

        #     for i, offset in zip(indices, offsets):
        #         elem2 = self.atoms.get_chemical_symbols()[i]
        #         d_vector = positions[i] + np.dot(offset, cell) - positions[at]
        #         d = np.linalg.norm(d_vector)
        #         energy += self.pair[elem1+elem2].eval_energy(d)
        #         # check this
        #         f = self.pair[elem1+elem2].eval_force(d)*(d_vector/d)
        #         s_tot += 0.5 * np.outer(f, d_vector)
        #         f_tot += f
        #     forces[at] = f_tot
        #     stresses[at] = s_tot
        # energy = 0.5*energy  # Only bothways true

        # # IF WE HAVE A LATTICE DEFINE STRESS
        # if self.atoms.number_of_lattice_vectors == 3:
        #     stresses = full_3x3_to_voigt_6_stress(stresses)
        #     self.results['stress'] = (
        #         stresses.sum(axis=0) / self.atoms.get_volume()
        #     )
        #     self.results['stresses'] = stresses / self.atoms.get_volume()

        # energy = energy + energy_eps
        # if self.charge:
        #     ewa = ew(self.atoms, self.q)
        #     energy_ccs = energy
        #     energy = energy + ewa.total_energy
        #     forces_ccs = forces
        #     forces = forces + ewa.forces
