# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2023  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
# ------------------------------------------------------------------------------#

# import logging
import copy
import numpy as np
import itertools as it
from collections import defaultdict
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import full_3x3_to_voigt_6_stress
from ase.calculators.lammpsrun import LAMMPS
from ase.cell import Cell


try:
    from pymatgen.core import Lattice, Structure
    from pymatgen.analysis import ewald
except:
    print("Could not import pymatgen Lattice and Structure.")


class spline_table:
    def __init__(self, elem1, elem2, CCS_params):
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
            val = self.b[index] + dr * (
                2 * self.c[index] + (3 * self.d[index] * dr)
            )
            return -val
        elif r < self.Rmin:
            val = -self.aa * np.exp(-self.aa * r + self.bb)
            return -val
        else:
            val = 0.0
            return val


def ew(atoms, q,lammps=False):
    #   structure = AseAtomsAdaptor.get_structure(atoms)
    if lammps == False:
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
        return Ew.total_energy,Ew.forces,None

    if lammps == True:
        Ew={}
        ES_atoms=atoms.copy()
        charges = []
        for a in atoms.get_chemical_symbols():
            charges.append(q[a])
        ES_atoms.set_initial_charges(charges)

        # LAMMPS potential parameters
        lammps_parameters = {
            "atom_style": "charge",
            "pair_style": "coul/long 12.0",  # Coulombic interactions with cutoff
            "pair_coeff": ["* *"],           # Default coefficients for charged particles
            "kspace_style": "ewald 1.0e-12", # Long-range electrostatics using PPPM
        }
        ES_calc = LAMMPS()
        ES_calc.set(**lammps_parameters)
        ES_atoms.calc=ES_calc
        ES_energy=ES_atoms.get_potential_energy()
        ES_forces=ES_atoms.get_forces()
        ES_stress=  ES_atoms.get_stress()
        return ES_energy,ES_forces,ES_stress

class CCS(Calculator):
    """
    CCS calculator

    Curvature constrained splines calculator compatible with the ASE
    format.
    """

    implemented_properties = {"energy", "forces", "stress"}

    def __init__(
        self,
        CCS_params=None,
        q_type="pymatgen",
        range_sep=None,
        **kwargs
    ):
        self.rc = 8.0  # SET THIS MAX OF ANY PAIR
        self.exp = None
        self.species = None
        self.pair = None
        self.q_type=q_type
        self.range_sep=range_sep
        try:
            self.q = CCS_params["Charges"]
        except:
            self.q = None
        self.CCS_params = CCS_params
        self.eps = CCS_params["One_body"]
        if "Range_sep" in CCS_params:
            if CCS_params["Range_sep"]=="Stitch":
                self.range_sep="stitch"
        Calculator.__init__(self, **kwargs)

    def calculate(
        self, atoms=None, properties=["energy"], system_changes=all_changes
    ):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.species = list(set(self.atoms.get_chemical_symbols()))
        self.pair = dict()
        for a, b in it.product(self.species, self.species):
            self.pair[a + b] = spline_table(a, b, self.CCS_params)
            if self.pair[a + b].rcut > self.rc:
                self.rc = self.pair[a + b].rcut

        # if self.atoms.get_pbc().all() == True:
        #     cell = atoms.get_cell()
        #     self.atoms.wrap()
        #     n_repeat = self.rc * np.linalg.norm(np.linalg.inv(cell), axis=0)
        #     n_repeat = np.ceil(n_repeat).astype(int)
        #     offsets = [*it.product(*[np.arange(-n, n + 1) for n in n_repeat])]

        if self.atoms.get_pbc().all() == False:
            cell = Cell([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            offsets = [[0, 0, 0]]
        elif self.atoms.get_pbc().all() == True:
            cell = atoms.get_cell()
            n_repeat = self.rc * np.linalg.norm(np.linalg.inv(cell), axis=0)
            n_repeat = np.ceil(n_repeat).astype(int)
            offsets = [
                *it.product(*[np.arange(-n, n + 1) for n in n_repeat])
            ]
            self.atoms.wrap()
        else:
            print("Error: PBC not set properly. Only non-periodic or 3D periodic systems are supported.")
            sys.exit(1)


        natoms = len(self.atoms)
        dict_species = defaultdict(int)
        for elem in self.atoms.get_chemical_symbols():
            dict_species[elem] += 1

        energy = 0.0
        E_sr=0.0
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms,3,3))

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
                dist = p1 - pos2
                norm_dist = np.linalg.norm(dist, axis=1)
                dist_mask = (norm_dist <= self.pair[x+y].rcut) & (norm_dist > 0)
                xy_distances.extend(norm_dist[dist_mask].tolist())
                # Sometimes there are no distances to append
                # Force calculation
                #try:
                if len(xy_distances) >0:
                    forces[id, :] += np.sum(
                        (
                            dist[dist_mask].T
                            * list(
                                map(
                                    self.pair[x + y].eval_force,
                                    norm_dist[dist_mask],
                                )
                            )
                            / norm_dist[dist_mask]
                        ).T,
                        axis=0,
                    )
                    if self.range_sep == 'stitch':
                        sr_F= np.array(norm_dist[dist_mask]**(-2)*self.q[x]*self.q[y]/norm_dist[dist_mask])
                        sr_F=14.39964547842567*sr_F[:, np.newaxis]*dist[dist_mask]
                        forces[id, :] -= np.sum(sr_F, axis=0)
                        
                    
                #except:
                #    pass

                # Stress calculation
                id2s = [i for i, x in enumerate(dist_mask) if x]
                if norm_dist.size != 0:
                    for id2 in id2s:
                        cur_f = (
                            self.pair[x + y].eval_force(norm_dist[id2])
                            * dist[id2, :]
                            / norm_dist[id2]
                        )
                        cur_dist = dist[id2, :]
                        cur_stress = -0.5 * np.outer(cur_f, cur_dist) # IS SIGN CORRECT! 
                        stresses[id] += cur_stress
                        #RANGE SEP

            energy += 0.5 * sum(map(self.pair[x + y].eval_energy, xy_distances))
            if self.range_sep == 'stitch':
                energy -=0.5*14.39964547842567*np.sum(np.array(xy_distances)**(-1.0)) *self.q[x]*self.q[y]

        if self.q is not None:
            if self.q_type == "lammps":
                ewa_energy,ewa_forces,ewa_stress = ew(self.atoms, self.q,lammps=True)
            if self.q_type == "pymatgen":
                ewa_energy,ewa_forces,ewa_stress = ew(self.atoms, self.q)
            energy = energy + ewa_energy 
            forces = forces + ewa_forces

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces

        if self.atoms.cell.rank == 3:
            stresses = full_3x3_to_voigt_6_stress(stresses)
            if self.q is not None:
                if ewa_stress is not None:
                    self.results['stress'] = stresses.sum(axis=0) / self.atoms.get_volume()+ewa_stress
            else:        
                self.results['stress'] = stresses.sum(axis=0) / self.atoms.get_volume()
            #self.results['stresses'] = stresses / self.atoms.get_volume()

