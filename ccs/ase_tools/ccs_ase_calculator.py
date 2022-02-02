
import logging
import numpy as np
import itertools as it

from numpy import linalg as LA
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import PropertyNotImplementedError
#from pymatgen import Lattice, Structure
#from pymatgen.analysis import ewald
#from pymatgen.io.ase import AseAtomsAdaptor
from ase.constraints import full_3x3_to_voigt_6_stress


logging.basicConfig(filename='ccs.spl', level=logging.DEBUG)
logg = logging.getLogger(__name__)


class spline_table():
    def __init__(self, elem1, elem2, CCS_params, exp=True):
        self.elem1 = elem1
        self.elem2 = elem2
        self.no_pair = False
        try:
            pair = elem1+'-'+elem2
            self.rcut = CCS_params['Two_body'][pair]['r_cut']
        except:
            try:
                pair = elem2+'-'+elem1
                self.rcut = CCS_params['Two_body'][pair]['r_cut']
            except:
                self.rcut = 0.0
                self.no_pair = True
        if(self.no_pair):
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
            self.Rmin = CCS_params['Two_body'][pair]['r_min']
            self.Rcut = CCS_params['Two_body'][pair]['r_cut']
            self.a = CCS_params['Two_body'][pair]['spl_a']
            self.b = CCS_params['Two_body'][pair]['spl_b']
            self.c = CCS_params['Two_body'][pair]['spl_c']
            self.d = CCS_params['Two_body'][pair]['spl_d']
            self.aa = CCS_params['Two_body'][pair]['exp_a']
            self.bb = CCS_params['Two_body'][pair]['exp_b']
            self.cc = CCS_params['Two_body'][pair]['exp_c']
            self.exp = False
            self.x = CCS_params['Two_body'][pair]['r']
            self.dx = CCS_params['Two_body'][pair]['dr']

    def eval_energy(self, r):
        index = int(np.floor((r - self.Rmin) / self.dx))

        if r >= self.Rmin and r <= self.rcut:
            dr = r - self.x[index]
            f0 = self.a[index] + dr * \
                (self.b[index] + dr*(self.c[index] + (self.d[index]*dr)))
            return float(f0)
        elif r < self.Rmin:
            val = np.exp(-self.aa*r+self.bb) + self.cc
            return val
        else:
            val = 0.0
            return val

    def eval_force(self, r):

        index = int(np.floor((r - self.Rmin) / self.dx))

        if r >= self.Rmin and r <= self.rcut:
            dr = r - self.x[index]
            f1 = self.b[index] + dr * \
                (2*self.c[index] + (3*self.d[index] * dr))
            return f1
        elif r < self.Rmin:
            val = self.aa * np.exp(-self.aa*r+self.bb)
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
    struct = Structure(lattice, atoms.get_chemical_symbols(),
                       coords, site_properties={"charge": atoms.charges})
    Ew = ewald.EwaldSummation(struct, compute_forces=True)
    return Ew


class CCS(Calculator):
    implemented_properties = {'stress', 'energy', 'forces'}

    def __init__(self, CCS_params=None, **kwargs):
        self.rc = 7.0
        self.exp = None
        self.charge = None
        self.species = None
        self.pair = None
        self.q = None
        self.CCS_params = CCS_params
        self.eps = CCS_params['One_body']
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.species = list(set(self.atoms.get_chemical_symbols()))
        self.pair = dict()
        for a, b in it.product(self.species, self.species):
            self.pair[a+b] = spline_table(a, b, self.CCS_params)

        natoms = len(self.atoms)
        if 'numbers' in system_changes:
            self.nl = NeighborList(
                [self.rc / 2] * natoms, bothways=True, self_interaction=False)

        self.nl.update(self.atoms)

        positions = self.atoms.positions
        cell = self.atoms.cell
        energy = 0.0
        energy_eps = 0.0
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))

        for at in range(natoms):
            elem1 = self.atoms.get_chemical_symbols()[at]
            if self.eps is not None:
                try:
                    energy_eps = energy_eps + self.eps[elem1]
                except:
                    pass
            indices, offsets = self.nl.get_neighbors(at)
            f_tot = np.zeros((1, 3))
            s_tot = np.zeros((3, 3))
            for i, offset in zip(indices, offsets):
                elem2 = self.atoms.get_chemical_symbols()[i]
                d_vector = positions[i] + np.dot(offset, cell) - positions[at]
                d = np.linalg.norm(d_vector)
                energy += self.pair[elem1+elem2].eval_energy(d)
                # check this
                f = self.pair[elem1+elem2].eval_force(d)*(d_vector/d)
                s_tot += 0.5 * np.outer(f, d_vector)
                f_tot += f
            forces[at] = f_tot
            stresses[at] = s_tot
        energy = 0.5*energy  # Only bothways true

        # IF WE HAVE A LATTICE DEFINE STRESS
        if self.atoms.number_of_lattice_vectors == 3:
            stresses = full_3x3_to_voigt_6_stress(stresses)
            self.results['stress'] = (
                stresses.sum(axis=0) / self.atoms.get_volume()
            )
            self.results['stresses'] = stresses / self.atoms.get_volume()

        energy = energy + energy_eps
        if self.charge:
            ewa = ew(self.atoms, self.q)
            energy_ccs = energy
            energy = energy + ewa.total_energy
            forces_ccs = forces
            forces = forces + ewa.forces

        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces
