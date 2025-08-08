from pathlib import Path
import json
import sys
import itertools as it
from collections import OrderedDict, defaultdict
from ase.db.core import Atoms
import numpy as np
import ase.db as db
from ase.cell import Cell
from tqdm import tqdm
import itertools
import random
import os
from ase.constraints import voigt_6_to_full_3x3_stress

from ccs_fit.scripts.helper import terminal_header


def pair_dist(atoms: Atoms, R_c: float, ch1: str, ch2: str, counter: int):
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
    if atoms.get_pbc().all() == False:
        cell = Cell([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        offsets = [[0, 0, 0]]
    elif atoms.get_pbc().all() == True:
        cell = atoms.get_cell()
        n_repeat = R_c * np.linalg.norm(np.linalg.inv(cell), axis=0)
        n_repeat = np.ceil(n_repeat).astype(int)
        offsets = [
            *itertools.product(*[np.arange(-n, n + 1) for n in n_repeat])
        ]
        atoms.wrap()
    else:
        print("Error: PBC not set properly. Only non-periodic or 3D periodic systems are supported.")
        sys.exit(1)



    mask1 = [atom == ch1 for atom in atoms.get_chemical_symbols()]
    mask2 = [atom == ch2 for atom in atoms.get_chemical_symbols()]
    pos1 = atoms[mask1].positions
    index1 = np.arange(0, len(atoms))[mask1]
    atoms_2 = atoms[mask2]

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
        forces["F" + str(counter) + "_" + str(id)] = np.asarray(
            tmp[dist_mask]
        ).tolist()

    if ch1 == ch2:
        r_distance.sort()
        r_distance = r_distance[::2]

    return r_distance, forces


def ccs_fetch(
    mode=None,
    DFT_DB=None,
    R_c=6.0,
    Ns="all",
    DFTB_DB=None,
    charge_dict=None,
    include_forces=False,
    include_stresses=False,
    write_json=True,
    q_type="pymatgen",
    read_q=False,
    range_separated=None,
):
    """
    Function to read ASE database and calculate pairwise distances between atoms of different species.

    Input
    -----
        mode : string
            To what target the spline should be fitted. Options are [CCS,CCS+Q,DFTB]
        DFT_DB : string
            optional: target database
        R_c : float
            optional: Distance cut-off. Defaults to 6.0.

    Returns
    -------
        structures dictionary : dict
            Collection of structures with pairwise distances between atoms of different species.

    Example
    -------
        None available.
    """

    if mode not in {"CCS", "CCS+Q", "CCS+fQ", "CCS2Q","DFTB","CCS+iQ"}:
        raise ValueError(f"Invalid mode: {mode}. Choose from 'CCS', 'CCS+Q', 'CCS+fQ', 'CCS2Q' , 'DFTB', 'CCS+iQ'.")

    file_path = Path(DFT_DB)
    if not file_path.exists():
        print(f"Error: The file '{file_path}' does not exist.")
        sys.exit(1)
    
    DFT_DB = db.connect(DFT_DB)

    if mode == "CCS":
        REF_DB = DFT_DB

    if mode == "CCS+Q" or mode == "CCS+fQ" or mode == "CCS2Q" or mode == "CCS+iQ":
        if q_type == "pymatgen":
            from pymatgen.core import Lattice, Structure
            from pymatgen.analysis.ewald import EwaldSummation
            from scipy.special import erfcinv
            class RecipEwaldSummation(EwaldSummation):
                def _calc_ewald_terms(self):
                    """Calculate and sets all Ewald terms (point, real and reciprocal), saving separate forces."""
                    self._recip, self._recip_forces = self._calc_recip()
                    self._real, self._point, real_point_forces = self._calc_real_and_point()
                    self._real_point_forces = real_point_forces
                    if self._compute_forces:
                        self._forces = self._recip_forces + self._real_point_forces

        if q_type == "lammps":    
            from ase.calculators.lammpsrun import LAMMPS
            # LAMMPS potential parameters
            lammps_parameters = {
                "atom_style": "charge",
                "pair_style": "coul/long 12.0",  # Coulombic interactions with cutoff
                "pair_coeff": ["* *"],           # Default coefficients for charged particles
                "kspace_style": "ewald 1.0e-12", # Long-range electrostatics 
            }
            ES_calc = LAMMPS()
            ES_calc.set(**lammps_parameters)
        REF_DB = DFT_DB

    if mode == "DFTB":
        REF_DB = db.connect(DFTB_DB)

    if mode == "PRUNED_CCS":
        REF_DB = db.connect(DFTB_DB)

    if Ns == "all":
        Ns = -1  # CONVERT TO INTEGER INPUT FORMAT

    if Ns > 0:
        mask = [a <= Ns - 1 for a in range(len(REF_DB))]
        random.shuffle(mask)
    else:
        mask = len(REF_DB) * [True]

    counter = -1
    d  = OrderedDict()
    ds = OrderedDict()
    cf = OrderedDict()

    for row in tqdm(
        REF_DB.select(),
        total=len(DFT_DB),
        desc="    Fetching data",
        colour="#008080",
    ):
        counter = counter + 1
        if mask[counter]:
            EREF=None  
            FREF=None
            SREF=None
          
            struct = row.toatoms()
            ce = OrderedDict()
            cs = OrderedDict()
            try:    
                EREF = row.energy
            except KeyError:
                raise KeyError(f"Energy key for configuration {row.id} not found.")
            ce["energy_dft"]=EREF
            if include_forces:
                try:
                    FREF = row.forces
                except:
                    pass
            if include_stresses:
                try:
                    SREF= ( voigt_6_to_full_3x3_stress(row.stress)).tolist()
                except:
                    pass
            if mode == "DFTB":
                EDFT = DFT_DB.get("id=" + str(row.id)).energy
                if include_forces and FREF is not None:
                    FDFT = DFT_DB.get("id=" + str(row.id)).forces
                if include_stresses and SREF is not None:
                    SDFT = (   voigt_6_to_full_3x3_stress( DFT_DB.get("id=" + str(row.id)).stress)).tolist()
                ce["energy_dft"] = EDFT
                ce["energy_dftb"] = EREF
            dict_species = defaultdict(int)
            struct.charges = []
            charges=[]
            for elem in struct.get_chemical_symbols():
                dict_species[elem] += 1
                if mode == "CCS+Q" or mode == "CCS+fQ" or mode == "CCS2Q" or mode == "CCS+iQ":
                    if read_q == False:
                        try:
                            charge_dict[elem]
                        except KeyError:
                            raise KeyError(f"Missing charge information for element {elem}.")
                            sys.exit()
                        charges.append(charge_dict[elem])
            if read_q:
                charges=row.data['mulliken']

            dict_species = {
                key: value for key, value in sorted(dict_species.items())
            }
            atom_pair = it.combinations_with_replacement(dict_species.keys(), 2)
            if mode == "CCS+Q" or mode == "CCS+fQ" or mode == "CCS2Q" or mode == "CCS+iQ":
                if q_type == "pymatgen":
                    struct.charges = charges
                    lattice = Lattice(struct.get_cell())
                    coords = struct.get_scaled_positions()
                    ew_struct = Structure(
                        lattice,
                        struct.get_chemical_symbols(),
                        coords,
                        site_properties={"charge": struct.charges},
                    )
                    if range_separated == None:
                        Ew = RecipEwaldSummation(ew_struct, compute_forces=True)
                    elif range_separated.lower() == "damped":
                        eta_value = erfcinv(1E-7*R_c) / (R_c) ;  eta_value = eta_value**2
                        Ew =RecipEwaldSummation(ew_struct, compute_forces=True,real_space_cut=R_c,eta=eta_value)
                    elif range_separated.lower() == "stitch":
                        Ew = RecipEwaldSummation(ew_struct, compute_forces=True)

                    if range_separated == None:
                        ce["ewald"] = Ew.total_energy
                    elif range_separated.lower() == "damped":
                        ce["ewald"] = Ew.reciprocal_space_energy+Ew.point_energy # Shall point be here!
                    elif range_separated.lower() == "stitch":
                        ce["ewald"] = Ew.total_energy


                    if include_forces and FREF is not None:
                        if range_separated == None:
                            ES_forces = Ew.forces
                        elif range_separated.lower() == "damped":
                            ES_forces = Ew._recip_forces
                        elif range_separated.lower() == "stitch":
                            ES_forces = Ew.forces

                    if include_stresses:
                        SREF=None
                        print("Stresses not supported in pymatgen Ewald routine.")
                if q_type == "lammps":
                    if range_separated == None:
                        pass
                    elif range_separated.lower() == "damped":
                        print("RangeSeparation Damped is not implemented with lammps")

                    ES_struct=struct.copy()
                    ES_struct.set_initial_charges(charges)
                    ES_struct.calc=ES_calc
                    ce["ewald"] = ES_struct.get_potential_energy()
                    if include_forces and FREF is not None:
                        ES_forces = ES_struct.get_forces()
                    if include_stresses and SREF is not None:
                        ES_stress= ( voigt_6_to_full_3x3_stress(  ES_struct.get_stress()) ).tolist()

            if include_forces and FREF is not None:
                for i in range(len(struct)):
                    if mode == "CCS":
                        cf["F" + str(counter) + "_" + str(i)] = {
                            "force_dft": list(FREF[i, :])
                        }
                    if mode == "DFTB":
                        cf["F" + str(counter) + "_" + str(i)] = {
                            "force_dft": list(FDFT[i, :]),
                            "force_dftb": list(FREF[i, :]),
                        }
                    if mode == "CCS+Q" or mode == "CCS+fQ" or mode == "CCS2Q" or mode == "CCS+iQ":
                        cf["F" + str(counter) + "_" + str(i)] = {
                            "force_dft": list(FREF[i, :]),
                            "force_ewald": list(ES_forces[i, :]),
                        }

            ce["atoms"]  = dict_species

            if include_stresses and SREF is not None:
                cs["volume"] = struct.get_volume()
                if mode == "CCS":
                    cs["stress_dft"] = SREF
                if mode == "CCS+Q" or mode == "CCS+iQ":
                    cs["stress_dft"] = SREF
                    cs["stress_ewald"]=ES_stress
                if mode == "DFTB":
                    cs["stress_dftb"]= SREF
                    cs["stress_dft"] = SDFT

            for x, y in atom_pair:
                pair_distances, forces = pair_dist(struct, R_c, x, y, counter)
                ce[str(x) + "-" + str(y)] = pair_distances
                if include_stresses and SREF is not None:
                    cs[str(x) + "-" + str(y)]=[]
                for i in range(len(struct)):
                    if include_forces and FREF is not None:
                        try:
                            cf["F" + str(counter) + "_" + str(i)][
                                str(x) + "-" + str(y)
                            ] = forces["F" + str(counter) + "_" + str(i)]
                        except:
                            pass
                    if include_stresses and SREF is not None:
                        try:
                            cs[str(x) + "-" + str(y)].extend( forces["F" + str(counter) + "_" + str(i)])
                        except:
                            pass
                # FORCES SHOULD BE DOUBLE COUNTED!
                if x != y:
                    pair_distances, forces = pair_dist(
                        struct, R_c, y, x, counter
                    )
                    for i in range(len(struct)):
                        if include_forces and FREF is not None:
                            try:
                                cf["F" + str(counter) + "_" + str(i)][
                                    str(x) + "-" + str(y)
                                ] = forces["F" + str(counter) + "_" + str(i)]
                            except:
                                pass
                        if include_stresses and SREF is not None:
                            try:
                                cs[str(x) + "-" + str(y)].extend( forces["F" + str(counter) + "_" + str(i)])
                            except:
                                pass

            d["S" + str(counter + 1)] = ce
            if include_stresses and SREF is not None:
                ds["S" + str(counter + 1)] = cs
            
    st = OrderedDict()
    st["energies"] = d
    if include_forces:
        st["forces"] = cf
    if include_stresses:
        st["stresses"] = ds
    if write_json:    
        with open("structures.json", "w") as f:
            json.dump(st, f, indent=8)
    else:
        return st

def main():
    import argparse

    terminal_header("C3S : Fetch")

    parser = argparse.ArgumentParser(description="CCS fetching tool")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        metavar="",
        default="CCS",
        help="Mode. Available options: CCS, CCS+Q, CCS+fQ, CCS2Q, DFTB, CCS+iQ",
    )
    parser.add_argument(
        "-d",
        "--DFT_DB",
        type=str,
        metavar="",
        default="DFT.db",
        help="Name of DFT reference database",
    )
    parser.add_argument(
        "-dd",
        "--DFTB_DB",
        type=str,
        metavar="",
        default=None,
        help="Name of DFTB reference database",
    )
    parser.add_argument(
        "-r",
        "--R_c",
        type=float,
        metavar="",
        default=6.0,
        help="Cut-off radius",
    )
    parser.add_argument(
        "-n",
        "--Ns",
        type=int,
        metavar="",
        default=-1,
        help="Number of structures to include.",
    )
    parser.add_argument(
        "-chg",
        "--charge_dict",
        type=json.loads,
        metavar="",
        help='Specify atomic charges in json format, e.g.: \n \'{ "Zn" : 2.0 , "O" : -2.0 }\'  ',
    )
    parser.add_argument(
        "-f", "--include_forces", action="store_true", help="Include forces."
    )
    parser.add_argument(
        "-s", "--include_stresses", action="store_true", help="Include stresses."
    )
    parser.add_argument(
        "-rq", "--read_q", action="store_true", help="Read charges."
    )
    parser.add_argument(
        "-rs",
        "--range_separated",
        type=str,
        metavar="",
        default=None,
        help="Range separated charges. Available options: stitch, damped.",
    )

    args = parser.parse_args()

    ccs_fetch(**vars(args))



if __name__ == "__main__":
    main()
