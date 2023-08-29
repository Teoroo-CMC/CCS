#!/usr/bin/python
import ase.db as db
import re
from ase.io import Trajectory, read, write
from ase.calculators.neighborlist import *
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np

# from pymatgen import Lattice, Structure
# from pymatgen.analysis import ewald
from ase.visualize import view

# from fortformat import Fnetdata


##############
# USER INPUT #
VASP_FOLDER = "VASP"

R_c = 6.0


pairs = [["Na", "Na"], ["Na", "Cl"], ["Cl", "Cl"]]

one_body = ["Ce", "Pr", "O"]

DFT_DB = db.connect("DFT.db")

f = open("list", "r")
line_count = 0
for line in f:
    if line != "\n":
        line_count += 1
f.close


f = open("list", "r")
counter = 0
for lns in f:
    lns = lns.replace("\n", "")
    counter = counter + 1
    print("   ----- READING STRUCUTURE " + lns + " -----\n")

    # READ VASP ENERGY
    structure_DFT = read(lns)
    EDFT = structure_DFT.get_potential_energy()

    DFT_DB.write(structure_DFT, PBE=True, key=counter)
