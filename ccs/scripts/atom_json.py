#------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2021  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
#------------------------------------------------------------------------------#

'''
Functionality to generate structures.json files, containing
structural information regarding the dataset to fit.
'''


import os
import argparse
import json
import itertools as it
from collections import OrderedDict, defaultdict
from ase import io

from ccs.common.io import read_detailedout, get_paths_from_file
from ccs.common.neighborlist import pair_dist
from ccs.common.math.ewald import ewald_summation


USAGE = \
'''
A tool to convert OUTCAR (VASP output) information to structures.json files.
'''


def main(cmdlineargs=None):
    '''Main driver routine for atom_json.

    Args:

        cmdlineargs: list of command line arguments
            When None, arguments in sys.argv are parsed (default: None)

    '''

    args = parse_cmdline_args(cmdlineargs)
    atom_json(args)


def parse_cmdline_args(cmdlineargs=None):
    '''Parses command line arguments.

    Args:

        cmdlineargs: list of command line arguments
            When None, arguments in sys.argv are parsed (default: None)

    '''

    parser = argparse.ArgumentParser(description=USAGE)

    msg = 'File containing paths to DFT calculations'
    parser.add_argument('inpaths_dft', type=str, help=msg)

    msg = 'File containing paths to DFTB calculations'
    parser.add_argument('inpaths_dftb', type=str, help=msg)

    msg = "Filename of the json file to write (default: 'structures.json')."
    parser.add_argument('-o', dest='outpath', type=str,
                        default='structures.json', help=msg)

    msg = 'Cutoff to build neighbor list for (default: 3.0Å)'
    parser.add_argument('-r', dest='rcut', type=float, help=msg)

    # msg = 'Energy term to extract from DFTB+ output'
    # parser.add_argument('-d', dest='ene', choices=['Elec', 'Rep', 'Tene'],
    #                     type=str, nargs='?', help=msg)

    args = parser.parse_args(cmdlineargs)

    return args


def atom_json(args):
    '''Reads desired output files and generates a structures.json file.

    Args:

        args: namespace of command line arguments

    '''

    inpaths_dft = get_paths_from_file(args.inpaths_dft)
    inpaths_dftb = get_paths_from_file(args.inpaths_dftb)

    paths = zip(inpaths_dft, inpaths_dftb)

    dd = OrderedDict()

    for counter, filename in enumerate(paths):
        struct = io.read(os.path.join(filename[0], 'OUTCAR'))
        tmp = OrderedDict()
        dict_species = defaultdict(int)

        for elem in struct.get_chemical_symbols():
            dict_species[elem] += 1

        atom_pair = it.combinations_with_replacement(dict_species.keys(), 2)
        tmp['energy_dft'] = struct.get_potential_energy()
        tmp['ewald'] = ewald_summation(struct)
        tmp['energy_dftb'] = read_detailedout(os.path.join(filename[1],
                                                           'detailed.out'))
        tmp['atoms'] = dict_species

        for (xx, yy) in atom_pair:
            tmp[str(xx) + '-' + str(yy)] = pair_dist(struct, args.rcut, xx, yy)

        dd['S' + str(counter + 1)] = tmp

    with open(args.outpath, 'w') as fid:
        json.dump(dd, fid, indent=8)
