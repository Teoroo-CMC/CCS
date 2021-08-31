#------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2021  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
#------------------------------------------------------------------------------#


'''Parses the inputs used by the ccs fitting script.'''


import json
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

from ccs.fitting.objective import Objective
from ccs.fitting.spline_functions import Twobody
from ccs.fitting.spline_functions import write_as_nxy
from ccs.data.conversion import Bohr__AA, eV__Hartree

logger = logging.getLogger(__name__)


def twp_fit(filename):
    '''Parses the input files and fits the reference data.

    Args:

        filename (str): The input file (input.json).

    '''

    gen_params = {'interface': None,
                  'spline': None,
                  'ctype': None,
                  'scan': False,
                  'smooth': False}

    try:
        with open(filename) as json_file:
            data = json.load(json_file, object_pairs_hook=OrderedDict)
    except FileNotFoundError:
        logger.critical(' input.json file missing')
        raise
    except ValueError:
        logger.critical('Input file not in json format')
        raise
    try:
        with open(data['Reference']) as json_file:
            struct_data = json.load(json_file, object_pairs_hook=OrderedDict)
    except FileNotFoundError:
        logger.critical(' Reference file with paiwise distances missing')
        raise
    except ValueError:
        logger.critical('Reference file not in json format')
        raise

    try:
        gen_params.update(data['General'])
    except KeyError:
        raise

    # Read the input.json file and structure file to see the keys are matching
    atom_pairs = []
    ref_energies = []
    dftb_energies = []
    ewald_energies = []
    nn = []
    # Loop over different species
    counter1 = 0
    for atmpair, values in data['Twobody'].items():
        counter1 = counter1 + 1
        logger.info('\n The atom pair is : %s' % (atmpair))
        list_dist = []
        for snum, vv in struct_data.items():
            try:
                list_dist.append(vv[atmpair])
            except KeyError:
                logger.critical(
                    'Name mismatch in input.json and structures.json')
                list_dist.append([0])

            if counter1 == 1:
                try:
                    ref_energies.append(vv['energy_dft'])
                    nn.append(
                        min(vv[atmpair])
                    )
                except KeyError:
                    logger.critical(' Check Energy key in structure file')
                    raise
                if gen_params['interface'] == 'DFTB':
                    try:
                        dftb_energies.append(vv['energy_dftb']['Elec'])
                    except KeyError:
                        logger.debug('Structure with no key Elec at %s', snum)
                        raise
                elif gen_params['interface'] == 'CCS+Q':
                    try:
                        ewald_energies.append(vv['ewald'])
                    except KeyError:
                        logger.debug('Struture with no ewald key at %s', snum)
                        raise
        if counter1 == 1:
            if gen_params['interface'] == 'DFTB':
                assert len(ref_energies) == len(dftb_energies)
                columns = ['DFT(eV)', 'DFTB(eV)', 'delta(Hartree)']
                energies = np.vstack(
                    (np.asarray(ref_energies), np.asarray(dftb_energies))
                )
                # convert energies from eV to Hartree
                ref_energies = (energies[0] - energies[1]) * eV__Hartree
                write_as_nxy(
                    'Train_energy.dat',
                    'The input energies',
                    np.vstack((energies, ref_energies)),
                    columns,
                )

        logger.info('\nThe minimum distance for atom pair %s is %s '
                    %(atmpair, min(list_dist)))
        dist_mat = pd.DataFrame(list_dist)
        dist_mat = dist_mat.fillna(0)
        dist_mat.to_csv(atmpair + '.dat', sep=' ', index=False)
        dist_mat = dist_mat.values
        logger.debug('Distance matrix for %s is \n %s ' % (atmpair, dist_mat))
        atom_pairs.append(
            Twobody(atmpair, dist_mat, len(struct_data), **values))

    sto = np.zeros((len(struct_data), len(data['Onebody'])))
    for i, key in enumerate(data['Onebody']):
        count = 0
        for _, vv in struct_data.items():
            try:
                sto[count][i] = vv['atoms'][key]
            except KeyError:
                sto[count][i] = 0
            count = count + 1
    np.savetxt('sto.dat', sto, fmt='%i')
    assert sto.shape[1] == np.linalg.matrix_rank(sto), \
        'Linear dependence in stochiometry matrix'

    if gen_params['scan']:
        mse_list = []
        mse_atom = []
        min_rcut = values['rcut']
        max_rcut = 2.7 * min_rcut
        rcuts = np.linspace(min_rcut, max_rcut, 20, endpoint=True)
        for rcut in rcuts:
            pair = []
            values['rcut'] = rcut
            nn = Objective(pair, sto, ref_energies)
            predicted_energies, mse = nn.solution()
            mse_list.append(mse)
            header = ['NN', 'DFT(H)', 'DFTB_elec(H)', 'delta', 'DFTB(H)']
            write_as_nxy('Energy.dat', 'Full energies',
                         np.vstack((
                             np.asarray(nn) * Bohr__AA,
                             energies[0] * eV__Hartree,
                             energies[1] * eV__Hartree,
                             ref_energies,
                             energies[1] * eV__Hartree + predicted_energies)),
                         header)
            err_atom = (ref_energies - predicted_energies) / np.ravel(sto)
            mse_atom.append(np.sum(np.square(err_atom)) / len(ref_energies))
        mse_arr = np.array(mse_list)
        rcuts_arr = np.array(rcuts)
        np.savetxt('new_RcutvsMse.dat',
                   np.c_[rcuts_arr, mse_arr, np.array(mse_atom)], newline='\n')

    else:
        nn = Objective(atom_pairs, sto, ref_energies, gen_params,
                       ewald=ewald_energies)
        predicted_energies, mse = nn.solution()
