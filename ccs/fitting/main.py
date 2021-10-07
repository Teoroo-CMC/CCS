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
from ccs.fitting.spline_functions import Onebody
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
                  'smooth': False,
                  'ewald_scaling' : 1.0
                 }   

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
                except KeyError:
                    logger.critical(' Check Energy key in structure file')
                    raise
                if 'DFTB' in gen_params['interface']:
                    try:
                        dftb_energies.append(vv['energy_dftb'])
                    except KeyError:
                        logger.debug('Structure with no key Elec at %s', snum)
                        raise
                if gen_params['interface'] == 'CCS+Q' or gen_params['interface'] == 'CCS2Q' or gen_params['interface'] == 'CCS+fQ':
                    try:
                        ewald_energies.append(vv['ewald'])
                    except KeyError:
                        logger.debug('Struture with no ewald key at %s', snum)
                        raise
    
        if counter1 == 1:
            if gen_params['interface'] == 'DFTB+':
                assert len(ref_energies) == len(dftb_energies)
                columns = ['DFT(eV)', 'DFTB(eV)', 'delta(Hartree)']
                energies = np.vstack(
                    (np.asarray(ref_energies), np.asarray(dftb_energies))
                )
                # convert energies from eV to Hartree
                ref_energies = (energies[0] - energies[1])  * eV__Hartree
                write_as_nxy(
                    'Train_energy.dat',
                    'The input energies',
                    np.vstack((energies, ref_energies)),
                    columns,
                )
                list_dist=[[element /Bohr__AA  for element in elements ] for elements in list_dist] 

            if gen_params['interface'] == 'CCS2Q':
                assert len(ref_energies) == len(ewald_energies)
                columns = ['DFT(eV)', 'Ewald(eV)', 'delta(eV)']
                energies = np.vstack(
                    (np.asarray(ref_energies), np.asarray(ewald_energies))
                )
                ref_energies = (energies[1])  

            if gen_params['interface'] == 'CCS+fQ':
                assert len(ref_energies) == len(ewald_energies)
                columns = ['DFT(eV)', 'Ewald(eV)', 'delta(eV)']
                energies = np.vstack(
                    (np.asarray(ref_energies), np.asarray(ewald_energies))
                )
                ref_energies = (energies[0] - gen_params['ewald_scaling']*energies[1] )  



        try:
           values['Rmin']
        except:
           values['Rmin']=min(min(list_dist))           

        logger.info('\nThe minimum distance for atom pair %s is %s '
                    %(atmpair, min(list_dist)))
        dist_mat = pd.DataFrame(list_dist)
        dist_mat = dist_mat.fillna(0)
        dist_mat.to_csv(atmpair + '.dat', sep=' ', index=False)
        dist_mat = dist_mat.values
        logger.debug('Distance matrix for %s is \n %s ' % (atmpair, dist_mat))
        atom_pairs.append(
            Twobody(atmpair, dist_mat, len(struct_data), **values))


    atom_onebodies=[]
    sto = np.zeros((len(struct_data), len(data['Onebody'])))
    for i, key in enumerate(data['Onebody']):
        atom_onebodies.append( Onebody( key )  )
        count = 0
        for _, vv in struct_data.items():
            # print(vv['atoms'][key] )
            try:
                sto[count][i] = vv['atoms'][key]
            except KeyError:
                sto[count][i] = 0
            count = count + 1
    #REDUCE STO-MATRIX IF RANK IS TOO LOW
    reduce=True
    n_redundant=0
    while(reduce):
        check=0
        for ci in range(np.shape(sto)[1]):
            if( np.linalg.matrix_rank(sto[:,0:ci+1]) <  (ci+1) ):
               print('There is linear dependence in stochiometry matrix!')
               print('    removing onebody term: ' + atom_onebodies[ci+n_redundant].name)
               sto=np.delete(sto,ci,1)
               atom_onebodies[ci+n_redundant].epsilon_supported=False
               check=1
               n_redundant+=1
               break
        if (check == 0):
           reduce=False

    assert sto.shape[1] == np.linalg.matrix_rank(sto), \
        'Linear dependence in stochiometry matrix'
   


    nn = Objective(atom_pairs,atom_onebodies, sto, ref_energies, gen_params,
                       ewald=ewald_energies)
    predicted_energies, mse = nn.solution()
