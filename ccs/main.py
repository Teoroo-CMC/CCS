import json
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

from ccs.objective import Objective
from ccs.spline_functions import Twobody

logger = logging.getLogger(__name__)


def twp_fit(filename):

    try:
        with open(filename) as json_file:
            data = json.load(json_file, object_pairs_hook=OrderedDict)
    except FileNotFoundError:
        logger.critical(" input.json file missing")
        raise
    except ValueError:
        logger.critical("Input file not in json format")
        raise
    try:
        with open(data['Reference']) as json_file:
            struct_data = json.load(json_file, object_pairs_hook=OrderedDict)
    except FileNotFoundError:
        logger.critical(" Reference file with paiwise distances missing")
        raise
    except ValueError:
        logger.critical("Reference file not in json format")
        raise

# Reading the input.json file and structure file to see the keys are matching
    atom_pairs = []
    ref_energies = []
    # Loop over different species
    for atmpair, values in data['Twobody'].items():
        logger.debug("\n The atom pair is : %s" % (atmpair))
        list_dist = []
        for snum, v in struct_data.items():  # loop over structures
            try:
                list_dist.append(v[atmpair])
            except KeyError:
                logger.critical(
                    " Name mismatch in input.json and structures.json")
                raise
            try:
                ref_energies.append(v['Energy'])
            except KeyError:
                logger.critical(" Check Energy key in structure file")
                raise

        dist_mat = pd.DataFrame(list_dist)
        dist_mat = dist_mat.values
        logger.debug(" Distance matrix for %s is \n %s " % (atmpair, dist_mat))
        atom_pairs.append(
            Twobody(atmpair, dist_mat, len(struct_data), **values))

    sto = np.zeros((len(struct_data), len(data['Onebody'])))
    for i, key in enumerate(data['Onebody']):
        count = 0
        for k, v in struct_data.items():
            sto[count][i] = v['Atoms'][key]
            count = count+1
    n = Objective(atom_pairs, sto, ref_energies)
    n.solution()
