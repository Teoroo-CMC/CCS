import json
import logging
from collections import OrderedDict 
from ccs.objective import Twobody
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)


def twp_fit(filename):


    
    try:
        with open(filename) as json_file:
            data = json.load(json_file)
            data = OrderedDict(data)
    except FileNotFoundError:
        logger.critical(" input.json file missing")
        raise
    except ValueError:
        logger.critical("Input file not in json format")
        raise
    try:
        with open(data['Reference']) as json_file:
                struct_data=json.load(json_file)
                struct_data=OrderedDict(struct_data)
    except FileNotFoundError:
        logger.critical(" Reference file with paiwise distances missing")
        raise
    except ValueError:        
        logger.critical("Reference file not in json format")
        raise

    atom_pairs = []
    for key, values in data['Twobody'].items():
        # Append dismat to the dictionary
        print(key)
        list_dist= []
        for k,v in struct_data.items(): # loop over structures 
            list_dist.append(v[key])
        dist_mat = pd.DataFrame(list_dist)
        dist_mat=dist_mat.values
        logger.debug(" Distance matrix for %s is \n %s " %(key,dist_mat))
        atom_pairs.append(Twobody(key,dist_mat,len(struct_data),**values))

     
    
        
    
    
