import json
import logging
from ccs.objective import Twobody
logger = logging.getLogger(__name__)


def twp_fit(filename):


    
    try:
        with open(filename) as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        logger.critical(" input.json file missing")
        raise
    except ValueError:
        logger.critical("Input file not in json format")
        raise
    try:
        with open(data['Reference']) as json_file:
                struct_data=json.load(json_file)
    except FileNotFoundError:
        logger.critical(" Reference file with paiwise distances missing")
        raise
    except ValueError:
        logger.critical("Reference file not in json format")
        raise
    for key,values in struct_data.items():
        print (len(struct_data))

    
    atom_pairs = []
    for key, values in data['Twobody'].items():
        # Append dismat to the dictionary
        print(key)
        for key1,values1 in struct_data.items():
            print(values1[key])
        atom_pairs.append(Twobody(key, **values))
    
        
    
    
