""" The  module to parse the inputs"""
import json
import logging
from collections import OrderedDict

import numpy as np

import pandas as pd

from ccs.objective import Objective
from ccs.spline_functions import Twobody
from ccs.spline_functions import write_as_nxy

logger = logging.getLogger(__name__)


def twp_fit(filename):
    """ The function parses the input files and fits the reference data.
    
    Args:
        filename (str): The input file (input.json).
    """
    gen_params = {'interface':None,'spline':None,'ctype':None,'scan':False}

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

    try:
        print(data['General'])
        gen_params.update(data['General'])
        print(gen_params)
    except KeyError:
        raise
     

# Reading the input.json file and structure file to see the keys are matching
    atom_pairs = []
    ref_energies = []
    dftb_energies = []
    NN=[]
    # Loop over different species
    counter1=0
    for atmpair, values in data['Twobody'].items():
        counter1=counter1+1
        logger.info("\n The atom pair is : %s" % (atmpair))
        list_dist = []
        for snum, v in struct_data.items():  # loop over structures
            try:
                list_dist.append(v[atmpair])
            except KeyError:
                logger.critical(
                    " Name mismatch in input.json and structures.json")
                list_dist.append([0])
                
            if(counter1 == 1):
                try:
                      ref_energies.append(v['Energy'])
                      NN.append(min(v[atmpair]))   # ALEEEEEEEEEEERRRRRRRRRRTT WONT WORK FOR MULTIPLE
                except KeyError:
                    logger.critical(" Check Energy key in structure file")
                    raise
                if(gen_params['interface'] == "DFTB"):
                    try:
                        dftb_energies.append(v["DFTB_energy"]["Elec"])
                    except KeyError:
                        logger.debug("Structure with no key Elec at %s",snum)
                        raise
        if(counter1 == 1):                
            if dftb_energies is not None:
                assert len(ref_energies) == len(dftb_energies)
                columns=["DFT(eV)","DFTB(eV)","delta(hatree)"]
                energies = np.vstack((np.asarray(ref_energies),np.asarray(dftb_energies)))
                ref_energies = (energies[0]-energies[1])*0.03674 #Energy in hatree
                write_as_nxy("Train_energy.dat","The input energies",np.vstack((energies,ref_energies)),columns)
            
  #      print(list_dist)
        logger.info("\n The minimum distance for atom pair %s is %s "%(atmpair,min(list_dist)))
        dist_mat = pd.DataFrame(list_dist)
        dist_mat= dist_mat.fillna(0)
        print(dist_mat)
        dist_mat.to_csv(atmpair+".dat",sep=" ",index=False)
        dist_mat = dist_mat.values
        logger.debug(" Distance matrix for %s is \n %s " % (atmpair, dist_mat))
        atom_pairs.append(
            Twobody(atmpair, dist_mat, len(struct_data), **values))

    sto = np.zeros((len(struct_data), len(data['Onebody'])))
    for i, key in enumerate(data['Onebody']):
        count = 0
        for k, v in struct_data.items():
            try:
                sto[count][i] = v['Atoms'][key]
            except KeyError:
                sto[count][i]=0                
            count = count+1
    np.savetxt("sto.dat",sto,fmt="%i")
    assert sto.shape[1]== np.linalg.matrix_rank(sto),"Linear dependence in stochiometry matrix"
    if gen_params["scan"]:
        mse_list = []
        mse_atom=[]
        min_Rcut = values["Rcut"]
        max_Rcut = 2.7*min_Rcut
        grid_size = (values["Rcut"] -values["Rmin"])/values["Nknots"] 
        rcuts = np.linspace(min_Rcut,max_Rcut,20,endpoint=True)
        for rcut in rcuts:
            pair=[]
            values["Rcut"] = rcut
     #       values["Nknots"]= int(round((rcut - values["Rmin"])/grid_size))
            mod_pair = pair.append(Twobody(atmpair,dist_mat,len(struct_data),**values)) 
            n = Objective(pair, sto, ref_energies)
            E_predicted,mse = n.solution()
            mse_list.append(mse)
            header= ["NN","DFT(H)","DFTB_elec(H)","delta","DFTB(H)"]
            write_as_nxy("Energy.dat","Full energies",np.vstack((np.asarray(NN)*0.52977,energies[0]*0.03674,energies[1]*0.03674,ref_energies,energies[1]*0.03674+E_predicted)),header)
            err_atom = (ref_energies - E_predicted)/np.ravel(sto)
            mse_atom.append(np.sum(np.square(err_atom))/len(ref_energies))
        mse_arr = np.array(mse_list)
        rcuts_arr = np.array(rcuts)
        print(rcuts_arr)
        print(mse_arr)
        np.savetxt("new_RcutvsMse.dat",np.c_[rcuts_arr,mse_arr,np.array(mse_atom)],newline="\n")
        
    else:
            n = Objective(atom_pairs, sto, ref_energies)
            E_predicted,mse = n.solution()
            header= ["NN","DFT(H)","DFTB_elec(H)","delta","DFTB(H)"]
            write_as_nxy("Energy.dat","Full energies",np.vstack((np.asarray(NN)*0.52977,energies[0]*0.03674,energies[1]*0.03674,ref_energies,energies[1]*0.03674+E_predicted)),header)


