# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2023  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
# ------------------------------------------------------------------------------#


"""Parses the inputs used by the ccs fitting script."""


import json
import copy
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

from ccs_fit.fitting.objective import Objective
from ccs_fit.fitting.spline_functions import Twobody
from ccs_fit.fitting.spline_functions import Onebody
from ccs_fit.data.conversion import Bohr__AA, eV__Hartree
from ccs_fit.debugging_tools.timing import timing

logger = logging.getLogger(__name__)


def prepare_input(filename):

    gen_params = {"interface": None, "ewald_scaling": 1.0, 'merging': 'False'}
    struct_data_test = {}

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
        gen_params.update(data["General"])
        data["General"] = gen_params
    except KeyError:
        raise
    try:
        gen_data = {
            "General": gen_params,
            "Train-set": "structures.json",
            "Test-set": "structures.json",
        }
        gen_data.update(data)
        data = gen_data
    except:
        raise

    try:
        with open(data["Train-set"]) as json_file:
            struct_data_full = json.load(
                json_file, object_pairs_hook=OrderedDict)
            struct_data = struct_data_full["energies"]
            try:
                struct_data_forces = struct_data_full["forces"]
            except:
                struct_data_forces = {}
    except FileNotFoundError:
        logger.critical(" Reference file with pairwise distances missing")
        raise
    except ValueError:
        logger.critical("Reference file not in json format")
        raise
    if "Test-set" not in data:
        data["Test-set"] = data["Train-set"]
    try:
        with open(data["Test-set"]) as json_file:
            struct_data_test_full = json.load(
                json_file, object_pairs_hook=OrderedDict)
            struct_data_test = struct_data_test_full["energies"]
            try:
                struct_data_test_forces = struct_data_test_full["forces"]
            except:
                struct_data_test_forces = {}
    except FileNotFoundError:
        logger.info("Could not locate Test-set.")

    # Make defaults or general setting for Twobody
    if "Twobody" not in data.keys():
        if "DFTB" in data["General"]["interface"]:
            data["Twobody"] = {
                "Xx-Xx": {"Rcut": 5.0, "Resolution": 0.1, "Swtype": "rep", "const_type": "Mono"}
            }
        if "CCS" in data["General"]["interface"]:
            data["Twobody"] = {
                "Xx-Xx": {"Rcut": 8.0, "Resolution": 0.1, "Swtype": "sw", "const_type": "Mono"}
            }

    # If onebody is not given it is generated from structures.json
    elements = set()
    [elements.add(key) for _, vv in struct_data.items()
     for key in vv["atoms"].keys()]
    elements = sorted(list(elements))
    try:
        data["Onebody"]
    except:
        print("    Generating one-body information from training-set.")
        print("        Added elements: ", elements)
        logger.info("Generating one-body information from training-set.")
        logger.info(f"    Added elements: {elements}")
        # list is now redundant here, but kept for future reference
        data["Onebody"] = list(elements)

    if "Xx-Xx" in data["Twobody"]:
        print("    Generating two-body potentials from one-body information.")
        logger.info("Generating two-body potentials from one-body information.")

    for atom_i in data["Onebody"]:
        for atom_j in data["Onebody"]:
            if (
                (atom_j >= atom_i)
                and (atom_i + "-" + atom_j not in data["Twobody"])
                and (atom_j + "-" + atom_i not in data["Twobody"])
                and ("Xx-" + atom_i not in data["Twobody"])
                and ("Xx-" + atom_j not in data["Twobody"])
            ):
                try:
                    data["Twobody"][atom_i + "-" + atom_j] = copy.deepcopy(
                        data["Twobody"]["Xx-Xx"]
                    )
                    print("    Adding pair: " + atom_i + "-" + atom_j)
                    logger.info(f"Adding pair: {atom_i}-{atom_j}")
                except:
                    pass
            if (
                ("Xx-" + atom_i in data["Twobody"])
                and (atom_i + "-" + atom_j not in data["Twobody"])
                and (atom_j + "-" + atom_i not in data["Twobody"])
            ):
                data["Twobody"][atom_i + "-" + atom_j] = copy.deepcopy(
                    data["Twobody"]["Xx-" + atom_i]
                )
            if (
                ("Xx-" + atom_j in data["Twobody"])
                and (atom_i + "-" + atom_j not in data["Twobody"])
                and (atom_j + "-" + atom_i not in data["Twobody"])
            ):
                data["Twobody"][atom_i + "-" + atom_j] = copy.deepcopy(
                    data["Twobody"]["Xx-" + atom_j]
                )

    tmp_data = copy.deepcopy(data)
    for dat in tmp_data["Twobody"]:
        if "Xx" in dat:
            del data["Twobody"][dat]
    del tmp_data

    return (
        data,
        struct_data,
        struct_data_test,
        struct_data_forces,
        struct_data_test_forces,
    )


# @timing
def parse(data, struct_data, struct_data_forces):

    atom_pairs = []
    ref_energies = []
    dftb_energies = []
    ewald_energies = []
    counter1 = 0
    ref_forces = []
    dftb_forces = []
    ewald_forces = []

    # ADD ENERGY-DATA
    for atmpair, values in data["Twobody"].items():
        atmpair_members = atmpair.split("-")
        atmpair_rev = atmpair_members[1] + "-" + atmpair_members[0]

        counter1 = counter1 + 1
        # logger.info("\n The atom pair is : %s" % (atmpair))
        list_dist = []
        for snum, vv in struct_data.items():
            try:
                list_dist.append(vv[atmpair])
            except KeyError:
                try:
                    list_dist.append(vv[atmpair_rev])
                except KeyError:
                    logger.critical(
                        "Name mismatch in CCS_input.json and structures.json"
                    )
                    list_dist.append([0])

            if counter1 == 1:
                try:
                    ref_energies.append(vv["energy_dft"])
                except KeyError:
                    logger.critical(" Check Energy key in structure file")
                    raise
                if "DFTB" in data["General"]["interface"]:
                    try:
                        dftb_energies.append(vv["energy_dftb"])
                    except KeyError:
                        logger.debug(
                            "Structure with no key energy_dftb at %s", snum)
                        raise
                if "Q" in data["General"]["interface"]:
                    try:
                        ewald_energies.append(vv["ewald"])
                    except KeyError:
                        logger.debug("Struture with no ewald key at %s", snum)
                        raise

        if counter1 == 1:
            if "DFTB" in data["General"]["interface"]:
                assert len(ref_energies) == len(dftb_energies)
                energies = np.vstack(
                    (np.asarray(ref_energies), np.asarray(dftb_energies))
                )
                ref_energies = energies[0] - energies[1]

            if data["General"]["interface"] == "CCS2Q":
                assert len(ref_energies) == len(ewald_energies)
                columns = ["DFT(eV)", "Ewald(eV)", "delta(eV)"]
                energies = np.vstack(
                    (np.asarray(ref_energies), np.asarray(ewald_energies))
                )
                ref_energies = energies[1]

            if data["General"]["interface"] == "CCS+fQ":
                assert len(ref_energies) == len(ewald_energies)
                columns = ["DFT(eV)", "Ewald(eV)", "delta(eV)"]
                energies = np.vstack(
                    (np.asarray(ref_energies), np.asarray(ewald_energies))
                )
                ref_energies = (
                    energies[0] -
                    data["General"]["ewald_scaling"] * energies[1]
                )

        try:
            Rmax = max([item for sublist in list_dist for item in sublist])
        except:
            Rmax = 0

        if Rmax > 0:
            try:
                values["Rmin"]
            except:
                values["Rmin"] = (
                    min([item for sublist in list_dist for item in sublist if item > 0]
                        ) - 0.5 * values["Resolution"]
                    # TO MAXIMIZE NUMERICAL STABILITY INNERMOST POINT IS PLACED IN THE MIDLE OF THE FIRST INTERVAL

                )

            if values["Rcut"] > Rmax:
                values["Rcut"] = Rmax

            dist_mat = pd.DataFrame(list_dist)
            dist_mat = dist_mat.fillna(0)
            dist_mat = dist_mat.values

            # ADD FORCE-DATA

            list_dist_forces = []
            for fnum, ff in struct_data_forces.items():
                try:
                    list_dist_forces.append(ff[atmpair])
                except KeyError:
                    try:
                        list_dist_forces.append(ff[atmpair_rev])
                    except KeyError:
                        list_dist_forces.append([0.0, 0.0, 0.0])

                if counter1 == 1:
                    if data["General"]["interface"] == "CCS":
                        try:
                            ref_forces.append(ff["force_dft"])
                        except KeyError:
                            logger.critical(
                                " Check force key in structure file")
                            raise
                    if "DFTB" in data["General"]["interface"]:
                        try:
                            ff_tmp = np.array(
                                ff["force_dft"])-np.array(ff["force_dftb"])
                            ref_forces.append(ff_tmp)
                        except KeyError:
                            logger.critical(
                                " Check force key in structure file")
                            raise
                    if data["General"]["interface"] == "CCS+Q":
                        try:
                            ref_forces.append(ff["force_dft"])
                            ewald_forces.append(ff["force_ewald"])
                        except KeyError:
                            logger.critical(
                                " Check force key in structure file")
                            raise
                    if data["General"]["interface"] == "CCS+fQ":
                        try:
                            ff_tmp = np.array(
                                ff["force_dft"]) - data["General"]["ewald_scaling"]*np.array(ff["force_ewald"])
                            ref_forces.append(ff_tmp)
                        except KeyError:
                            logger.critical(
                                " Check force key in structure file")
                            raise

            dist_mat_forces = pd.DataFrame(list_dist_forces)
            dist_mat_forces = dist_mat_forces.fillna(0.0)
            dist_mat_forces = dist_mat_forces.values

            # APPEND DATA
            if values["Rmin"] < values["Rcut"]:
                atom_pairs.append(
                    Twobody(atmpair, dist_mat, dist_mat_forces, **values))

    # ADD ONEBODY DATA
    atom_onebodies = []
    sto = np.zeros((len(struct_data), len(data["Onebody"])))
    for i, key in enumerate(data["Onebody"]):
        count = 0
        for _, vv in struct_data.items():
            try:
                sto[count][i] = vv["atoms"][key]
            except KeyError:
                sto[count][i] = 0
            count = count + 1
        atom_onebodies.append(Onebody(key, sto[:, i].flatten()))

    with open("CCS_input_interpreted.json", "w") as f:
        json.dump(data, f, indent=8)

    return (
        atom_pairs,
        atom_onebodies,
        sto,
        ref_energies,
        ref_forces,
        ewald_energies,
        ewald_forces,
        data,
    )


def twp_fit(filename):
    """Parses the input files and fits the reference data.

    Args:

        filename (str): The input file (input.json).

    """
    # Read the input.json file and structure file to see if the keys are matching
    (
        data,
        struct_data,
        struct_data_test,
        struct_data_forces,
        struct_data_test_forces,
    ) = prepare_input(filename)
    # Parse the input
    (
        atom_pairs,
        atom_onebodies,
        sto,
        ref_energies,
        ref_forces,
        ewald_energies,
        ewald_forces,
        data,
    ) = parse(data, struct_data, struct_data_forces)

    # set up the QP problem
    nn = Objective(
        atom_pairs,
        atom_onebodies,
        sto,
        ref_energies,
        ref_forces,
        data["General"],
        energy_ewald=ewald_energies,
        force_ewald=ewald_forces,
    )

    # Solve QP problem
    predicted_energies, mse, xx_unfolded = nn.solution()
