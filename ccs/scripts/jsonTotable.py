# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2022  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
# ------------------------------------------------------------------------------#

"""
Functionality to derive pair style table lammps potential
"""

import json
import sys
import itertools
import numpy as np
from collections import OrderedDict
from ccs.ase_calculator.ccs_ase_calculator import spline_table


def asecalcTotable(jsonfile, scale=10, table="CCS.table"):
    json_file = open(jsonfile)
    CCS_params = json.load(json_file)
    energy = []
    force = []
    tags = []
    with open(table, "w") as f:
        for pair in CCS_params["Two_body"].keys():
            elem1, elem2 = pair.split("-")
            tb = spline_table(elem1, elem2, CCS_params)
            rmin = CCS_params["Two_body"][pair]["r_min"]
            dr = CCS_params["Two_body"][pair]["dr"] / scale
            r = np.arange(rmin, tb.Rcut + dr, dr)
            tags.extend((pair, len(r), rmin, tb.Rcut))
            #        onebody=0.0
            #        if CCS_params['One_body']:
            #            try:
            #                elem1_e = CCS_params['One_body'][elem1]
            #            except KeyError as err:
            #                print('Warning: Onebody energy of {} missing and set to  0.0'.format(elem1))
            #                elem1_e = 0.0
            #            try:
            #                 elem2_e = CCS_params['One_body'][elem2]
            #            except KeyError as err:
            #                print('Warning: Onebody energy of {} missing and set to  0.0'.format(elem2))
            #                elem2_e = 0.0
            #            onebody= elem1_e +elem2_e
            f.write("\n {}".format(pair))
            f.write("\n N {} R {} {} \n".format(len(r), rmin, tb.Rcut))
            [
                f.write(
                    "\n {} {} {} {}".format(
                        index + 1, elem, tb.eval_energy(elem), -1 * tb.eval_force(elem)
                    )
                )
                for index, elem in enumerate(r)
            ]

    return tags


def jsontotable(jsonfile):
    """This function converts json format to pair table lammps format"""
    json_file = open(jsonfile)
    data = json.load(json_file)
    with open(table, "w") as f:
        for key, value in data["Two_body"].items():
            r = value["r"]
            a = value["spl_a"]
            b = value["spl_b"]  # -dE/dR lammps
            f.write("\n {}".format(key))
            f.write("\n N {} R {} {} \n".format(len(r), value["r_min"], value["r_cut"]))
            [
                f.write(
                    "\n {} {} {} {}".format(
                        index + 1, r[index], a[index], -1 * b[index]
                    )
                )
                for index, elem in enumerate(value["r"])
            ]


if __name__ == "__main__":
    f = sys.argv[1]
    asecalcTotable(f)
