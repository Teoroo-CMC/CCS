# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2023  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
# ------------------------------------------------------------------------------#

"""
Functionality to derive spline table for LAMMPS
"""

import json
import sys
import itertools
import numpy as np
from collections import OrderedDict
from ccs_fit.ase_calculator.ccs_ase_calculator import spline_table


def asecalcTotable(jsonfile, scale=10, table="CCS.table"):
    json_file = open(jsonfile)
    CCS_params = json.load(json_file)
    energy = []
    force = []
    tags = {}
    with open(table, "w") as f:
        for pair in CCS_params["Two_body"].keys():
            elem1, elem2 = pair.split("-")
            tb = spline_table(elem1, elem2, CCS_params)
            rmin=CCS_params["Two_body"][pair]["r_min"]
            dr = CCS_params["Two_body"][pair]["dr"] / scale
            r = np.arange(rmin, tb.Rcut + dr, dr)
            tags[pair]=dict({'Rmin':rmin,'Rcut':tb.Rcut,'dr':dr,'N':len(r)})
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



if __name__ == "__main__":
    f = sys.argv[1]
    asecalcTotable(f)
