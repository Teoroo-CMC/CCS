# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2021  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
# ------------------------------------------------------------------------------#

# read version from installed package
from importlib.metadata import version

__version__ = version("ccs")

from ccs.ase_calculator.ccs_ase_calculator import CCS
from ccs.scripts.ccs_build_db as ccs_build_db
from ccs.scripts import ccs_export_sktable as ccs_export_sktable
from ccs.scripts import ccs_fetch as ccs_fetch
from ccs.scripts import ccs_fit as ccs_fit
from ccs.scripts import ccs_validate as ccs_validate
