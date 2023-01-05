# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2023  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
# ------------------------------------------------------------------------------#

# read version from installed package
from importlib.metadata import version

#__version__ = version("ccs")

# from ccs.ase_calculator.ccs_ase_calculator import CCS
# from ccs.scripts.ccs_build_db import main as ccs_build_db
# from ccs.scripts.ccs_export_sktable import main as ccs_export_sktable
from ccs_fit.scripts.ccs_fetch import ccs_fetch as ccs_fetch
from ccs_fit.fitting.main import twp_fit as ccs_fit
# from ccs.scripts.ccs_validate import main as ccs_validate
