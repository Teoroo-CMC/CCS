#------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2021  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
#------------------------------------------------------------------------------#

# read version from installed package
from importlib.metadata import version
__version__ = version("ccs")

from ccs.ase_calculator import CCS