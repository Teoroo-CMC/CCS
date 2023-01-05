# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2023  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
# ------------------------------------------------------------------------------#

"""
Conversion factory required by the CCS project.
"""

from scipy.constants import value, angstrom

# Bohr --> Angstrom
Bohr__AA = value('Bohr radius')/angstrom  # 0.529177249 
# Angstrom --> Bohr
AA__Bohr = 1.0 / Bohr__AA

# Hartree --> eV
Hartree__eV = value('hartree-electron volt relationship') # 27.2113845
# eV --> Hartree
eV__Hartree = 1.0 / Hartree__eV
