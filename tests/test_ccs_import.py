#------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2022  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
#------------------------------------------------------------------------------#

import ccs

def test_import():
    print(ccs.__path__)
    print(dir(ccs))
    assert ccs.__version__, "CCS not successfully imported!"