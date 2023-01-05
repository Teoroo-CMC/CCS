# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2023  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
# ------------------------------------------------------------------------------#

"""
Common IO routines used by the CCS project.
"""


def read_detailedout(fname):
    """Reads desired energy terms from DFTB+ detailed.out files.

    Args:

        fname (str): filename to read from

    """

    tags = [
        ("Elec", "Total Electronic energy:"),
        ("Rep", "Repulsive energy"),
        ("Tene", "Total energy"),
    ]

    tag_values = {}

    with open(fname, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            words = line.split()
            for tag in tags:
                if tag[1] in line:
                    tag_values[tag[0]] = float(words[-2])

    return tag_values


def get_paths_from_file(fname):
    """Extracts a list of paths from a given file.

    Args:

        fname (str): filename to read from

    Returns:

        paths (list): list of raw paths

    """

    with open(fname, "r") as infile:
        paths = infile.readlines()

    paths = [entry.strip("\n") for entry in paths]

    return paths
