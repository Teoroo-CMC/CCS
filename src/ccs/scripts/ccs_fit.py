# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2021  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
# ------------------------------------------------------------------------------#

"""
Functionality to fit curvature constraint splines.
"""


import logging
import argparse

from ccs.fitting.main import twp_fit

FILENAME = "CCS_input.json"


USAGE = """
A tool to fit two body potentials using constrained cubic splines.
"""


def main(cmdlineargs=None):
    """Main driver routine for ccs_fit.

    Args:

        cmdlineargs: list of command line arguments
            When None, arguments in sys.argv are parsed (default: None)

    """

    args = parse_cmdline_args(cmdlineargs)
    ccs_fit(args)


def parse_cmdline_args(cmdlineargs=None):
    """Parses command line arguments.

    Args:

        cmdlineargs: list of command line arguments
            When None, arguments in sys.argv are parsed (default: None)

    """

    parser = argparse.ArgumentParser(description=USAGE)

    msg = "Json file containing pairwise distances and energies"
    parser.add_argument("input", nargs="?", default=FILENAME, help=msg)

    msg = "Log level for debugging"
    parser.add_argument(
        "-d",
        "--debug",
        dest="loglvl",
        default=logging.INFO,
        const=logging.DEBUG,
        action="store_const",
        help=msg,
    )

    args = parser.parse_args(cmdlineargs)

    fmt = "%(asctime)s - %(name)s - %(levelname)s -       %(message)s"
    logging.basicConfig(filename="ccs.log", format=fmt, level=args.loglvl)

    return args


def ccs_fit(args):
    """Reads desired output file and generates a structures.json file.

    Args:

        args: namespace of command line arguments

    """

    logging.info("Started")
    twp_fit(args.input)
    logging.info("Ended")
