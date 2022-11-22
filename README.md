# CCS - Curvature Constrained Splines  

[![Latest Release](https://img.shields.io/github/v/release/teoroo-cmc/ccs?display_name=tag&color=brightgreen&sort=semver)](https://github.com/Teoroo-CMC/CCS/releases/latest)
[![License](https://img.shields.io/github/license/teoroo-cmc/ccs)](https://opensource.org/licenses/LGPL-3.0)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.cpc.2020.107602-blue)](https://doi.org/10.1016/j.cpc.2020.107602)
[![Build](https://img.shields.io/github/workflow/status/teoroo-cmc/CCS/ci-cd)](https://github.com/Teoroo-CMC/CCS/actions)

<!--- [![Build Status](https://github.com/tblite/tblite/workflows/CI/badge.svg)](https://github.com/tblite/tblite/actions)
[![Documentation](https://img.shields.io/badge/Github%20Pages-Pages-blue)](https://teoroo-cmc.github.io/CCS/)
[![codecov](https://codecov.io/gh/tblite/tblite/branch/main/graph/badge.svg?token=JXIE6myqNH)](https://codecov.io/gh/tblite/tblite) --->

The `CCS` package is a tool to construct two-body potentials using the idea of curvature constrained splines.
## Getting Started
### Package Layout

```
ccs-x.y.z
├── CHANGELOG.md
├── LICENSE
├── MANIFEST.in
├── README.md
├── bin
│   ├── ccs_build_db
│   ├── ccs_export_sktable
│   ├── ccs_fetch
│   ├── ccs_fit
│   └── ccs_validate
├── docs
├── examples
│   ├── CCS
│   ├── CCS_with_LAMMPS
│   ├── DFTB_repulsive_fitting
│   ├── Preparing_ASE_db_trainingsets
│   ├── Simple_regressor
│   ├── Twobody_fit_for_an_O2_molecule
│   ├── Twobody_fit_for_solid_Ne
│   └── ppmd_interfacing
├── logo.png
├── poetry.lock
├── pyproject.toml
├── src
│   └── ccs
│       ├── ase_calculator
│       ├── common
│       ├── data
│       ├── debugging_tools
│       ├── fitting
│       ├── ppmd_interface
│       ├── regression_tool
│       └── scripts
│           ├── ccs_build_db.py
│           ├── ccs_export_sktable.py
│           ├── ccs_fetch.py
│           ├── ccs_fit.py
│           └── ccs_validate.py
└── tests
```

* `ccs_fetch`           - Executable to construct the traning-set (structures.json) from a pre-existing ASE-database with DFT-data.
* `ccs_fit`             - The primary executable file for the ccs package.
* `main.py`             - A module to parse input files.
* `objective.py`        - A module which contains the objective function and solver.
* `spline_functions.py` - A module for spline construction/evaluation/output. 

<!---
### Prerequisites

You need to install the following softwares

```
pip install numpy
pip install scipy
pip install ase
pip install cvxopt
```
### Installing from source

#### Git clone

```
git clone git@github.com/Teoroo-CMC/CCS.git
cd CCS
python setup.py install
```
--->
### Installing from source using poetry
```
pip install poetry
pip install git+https://github.com/Teoroo-CMC/CCS@master
```
### Environment Variables
Set the following environment variables:
```
$export PYTHONPATH=<path-to-CCS-package>:$PYTHONPATH
$export PATH=<path-to-CCS-bin>:$PATH

Within a conda virtual environment, you can update the path by using:
conda develop <path-to-CCS-package>
```


## Tutorials

We provide tutorials in the [examples](examples/) folder. To run the example, go to one of the folders. Each contain the neccesery input files required for the task at hand. A sample `CCS_input.json` for O2 is shown below:
```
{
        "General": {
                "interface": "CCS"
        },
        "Train-set": "structures.json",
        "Twobody": {
                "O-O": {
                        "Rcut": 2.5,
                        "Resolution": 0.02,
                        "Swtype": "sw"
                }
        },
        "Onebody": [
                "O"
        ]
}

```
The `CCS_input.json` file should provide at a minimum the block "Genaral" specifying an interface. The default is to look for input structures in the file `structure.json` file. The format for `structure.json` is shown below :
```
{
"energies":{
        "S1": {
                "Energy": -4.22425752,
                "Atoms": {
                        "O": 2
                },
                "O-O": [
                        0.96
                ]
        },
        "S2": {
                "Energy": -5.29665634,
                "Atoms": {
                        "O": 2
                },
                "O-O": [
                        0.98
                ]
        },
        "S3": {
                "Energy": -6.20910363,
                "Atoms": {
                        "O": 2
                },
                "O-O": [
                        1.0
                ]
        },
        "S4": {
                "Energy": -6.98075271,
                "Atoms": {
                        "O": 2
                },
                "O-O": [
                        1.02
                ]
        }
}
}
```
The `structure.json` file contains different configurations labeled ("S1", "S2"...) and corresponding energy, pairwise distances (contained in an array labelled as "O-O" for oxygen). The stoichiometry of each configuration is given under the atoms label ("Atoms") as a key-value pair ("O" : 2 ). 


To perform the fit : 
```
ccs_fit
```
The following output files are obtained:
```
CCS_params.json CCS_error.out ccs.log 
```
* CCS_params.json  - Contains the spline coefficients, and one-body terms for two body potentials.
* error.out        - Contains target energies, predicted energies and absolute error for each configuration.
* ccs.log          - Contains debug information
## Authors

* **Akshay Krishna AK** 
* **Jolla Kullgren** 
* **Eddie Wadbro** 


## License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgement
We thank all the members of  [TEOROO-group](http://www.teoroo.kemi.uu.se/) at Uppsala University, Sweden.



