# CCS - Curvature Constrained Splines  

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
│   ├── ccs_build_db
│   ├── ccs_export_sktable
│   ├── ccs_fetch
│   ├── ccs_fit
│   └── ccs_validate
├── docs
├── examples
│   ├── CCS
│   ├── DFTB_repulsive_fitting
│   ├── Preparing_ASE_db_trainingsets
│   ├── Simple_regressor
│   ├── Twobody_fit_for_an_O2_molecule
│   ├── Twobody_fit_for_solid_Ne
│   └── ppmd_interfacing
├── logo.png
├── poetry.lock
├── pyproject.toml
├── src
│   └── ccs
│       ├── ase_calculator
│       │   └── ccs_ase_calculator.py
│       ├── common
│       │   ├── exceptions.py
│       │   ├── io.py
│       │   ├── math
│       │   │   └── ewald.py
│       │   └── neighborlist.py
│       ├── data
│       │   └── conversion.py
│       ├── debugging_tools
│       │   └── timing.py
│       ├── fitting
│       │   ├── main.py
│       │   ├── objective.py
│       │   └── spline_functions.py
│       ├── ppmd_interface
│       │   └── ccs_ppmd.py
│       ├── regression_tool
│       │   └── regression.py
│       └── scripts
│           ├── ccs_build_db.py
│           ├── ccs_export_sktable.py
│           ├── ccs_fetch.py
│           ├── ccs_fit.py
│           └── ccs_validate.py
└── tests
    └── test_ccs_import.py
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
### Installing from source
```
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



