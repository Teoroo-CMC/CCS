# CCS_fit - Fitting using Curvature Constrained Splines  

[![PyPI](https://img.shields.io/pypi/v/ccs_fit?color=g)](https://pypi.org/project/ccs-fit/)
[![License](https://img.shields.io/github/license/teoroo-cmc/ccs)](https://opensource.org/licenses/LGPL-3.0)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.cpc.2020.107602-blue)](https://doi.org/10.1016/j.cpc.2020.107602)
[![Build](https://img.shields.io/github/actions/workflow/status/teoroo-cmc/CCS/ci-cd.yml)](https://github.com/Teoroo-CMC/CCS/actions)
[![Documentation](https://img.shields.io/badge/Github%20Pages-CCS_fit-orange)](https://teoroo-cmc.github.io/CCS/)
[![Python version](https://img.shields.io/pypi/pyversions/ccs_fit)](https://pypi.org/project/ccs-fit/)

<!--- [![Build Status](https://github.com/tblite/tblite/workflows/CI/badge.svg)](https://github.com/tblite/tblite/actions)
[![Latest Release](https://img.shields.io/github/v/release/teoroo-cmc/ccs?display_name=tag&color=brightgreen&sort=semver)](https://github.com/Teoroo-CMC/CCS/releases/latest)
[![Documentation](https://img.shields.io/badge/Github%20Pages-Pages-blue)](https://teoroo-cmc.github.io/CCS/)
[![codecov](https://codecov.io/gh/tblite/tblite/branch/main/graph/badge.svg?token=JXIE6myqNH)](https://codecov.io/gh/tblite/tblite) 
[![Coverage](codecov.io/gh/:vcsName/:user/:repo?flag=flag_name&token=a1b2c3d4e5)(https://github.com/Teoroo-CMC/CCS/actions)
--->

![](logo.png)

The `CCS_fit` package is a tool to construct two-body potentials using the idea of curvature constrained splines.
## Getting Started
### Package Layout

```
ccs_fit-1.0.0
├── CHANGELOG.md
├── LICENSE
├── MANIFEST.in
├── README.md
├── bin
│   ├── ccs_build_db
│   ├── ccs_export_sktable
|   ├── ccs_export_FF
│   ├── ccs_fit
│   └── ccs_validate
├── docs
├── examples
│   └── Basic_Tutorial
│       └── tutorial.ipynb
│   └── Advanced_Tutorials
│       ├── CCS+Q
│       └── DFTB_repulsives

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
│           ├── ccs_export_FF.py
│           ├── ccs_export_sktable.py
│           ├── ccs_fetch.py
│           ├── ccs_fit.py
│           └── ccs_validate.py
└── tests
```

* `ccs_build_db`        - Routine that builds an ASE-database.
* `ccs_fit`             - The primary executable file for the ccs_fit package.
* `ccs_export_sktable`  - Export the spline in a dftbplus-compatible layout.
* `ccs_export_FF`       - Export the spline to lammps or GULP.
* `ccs_validate`        - Validation of the energies and forces of the fit compared to the training set.
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

### (Recommended) installing from pip
```
pip install ccs_fit
```

### Installing from source using poetry
```
git clone https://github.com/Teoroo-CMC/CCS_fit.git ccs_fit
cd ccs_fit

# Install python package manager poetry (see https://python-poetry.org/docs/ for more explicit installation instructions)
curl -sSL https://install.python-poetry.org | python3 -
# You might have to add poetry to your PATH
poetry --version # to see if poetry installed correctly
poetry install # to install ccs_fit
```
<!---
### Environment Variables
Set the following environment variables:
```
$export PYTHONPATH=<path-to-CCS-package>:$PYTHONPATH
$export PATH=<path-to-CCS-bin>:$PATH

Within a conda virtual environment, you can update the path by using:
conda develop <path-to-CCS-package>
```
--->

## Tutorials

We provide tutorials in the [examples](examples/) folder. To run the example, go to one of the folders. Each contain the neccesery input files required for the task at hand. 

## Authors

* **Akshay Krishna AK** 
* **Jolla Kullgren** 
* **Eddie Wadbro** 
* **Peter Broqvist**
* **Thijs Smolders**

## Funding
This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 957189, and the Swedish National Strategic e-Science programme eSSENCE. The project is part of BATTERY 2030+, the large-scale European research initiative for inventing the sustainable batteries of the future.

## License
This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgement
We want to thank Pavlin Mitev, Christof Köhler, Matthew Wolf, Kersti Hermansson, Bálint Aradi and Tammo van der Heide, and all the members of the [TEOROO-group](http://www.teoroo.kemi.uu.se/) at Uppsala University, Sweden for fruitful discussions and general support.
