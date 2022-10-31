# CCS - Curvature Constrained Splines  
<div style="text-align:center"><img src="https://raw.githubusercontent.com/aksam432/CCS/master/logo.png" width=600/></div>

The `CCS` package is a tool to construct two-body potentials using the idea of curvature constrained splines.
## Getting Started
### Package Layout
```
ccs-x.y.z
├── bin
│   ├── ccs_fetch
│   ├── ccs_fit               
│   └── ccs_validate                 
├── ccs
│   ├── __init__.py                     
│   ├── main.py                 
│   ├── objective.py            
│   └── spline_functions.py     
├── examples
│   ├── Ne-FCC
│   │   ├── input.json          
│   │   └── structures.json     
│   └── O2
│       ├── input.json
│       └── structures.json
├── LICENSE
├── README.md
└── setup.py
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
pip install -e git+<https://github.com/Teoroo-CMC/CCS>
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



