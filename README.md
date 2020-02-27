# CCS - Curvature Constrained Splines  
<div style="text-align:center"><img src="https://raw.githubusercontent.com/aksam432/CCS/master/logo.png" width=400/></div>

The `CCS` package is a tool to construct two-body potentials using the idea of curvature constrained splines.
## Getting Started
### Package Layout
```
ccs-0.1.0
├── bin
│   ├── atom-json               
│   └── ccs_fit                 
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
* `atom-json`-A tool to convert OUTCAR (VASP output) files to structures.json
* `ccs_fit`             - The executable file for ccs package.
* `main.py`             - A module to parse input files.
* `objective.py`        - A module which contains the objective function and solver.
* `spline_functions.py` - A module for spline construction/evaluation/output. 
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
git clone git@github.com:aksam432/CCS.git
cd CCS
python setup.py install
```
### Environment Variables
Set the following environment variables:
```
$pip export PYTHONPATH=<path-to-CCS-package>:$PYTHONPATH
$ export PATH=<path-to-CCS-bin>:$PATH
```


## Tutorials

We provide two tutorials in the [examples](examples/) folder. To run the example,  go to one of the folders ( O2 or Ne-FCC ). Each contain two primary input files required for fitting. A sample `input.json` for O2 is shown below:
```
{

"Twobody":{
	"O-O":{
		"Rmin":0.95,   
		"Rcut":2.5,
		"Nknots":20

	}
	},

"Onebody":["O"],

"Reference":"structures.json"
			
}
```
The `input.json` file should provide spline interval cutoff's (Rcut and Rmin), number of knots (Nknots), and path to the `structure.json` file. The onebody energy terms should be provided as a json array using atomic symbols. 
The format for `structure.json` is shown below :
```
{
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
```
The `structure.json` file contains different configurations labeled ("S1", "S2"...) and corresponding energy, pairwise distances (contained in an array labelled as "O-O" for oxygen). The stoichiometry of each configuration is given under the atoms label (" Atoms") as a key-value pair ("O" : 2 ). 


To perform the fit : 
```
ccs_fit
```
The following output files are obtained:
```
splines.out error.out ccs.log summary.png 
```
* splines.out  - Contains the spline coefficients  for two body potential.
* error.out    - Contains target energies, predicted energies and absolute error for each configuration.
* ccs.log       - Contains debug information
* summary.png   -  Plot showing fit quality, selected spline coefficients, and distance histogram.
## Authors

* **Akshay Krishna AK** 
* **Jolla Kullgren** 
* **Eddie Wadbro** 


## License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgement
We thank all the members of  [TEOROO-group](http://www.teoroo.kemi.uu.se/) at Uppsala University, Sweden.



