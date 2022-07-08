# DFTB repulsive fitting example

The example can be run in two modes:

1. Using the command-line.
2. Via jupyter-notebook file [`run_example.ipynb`](run_example.ipynb)

## 1. The "command-line way"

Start by enwoking the command `ccs_fetch` to build the `structures.json` file from the two ASE-databases `DFT.db` and `DFTB.db`.

```
ccs_fetch DFTB 6.0 all DFT.db DFTB.db
```
This should produce the following standard output. The progress bar at bottom allow the user to get an idea of how long time the process will take.

```
--- USAGE:  ccs_fetch MODE [...] --- 
    The following modes and inputs are supported:
      CCS:   CutoffRadius(float) NumberOfSamples(int) DFT.db(string)
      CCS+Q: CutoffRadius(float) NumberOfSamples(int) DFT.db(string) charge_dict(string)
      DFTB:  CutoffRadius(float) NumberOfSamples(int) DFT.db(string) DFTB.db(string)
 
    Mode:  DFTB
    R_c set to:  6.0
    Number of samples:  all
    DFT reference data base:  DFT.db
    DFTB reference data base:  DFTB.db

-------------------------------------------------
100%|████████████████████████████████| 75/75 [00:00<00:00, 89.50it/s]
```
Next write the input to the `CCS_input.json` file as shown below:
```
{
        "General": {
                "interface": "DFTB"
        },
        "Twobody": {
                "Ce-O": {
                        "Rcut": 6.0,
                        "Resolution": 0.05,
                        "Swtype": "rep"
                }
        }
}
```
This input instructs `CCS` to only fit the Ce-O contribution using a strictly repulsive potential with a cutoff of 6.0 Å. 

The actual fitting is performed by issuing the command:
```
ccs_fit
```
This will produce two primary output files: 

* `CCS_params.json` containing the spline tables and other all parameters defining the repulsive potential. 
* `CCS_error.out` a summary of the fitting quality.

A compressed version of the  `CCS_error.out` file is shown below: 
``` 
# Reference      Predicted      Error          #atoms         
15245.26615     15240.46770     4.79845         240.00000      
15215.18838     15219.35916     4.17077         240.00000      
14246.75475     14249.55891     2.80415         225.00000      
18269.90518     18266.58382     3.32136         288.00000      
5304.43208      5304.96708      0.53501         81.00000       
5233.40533      5234.02350      0.61817         81.00000       
...       
190.13442       190.57544       0.44102         3.00000        
189.65926       190.14065       0.48139         3.00000        
189.22154       189.73873       0.51719         3.00000        
188.82129       189.36891       0.54762         3.00000        
188.45163       189.03116       0.57954         3.00000        
188.11116       188.72550       0.61434         3.00000        
187.79439       188.45173       0.65733         3.00000        
187.50071       188.20836       0.70765         3.00000        
187.22647       187.99306       0.76659         3.00000        
# MSE = 1.27630E+00
# Maxerror = 4.79845E+00
```