import sys
import numpy as np
import itertools as it
from collections import OrderedDict, defaultdict
from numpy import linalg as LA
import json


def _write(elem1, elem2, CCS_params, exp=True):
    elem1 = elem1
    elem2 = elem2
    no_pair = False
    try:
        pair = elem1+'-'+elem2
        rcut = CCS_params['Two_body'][pair]['r_cut']
    except:
        try:
            pair = elem2+'-'+elem1
            rcut = CCS_params['Two_body'][pair]['r_cut']
        except:
            rcut = 0.0
            no_pair = True
    if(no_pair):
        pass
    else:
        Rmin = CCS_params['Two_body'][pair]['r_min']
        Rcut = CCS_params['Two_body'][pair]['r_cut']
        a = CCS_params['Two_body'][pair]['spl_a']
        b = CCS_params['Two_body'][pair]['spl_b']
        c = CCS_params['Two_body'][pair]['spl_c']
        d = CCS_params['Two_body'][pair]['spl_d']
        aa = CCS_params['Two_body'][pair]['exp_a']
        bb = CCS_params['Two_body'][pair]['exp_b']
        cc = CCS_params['Two_body'][pair]['exp_c']
        exp = False
        x = CCS_params['Two_body'][pair]['r']
        dx = CCS_params['Two_body'][pair]['dr']

        with open(f'{elem1}-{elem2}.spl', 'w') as f:
            print('Spline', file=f)
            print(len(x), Rcut+dx, file=f)
            print(aa, bb, cc, file=f)
            for i in range(len(x)):
                print(x[i], x[i]+dx, a[i], b[i], c[i], d[i], file=f)
        with open(f'{elem2}-{elem1}.spl', 'w') as f:
            print('Spline', file=f)
            print(len(x), Rcut+dx, file=f)
            print(aa, bb, cc, file=f)
            for i in range(len(x)):
                print(x[i], x[i]+dx, a[i], b[i], c[i], d[i], file=f)


def write_dftb_spline(CCS_params_file):

    with open(CCS_params_file, 'r') as f:
        CCS_params = json.load(f)

    for pair in CCS_params['Two_body']:
        elem = pair.split("-")
        _write(elem[0], elem[1], CCS_params, exp=True)


def main():
    CCS_params_file = sys.argv[1]
    write_dftb_spline(CCS_params_file)


if __name__ == "__main__":
    main()
