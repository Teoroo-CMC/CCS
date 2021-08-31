#!/usr/bin/env python3

import sys
from setuptools import setup, find_packages


if sys.version_info[:2] < (3,0):
    raise SystemExit('Python in version 3 required.')

setup(
    name='ccs',
    version='0.1.0',
    url='https://github.com/aksam432/CCS',
    description=' A Python package for fitting two-body potentials using' +\
    ' curvature constrained splines',
    author='Akshay Krishna AK',
    author_email='akshay.kandy@kemi.uu.se',
    platforms=['unix'],
    packages=find_packages(),
    scripts=['bin/atom_json', 'bin/ccs_fit'],
    license='GPLv3',
    install_requires=[
        'cvxopt==1.2.4',
        'numpy>=1.18.1',
        'pandas==0.24.2',
        'scipy',
        'ase>=3.19.0'],
    classifiers=[
        'Programming Language :: Python',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering'],
    keywords = ['Two-Body Potentials', 'Optimization', 'Force-Field'])
