#!/usr/bin/env python3

import sys
from setuptools import setup, find_packages


if sys.version_info[:2] < (3,0):
    raise SystemExit('Python in version 3 required.')

setup(
    name='ccs',
    version='0.1.0',
    url='https://github.com/Teoroo-CMC/CCS',
    description='A Python package for fitting two-body potentials using' +\
    ' curvature constrained splines',
    author='Akshay Krishna AK',
    author_email='akshay.kandy@kemi.uu.se',
    platforms=['unix'],
    packages=find_packages(),
    scripts=['bin/ccs_build_db', 'bin/ccs_export_sktable', 'bin/ccs_fetch',
             'bin/ccs_fit', 'bin/ccs_validate'],
    license='GPLv3',
    install_requires=[
        'cvxopt',
        'numpy',
        'pandas',
        'scipy',
        'sympy',
        'tqdm',
        'ase'],
    classifiers=[
        'Programming Language :: Python',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering'],
    keywords = ['Two-Body Potentials', 'Optimization', 'Force-Field'])
