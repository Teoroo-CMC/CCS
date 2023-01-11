#!/usr/bin/env python3
import ccs_fit.regression_tool.regression as ccsr

import numpy as np
from matplotlib import pyplot

x = [132.3, 151.4, 200.0, 300.0, 600.0, 800.0, 1051.0, 1253.6, 1486.6, 8047.8]
y = [0.6029, 0.4967, 0.4278, 0.3004, 0.1609,
     0.44E-1, 0.24E-1, 0.13E-1, 0.58E-2, 0.66E-4]

CF = ccsr.CCS_regressor(xmin=120, xmax=10000, N=20)

CF.fit(x, y)

xf = np.linspace(132.3, 10000.0, 10000, dtype=float)
yf = CF.predict(xf)

pyplot.scatter(x, y,)
pyplot.plot(xf, yf, '-', color='black')

pyplot.ylabel(r'y')
pyplot.xlabel(r'x')
pyplot.savefig("output.png", dpi=1000)
