import ccs.regression_tool.regression as ccsr
import numpy as np

x = np.linspace(0.0, 1.0, 10, dtype=float)
y = np.exp(-x)

CF = ccsr.CCS_regressor()
CF.fit(x, y)
yf = CF.predict(x)

print(yf, y)
