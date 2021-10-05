import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

dat = np.loadtxt('../rad1.txt', delimiter='	', dtype=None, encoding=None, skiprows=2)
dat1 = np.loadtxt('../rad1bkg.txt', delimiter='	', dtype=None, encoding=None, skiprows=2)
bkg = dat1.transpose()[1]
num, count = dat.transpose()[0], dat.transpose()[1]
uncertain = np.sqrt(count + np.mean(bkg))
# sample time = 20 seconds
count -= np.mean(bkg)
print(count)
rate = count / 20
rate_uncertain = np.sqrt(count) / 20
print(rate_uncertain)
print(uncertain)


def model_function(x, a, b):
    return a * x + b


def function(x, a, b):
    return b * np.e ** (a * x)
