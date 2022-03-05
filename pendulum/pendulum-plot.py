import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def T(x, a, b):
    return a*x + b


def chi_square(ys, fs, sigma):
    chi_sqr = 0
    for i in range(0, len(ys)):
        chi_sqr += ((ys[i] - fs[i]) / sigma[i]) ** 2
    return chi_sqr


x = np.array([0.39, 0.39, 0.39, 0.39, 0.39, 0.39, 0.48, 0.48, 0.70, 0.70])
p = np.array([1.2631, 1.25, 1.2333, 1.25, 1.2917, 1.25, 1.3793, 1.3637, 1.6667, 1.6667])
err = np.zeros(10) + 0.1

popt, pcov = curve_fit(T, xdata=x, ydata=p, sigma=err, absolute_sigma=True)
pvar = np.diag(pcov)
plt.errorbar(x, p, yerr=err, fmt='.', label='Original data')
plt.plot(x, T(x, *popt), color='red', marker='|',
         label='fit')

plt.xlabel("Length (m)")
plt.ylabel("Period (s)")
plt.title('Observed period vs. Length (L+D)')
plt.legend(loc='upper left')
plt.show()
print(popt, ' +- ', np.sqrt(pvar))

ys = np.zeros(len(x))
for i in range(0, len(x)):
    ys[i] = T(x[i], *popt)
v = len(x) - len(popt)
r2 = chi_square(ys, p, err)
print(r2 / v)


m = np.array([31, 31, 31, 31, 31, 36.9, 31, 36.9, 31, 36.9])
t = np.array([22.38, 21.18, 30.07, 33.19, 33.33, 98.79, 126.1, 47.14, 153.7, 105.1])
terr = np.array([0.047, 0.034, 0.074, 0.079, 0.301, 1.140, 4.834, 1.194, 1.068, 2.201])
popt, pcov = curve_fit(T, xdata=x, ydata=t, sigma=terr, absolute_sigma=True)
pvar = np.diag(pcov)
plt.errorbar(x, t, yerr=terr, fmt='.', label='Original data')
plt.plot(x, T(x, *popt), color='red', marker='|',
         label='fit')

plt.xlabel("Length (m)")
plt.ylabel("Time constant")
plt.title('Time constant vs. Length (L+D)')
plt.legend(loc='upper left')
plt.show()

print(popt, ' +- ', np.sqrt(pvar))

ys = np.zeros(len(x))
for i in range(0, len(x)):
    ys[i] = T(x[i], *popt)
v = len(x) - len(popt)
r2 = chi_square(ys, p, err)
print(r2 / v ** 2)
