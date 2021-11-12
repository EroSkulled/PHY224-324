import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

dat = np.loadtxt('../wire.csv', delimiter=',', dtype=None, encoding=None, skiprows=1)
v = dat.transpose()[1]
a = dat.transpose()[2] / 1000
rl = dat.transpose()[3]

vuncertain = v * 0.0025
auncertain = a * 0.0075

vb = v[:4]
ab = a[:4]
vbuncertain = vuncertain[:4]
abuncertain = auncertain[:4]

vc = v[4:8]
ac = a[4:8]
vcuncertain = vuncertain[4:8]
acuncertain = auncertain[4:8]


def model_function(x, a, b):
    return -a * x + b


popt, pcov = curve_fit(model_function, ab, vb, sigma=abuncertain, absolute_sigma=True)
pstd = np.sqrt(np.diag(pcov))
a, b = popt


popt2, pcov2 = curve_fit(model_function, ac, vc, sigma=acuncertain, absolute_sigma=True)
pstd2 = np.sqrt(np.diag(pcov))
a2, b2 = popt2
print("Option1 output resistance (R_b): ", np.round(a2, 5), " +-", np.round(pstd2[0], 5) , " Ohm, open-circuit voltage: ", np.round(b2, 5),  " +-", np.round(pstd2[1], 5), " V")
print("Option2 output resistance (R_b): ", np.round(a, 5), " +-", np.round(pstd[0], 5) , " Ohm, open-circuit voltage:", np.round(b, 5),  " +-", np.round(pstd[1], 5), " V")

plt.errorbar(ac, vc, yerr=vcuncertain, xerr=acuncertain, label='Cell Battery Option1 data with uncertainty', ls='None')
plt.errorbar(ab, vb, yerr=vbuncertain, xerr=abuncertain, label='Cell Battery Option2 data with uncertainty', ls='None')

plt.xlabel('Current(A)')
plt.ylabel('Voltage(V)')
plt.legend(loc='best')
plt.title('Voltage vs. Current Plot')
plt.show()


