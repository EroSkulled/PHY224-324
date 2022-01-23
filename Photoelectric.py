import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd


def f(x, a, b):
    return a * x + b


def error(ydata):
    v_error = np.empty(len(ydata))
    for i in range(len(ydata)):
        v_error[i] = max(ydata[i] * 0.0010, 0.01)
    return v_error


data = pd.read_csv(r"C:\Users\Walter\IdeaProjects\PHY224\Photoelectric\EXP1.csv", delimiter=',', skiprows=0)

plt.title('LED Color stopping Voltage vs. wave length')
plt.ylabel('Stop Voltage (V)')
plt.xlabel('Wave Length (nm)')
xdata1 = data["wavelength "]
ydata1 = data["V stop"]

plt.errorbar(xdata1, ydata1, label='Test 1', yerr=error(ydata1), linestyle='None', marker=".")

data = pd.read_csv(r"C:\Users\Walter\IdeaProjects\PHY224\Photoelectric\EXP2.csv", delimiter=',', skiprows=0)
xdata2 = data["wavelength"]
ydata2 = data["Vstop"]
plt.errorbar(xdata2, ydata2, label='Test 2', yerr=error(ydata2), linestyle='None', marker=".")

data = pd.read_csv(r"C:\Users\Walter\IdeaProjects\PHY224\Photoelectric\EXP3.csv", delimiter=',', skiprows=0)
xdata3 = data["wavelength"]
ydata3 = data["Vstop"]
plt.errorbar(xdata3, ydata3, label='Test 3', yerr=error(ydata3), linestyle='None', marker=".")

xdata = (xdata1 + xdata2 + xdata3) / 3
ydata = (ydata1 + ydata2 + ydata3) / 3

plt.plot(xdata, ydata, label='Average')
plt.legend()
plt.show()

# Computing frequency from wavelength
frequency = (3 * 10 ** 8) / xdata

# Estimated errors from equipment
v_error = np.empty(len(ydata))
for i in range(len(ydata)):
    v_error[i] = max(ydata[i] * 0.0010, 0.01)

# Linear regression
p_opt, p_cov = curve_fit(f, frequency, ydata, (0, 0), v_error, True)
lin_output = f(frequency, p_opt[0], p_opt[1])

# Outputting Planck's Constant h
h = p_opt[0] * (1.6 * 10 ** (-19))
h_error = p_cov[0, 0] * (1.6 * 10 ** (-19))
print('Estimated Plancks Constant: ', h, '(J*s) +/-', h_error, '(J**s)')

# Outputting the Work Function
wf = -p_opt[1] * (1.6 * 10 ** (-19))
wf_error = p_cov[1, 1] * (1.6 * 10 ** (-19))
print('Estimated Work Function: ', wf, '(J) +/-', wf_error, '(J)')

# Outputting the cut-off frequency
f0 = -(1.6 * 10 ** (-19)) * p_opt[1] / h
f0_error = p_cov[1, 1] * (1.6 * 10 ** (-19)) / h
print('Estimated Cut-off Frequency: ', f0, '(Hz) +/-', f0_error, '(Hz)')


# Calculating chi squared
chi_sq = (1 / 2) * (np.sum(((ydata - lin_output) / v_error) ** 2))
print('Chi squared for linear regression: ', chi_sq)

plt.title('Curve Fit vs. Original Data')
plt.ylabel('Stop Voltage (V)')
plt.xlabel('Wave Length (nm)')
plt.errorbar(xdata, ydata, label='Average', yerr=v_error, linestyle='None', marker=".")
plt.plot(xdata, lin_output, label='Curve fit ')
plt.legend()
plt.show()
