import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from scipy import stats

data = pd.read_csv("../big.csv", delimiter=',', skiprows=0)
position = data['Big']
period_up = data['UP']
period_down = data['DOWN']
uncer = pd.read_csv("../uncertainty.csv", delimiter=',', skiprows=0)
uncertainty = uncer['period']

length = 1.0020  # In meter
length_uncertainty = 0.0020
# Two measurements were taken, and uncertainty for each measurement was 0.0010m
uncert = np.ones(len(period_up)) * np.std(uncertainty)
# uncertainty for the period based on the 8-oscillation test


def chi_square(ys, fs, sigma):
    chi_sqr = 0
    for i in range(0, len(ys)):
        chi_sqr += ((ys[i] - fs[i]) / sigma[i]) ** 2
    chi_sqr = (1 / (len(ys) - 2)) * chi_sqr
    return chi_sqr


def model_func(x, a, b, c):
    # model for coarse mass
    return a * x ** 2 + b * x + c


def model_linear(x, a, b):
    # model for fine mass
    return a * x + b


############################
# coarse  mass

fig, ax = plt.subplots()
popt1, pcov1 = curve_fit(model_func, position, period_up, absolute_sigma=True, sigma=uncert)
popt2, pcov2 = curve_fit(model_func, position, period_down, absolute_sigma=True, sigma=uncert)
ax.errorbar(position, period_up, yerr=uncert, fmt='ro', label='Period 1')
ax.errorbar(position, period_down, yerr=uncert, fmt='bo', label='Period 2')
ax.plot(position, model_func(position, *popt1), 'r-')
ax.plot(position, model_func(position, *popt2), 'b-')

plt.xlabel('Position of mass (cm)')
plt.ylabel('Period of 8 oscillations (s)')
plt.title('Period of Oscillation vs. Coarse Adjustment Mass Position')
plt.legend(loc='lower right')
chi_square1 = chi_square(period_up, model_func(position, *popt1), uncert)
chi_square2 = chi_square(period_down, model_func(position, *popt2), uncert)
textstr = '\n'.join((
    r'period 1 $X^2/DoF=%.2f$/7' % (chi_square1,),
    r'period 2 $X^2/DoF=%.2f$/7' % (chi_square2,),))
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top')
plt.show()
print(chi_square1, chi_square2)

############################
# Fine mass

data = pd.read_csv("../small.csv", delimiter=',', skiprows=0)
position = data['Small']
period_up = data['UP']
period_down = data['DOWN']
uncert = np.ones(len(period_up)) * np.std(uncertainty)
popt1, pcov1 = curve_fit(model_linear, position, period_up, absolute_sigma=True, sigma=uncert)
popt2, pcov2 = curve_fit(model_linear, position, period_down, absolute_sigma=True, sigma=uncert)
root = (popt2[1] - popt1[1]) / (popt1[0] - popt2[0])
# position of intersection
fig, ax = plt.subplots()
ax.errorbar(position, period_up, yerr=uncert, fmt='ro', label='Period 1')
ax.errorbar(position, period_down, yerr=uncert, fmt='bo', label='Period 2')
ax.plot(position, model_linear(position, *popt1), 'r-')
ax.plot(position, model_linear(position, *popt2), 'b-')

plt.xlabel('Position of mass (cm)')
plt.ylabel('Period of 8 oscillations (s)')
plt.title('Period of Oscillation vs.Fine Adjustment Mass Position')

period = model_linear(root, *popt2) / 8
period_uncertainty = (abs(pcov1[0][0] / popt1[0]) + abs(pcov1[1][1] / popt1[1]) +
                      abs(pcov2[0][0] / popt2[0]) + abs(pcov2[1][1] / popt2[1])) * period
# period + uncertainty

g = (2 * np.pi) ** 2 * (length / period ** 2)
g_uncertainty = g * ((period_uncertainty / period) * 2 + length_uncertainty / length)
# g + uncertainty
chi_square1 = chi_square(period_up, model_linear(position, *popt1), uncert)
chi_square2 = chi_square(period_down, model_linear(position, *popt2), uncert)
prob1 = 1 - stats.chi2.cdf(chi_square1, len(period_up) - 1)
prob2 = 1 - stats.chi2.cdf(chi_square2, len(period_up) - 1)
# chi-square and fit probability
textstr = '\n'.join((
    r'period 1 $X^2/DoF=%.2f$/12' % (chi_square1,),
    r'period 2 $X^2/DoF=%.2f$/12' % (chi_square2,),
    r'period 1 $X^2prob.=%.2f$' % (prob1,),
    r'period 2 $X^2prob.=%.2f$' % (prob2,)))

ax.axvline(x=root, ymin=0, ymax=4.0, linewidth=1)
ax.text(root, 16.057, r' Intersection position $=%.2f$ cm' % (root,))
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top')
plt.legend(loc='lower right')
plt.show()
print('Period:\n', period, '+-', period_uncertainty, '(s)', chi_square1, chi_square2)
print('Calculated gravitational pull:\n', g, '+-', g_uncertainty, '(m/s^2)')
print('Chi-Square fit probability:\n', prob1, prob2)
print(popt1, pcov1)
print(popt2, pcov2)
print(np.std(uncertainty))
