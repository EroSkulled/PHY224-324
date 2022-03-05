import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def T(LD):
    return 2 * ((LD) ** 0.5)


def radian(degree):
    return degree * np.pi / 180


def chi_square(ys, fs, sigma):
    chi_sqr = 0
    for i in range(0, len(ys)):
        chi_sqr += ((ys[i] - fs[i]) / sigma[i]) ** 2
    return chi_sqr


def fit(t, tau):
    return rad[0] * np.exp(-t / tau) * np.cos(2 * np.pi * (t / period))


def refit(t, a, b, c):
    return rad[0]*a**(b*t)*np.cos(c*t)


data = pd.read_csv(r"/pendulum/Data1.csv", delimiter=',', skiprows=0)
t = data['t'] - data['t'].iloc[0]  # set the beginning of timeline to 0
x = data['x']
y = data['y']
rad = data['rad']
err = radian(np.zeros(len(y)) + 0.5)
period = T(0.35)
Terr = T(0.36) - period
print(Terr)
print("Period from formula: " + str(period))
print("Period observed: " + str(24 / 19))

# popt, pcov = curve_fit(fit, t, rad, 1, err, True)
# cov2 = np.diag(pcov)
plt.figure(figsize=(10, 10))
plt.errorbar(t, rad, yerr=err, xerr=0.1, fmt='.', label='Original data')
# plt.plot(t, fit(t, *popt), color='red', marker='|',
#          label='fit: tau=%.3e +- ' % tuple(popt) + '%.3e' % tuple(np.sqrt(cov2)))

popt , pcov = curve_fit(refit, t, rad,  (1, 1, 1), err, True)
pvar = np.diag(pcov)
plt.plot(t, refit(t, *popt), color='red', marker='|',
         label='refit: tau=%.3e +- ' % popt[1] + '%.3e' % np.sqrt(pvar[1]))


plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title('Figure.1 ')
plt.legend(loc='best')
plt.show()
print(popt, ' +- ', np.sqrt(pvar))
ys = np.zeros(len(rad))
for i in range(0, len(rad)):
    ys[i] = refit(t[i], *popt)
v = len(t) - len(popt)
r2 = chi_square(ys, rad, err)
print(r2 / v)

data2 = pd.read_csv(r"/pendulum/Data2.csv", delimiter=',', skiprows=1)
t = data2['t'] - data2['t'].iloc[0]  # set the beginning of timeline to 0
x = data2['x']
y = data2['y']
rad = data2['rad']
err = radian(np.zeros(len(y)) + 0.5)
period = T(0.35)
Terr = T(0.36) - period
print(Terr)
print("Period from formula: " + str(period))
print("Period observed: " + str(35 / 28))

# popt, pcov = curve_fit(fit, t, rad, 10, err, True)
# cov2 = np.diag(pcov)
plt.figure(figsize=(10, 10))
plt.errorbar(t, rad, yerr=err, xerr=0.1, fmt='.', label='Original data')
# plt.plot(t, fit(t, *popt), color='red', marker='|',
#          label='fit: tau=%.3e +- ' % tuple(popt) + '%.3e' % tuple(np.sqrt(cov2)))
popt , pcov = curve_fit(refit, t, rad,  (1, 1, 1), err, True)
pvar = np.diag(pcov)
plt.plot(t, refit(t, *popt), color='red', marker='|',
         label='refit: tau=%.3e +- ' % popt[1] + '%.3e' % np.sqrt(pvar[1]))
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title('Figure.2 ')
plt.legend(loc='best')
plt.show()
print(popt, ' +- ', np.sqrt(pvar))
ys = np.zeros(len(rad))
for i in range(0, len(rad)):
    ys[i] = refit(t[i], *popt)
v = len(t) - len(popt)
r2 = chi_square(ys, rad, err)
print(r2 / v)

data3 = pd.read_csv(r"/pendulum/Data3.csv", delimiter=',', skiprows=0)
t = data3['t'] - data3['t'].iloc[0]  # set the beginning of timeline to 0
x = data3['x']
y = data3['y']
rad = data3['rad']
err = radian(np.zeros(len(y)) + 0.5)
period = T(0.35)
Terr = T(0.36) - period
print(Terr)
print("Period from formula: " + str(period))
print("Period observed: " + str(37 / 30))

# popt, pcov = curve_fit(fit, t, rad, 10, err, True)
# cov2 = np.diag(pcov)
plt.figure(figsize=(10, 10))
plt.errorbar(t, rad, yerr=err, xerr=0.1, fmt='.', label='Original data')
popt , pcov = curve_fit(refit, t, rad,  (1, 1, 1), err, True)
pvar = np.diag(pcov)
plt.plot(t, refit(t, *popt), color='red', marker='|',
         label='refit: tau=%.3e +- ' % popt[1] + '%.3e' % np.sqrt(pvar[1]))
# plt.plot(t, fit(t, *popt), color='red', marker='|',
#          label='fit: tau=%.3e +- ' % tuple(popt) + '%.3e' % tuple(np.sqrt(cov2)))
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title('Figure.3 ')
plt.legend(loc='best')
plt.show()
print(popt, ' +- ', np.sqrt(pvar))
ys = np.zeros(len(rad))
for i in range(0, len(rad)):
    ys[i] = refit(t[i], *popt)
v = len(t) - len(popt)
r2 = chi_square(ys, rad, err)
print(r2 / v)

data4 = pd.read_csv(r"/pendulum/Data4.csv", delimiter=',', skiprows=0)
t = data4['t'] - data4['t'].iloc[0]  # set the beginning of timeline to 0
x = data4['x']
y = data4['y']
rad = data4['rad']
err = radian(np.zeros(len(y)) + 0.5)
period = T(0.35)
Terr = T(0.36) - period
print(Terr)
print("Period from formula: " + str(period))
print("Period observed: " + str(40 / 31))

# popt, pcov = curve_fit(fit, t, rad, 10, err, True)
# cov2 = np.diag(pcov)
plt.figure(figsize=(10, 10))
plt.errorbar(t, rad, yerr=err, xerr=0.1, fmt='.', label='Original data')
popt , pcov = curve_fit(refit, t, rad,  (1, 1, 1), err, True)
pvar = np.diag(pcov)
plt.plot(t, refit(t, *popt), color='red', marker='|',
         label='refit: tau=%.3e +- ' % popt[1] + '%.3e' % np.sqrt(pvar[1]))
# plt.plot(t, fit(t, *popt), color='red', marker='|',
#          label='fit: tau=%.3e +- ' % tuple(popt) + '%.3e' % tuple(np.sqrt(cov2)))
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title('Figure.4 ')
plt.legend(loc='best')
plt.show()
print(popt, ' +- ', np.sqrt(pvar))
ys = np.zeros(len(rad))
for i in range(0, len(rad)):
    ys[i] = refit(t[i], *popt)
v = len(t) - len(popt)
r2 = chi_square(ys, rad, err)
print(r2 / v)

