import csv

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


def refit(t, a, b, c, d):
    return rad[0] * a * np.exp(-t / b) * np.cos(2 * np.pi * c * (t / period) + d)


result = []
data = pd.read_csv(r"/pendulum/Data1.csv", delimiter=',', skiprows=0)
t = data['t'] - data['t'].iloc[0]  # set the beginning of timeline to 0
x = data['x']
y = data['y']
rad = data['rad']
err = radian(np.zeros(len(y)) + 0.5)
period = T(0.39)
Terr = T(0.40) - period
print('Time err: ', Terr)
print("Period from formula: " + str(period))
print("Period observed: " + str(23 / 18))

popt, pcov = curve_fit(fit, t, rad, 1, err, True)
cov2 = np.diag(pcov)
plt.figure(figsize=(10, 10))
plt.errorbar(t, rad, yerr=err, xerr=Terr, fmt='.', label='Original data')
plt.plot(t, fit(t, *popt), color='orange', marker='|',
         label='fit: tau=%.3e +- ' % tuple(popt) + '%.3e' % tuple(np.sqrt(cov2)))

popt, pcov = curve_fit(refit, t, rad, (1, 1, 1, 1), err, True)
pvar = np.diag(pcov)
plt.plot(t, refit(t, *popt), color='red', marker='|',
         label='Refit: tau=%.3e +- ' % popt[1] + '%.3e' % np.sqrt(pvar[1]))

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
print(r2 / v ** 2)
result.append([*popt])

data2 = pd.read_csv(r"/pendulum/Data2.csv", delimiter=',', skiprows=1)
t = data2['t'] - data2['t'].iloc[0]  # set the beginning of timeline to 0
x = data2['x']
y = data2['y']
rad = data2['rad']
err = radian(np.zeros(len(y)) + 0.5)
period = T(0.39)
Terr = T(0.40) - period

print("Period from formula: " + str(period))
print("Period observed: " + str(35 / 28))

popt, pcov = curve_fit(fit, t, rad, 10, err, True)
cov2 = np.diag(pcov)
plt.figure(figsize=(10, 10))
plt.errorbar(t, rad, yerr=err, xerr=Terr, fmt='.', label='Original data')
plt.plot(t, fit(t, *popt), color='orange', marker='|',
         label='fit: tau=%.3e +- ' % tuple(popt) + '%.3e' % tuple(np.sqrt(cov2)))
popt, pcov = curve_fit(refit, t, rad, (1, 1, 1, 1), err, True)
pvar = np.diag(pcov)
plt.plot(t, refit(t, *popt), color='red', marker='|',
         label='Refit: tau=%.3e +- ' % popt[1] + '%.3e' % np.sqrt(pvar[1]))
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
print(r2 / v ** 2)
result.append([*popt])


data3 = pd.read_csv(r"/pendulum/Data3.csv", delimiter=',', skiprows=0)
t = data3['t'] - data3['t'].iloc[0]  # set the beginning of timeline to 0
x = data3['x']
y = data3['y']
rad = data3['rad']
err = radian(np.zeros(len(y)) + 0.5)
period = T(0.39)
Terr = T(0.40) - period

print("Period from formula: " + str(period))
print("Period observed: " + str(37 / 30))

popt, pcov = curve_fit(fit, t, rad, 10, err, True)
cov2 = np.diag(pcov)
plt.figure(figsize=(10, 10))
plt.errorbar(t, rad, yerr=err, xerr=Terr, fmt='.', label='Original data')
plt.plot(t, fit(t, *popt), color='orange', marker='|',
         label='fit: tau=%.3e +- ' % tuple(popt) + '%.3e' % tuple(np.sqrt(cov2)))
popt, pcov = curve_fit(refit, t, rad, (1, 1, 1, 1), err, True)
pvar = np.diag(pcov)
plt.plot(t, refit(t, *popt), color='red', marker='|',
         label='Refit: tau=%.3e +- ' % popt[1] + '%.3e' % np.sqrt(pvar[1]))

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
print(r2 / v ** 2)
result.append([*popt])

data4 = pd.read_csv(r"/pendulum/Data4.csv", delimiter=',', skiprows=0)
t = data4['t'] - data4['t'].iloc[0]  # set the beginning of timeline to 0
x = data4['x']
y = data4['y']
rad = data4['rad']
err = radian(np.zeros(len(y)) + 0.5)
period = T(0.39)
Terr = T(0.40) - period

print("Period from formula: " + str(period))
print("Period observed: " + str(30 / 24))

popt, pcov = curve_fit(fit, t, rad, 10, err, True)
cov2 = np.diag(pcov)
plt.figure(figsize=(10, 10))
plt.errorbar(t, rad, yerr=err, xerr=Terr, fmt='.', label='Original data')
plt.plot(t, fit(t, *popt), color='orange', marker='|',
         label='fit: tau=%.3e +- ' % tuple(popt) + '%.3e' % tuple(np.sqrt(cov2)))
popt, pcov = curve_fit(refit, t, rad, (1, 1, 1, 1), err, True)
pvar = np.diag(pcov)
plt.plot(t, refit(t, *popt), color='red', marker='|',
         label='Refit: tau=%.3e +- ' % popt[1] + '%.3e' % np.sqrt(pvar[1]))

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
print(r2 / v ** 2)
result.append([*popt])

data5 = pd.read_csv(r"/pendulum/Data5.csv", delimiter=',', skiprows=0)
t = data5['t'] - data5['t'].iloc[0]  # set the beginning of timeline to 0
x = data5['x']
y = data5['y']
rad = radian(data5['deg'])
err = radian(np.zeros(len(y)) + 0.5)
period = T(0.39)
Terr = T(0.40) - period

print("Period from formula: " + str(period))
print("Period observed: " + str(31 / 24))

popt, pcov = curve_fit(fit, t, rad, 10, err, True)
cov2 = np.diag(pcov)
plt.figure(figsize=(10, 10))
plt.errorbar(t, rad, yerr=err, xerr=Terr, fmt='.', label='Original data')
plt.plot(t, fit(t, *popt), color='orange', marker='|',
         label='fit: tau=%.3e +- ' % tuple(popt) + '%.3e' % tuple(np.sqrt(cov2)))
popt, pcov = curve_fit(refit, t, rad, (1, 1, 1, 1), err, True)
pvar = np.diag(pcov)
plt.plot(t, refit(t, *popt), color='red', marker='|',
         label='Refit: tau=%.3e +- ' % popt[1] + '%.3e' % np.sqrt(pvar[1]))

plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title('Figure.5 ')
plt.legend(loc='best')
plt.show()
print(popt, ' +- ', np.sqrt(pvar))
ys = np.zeros(len(rad))
for i in range(0, len(rad)):
    ys[i] = refit(t[i], *popt)
v = len(t) - len(popt)
r2 = chi_square(ys, rad, err)
print(r2 / v / 2)
result.append([*popt])

data6 = pd.read_csv(r"/pendulum/Data6.csv", delimiter=',', skiprows=0)
t = data6['t'] - data6['t'].iloc[0]  # set the beginning of timeline to 0
x = data6['x']
y = data6['y']
rad = radian(data6['deg'])
err = radian(np.zeros(len(y)) + 0.5)
period = T(0.39)
Terr = T(0.40) - period

print("Period from formula: " + str(period))
print("Period observed: " + str(40 / 32))

popt, pcov = curve_fit(fit, t, rad, 10, err, True)
cov2 = np.diag(pcov)
plt.figure(figsize=(10, 10))
plt.errorbar(t, rad, yerr=err, xerr=Terr, fmt='.', label='Original data')
plt.plot(t, fit(t, *popt), color='orange', marker='|',
         label='fit: tau=%.3e +- ' % tuple(popt) + '%.3e' % tuple(np.sqrt(cov2)))
popt, pcov = curve_fit(refit, t, rad, (1, 1, 1, 1), err, True)
pvar = np.diag(pcov)
plt.plot(t, refit(t, *popt), color='red', marker='|',
         label='Refit: tau=%.3e +- ' % popt[1] + '%.3e' % np.sqrt(pvar[1]))

plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title('Figure.6 ')
plt.legend(loc='best')
plt.show()
print(popt, ' +- ', np.sqrt(pvar))
ys = np.zeros(len(rad))
for i in range(0, len(rad)):
    ys[i] = refit(t[i], *popt)
v = len(t) - len(popt)
r2 = chi_square(ys, rad, err)
print(r2 / 2 / v)
result.append([*popt])

data7 = pd.read_csv(r"/pendulum/Data7.csv", delimiter=',', skiprows=0)
t = data7['t'] - data7['t'].iloc[0]  # set the beginning of timeline to 0
x = data7['x']
y = data7['y']
rad = radian(data7['deg'])
err = radian(np.zeros(len(y)) + 0.5)
period = T(0.48)
Terr = T(0.49) - period
print('Time err: ', Terr)
print("Period from formula: " + str(period))
print("Period observed: " + str(40 / 29))

popt, pcov = curve_fit(fit, t, rad, 10, err, True)
cov2 = np.diag(pcov)
plt.figure(figsize=(10, 10))
plt.errorbar(t, rad, yerr=err, xerr=Terr, fmt='.', label='Original data')
plt.plot(t, fit(t, *popt), color='orange', marker='|',
         label='fit: tau=%.3e +- ' % tuple(popt) + '%.3e' % tuple(np.sqrt(cov2)))
popt, pcov = curve_fit(refit, t, rad, (1, 1, 1, 1), err, True)
pvar = np.diag(pcov)
plt.plot(t, refit(t, *popt), color='red', marker='|',
         label='Refit: tau=%.3e +- ' % popt[1] + '%.3e' % np.sqrt(pvar[1]))

plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title('Figure.7 ')
plt.legend(loc='best')
plt.show()
print(popt, ' +- ', np.sqrt(pvar))
ys = np.zeros(len(rad))
for i in range(0, len(rad)):
    ys[i] = refit(t[i], *popt)
v = len(t) - len(popt)
r2 = chi_square(ys, rad, err)
print(r2 / 2 / v)
result.append([*popt])

data8 = pd.read_csv(r"/pendulum/Data8.csv", delimiter=',', skiprows=0)
t = data8['t'] - data8['t'].iloc[0]  # set the beginning of timeline to 0
x = data8['x']
y = data8['y']
rad = radian(data8['deg'])
err = radian(np.zeros(len(y)) + 0.5)
period = T(0.70)
Terr = T(0.71) - period

print("Period from formula: " + str(period))
print("Period observed: " + str(30 / 18))

popt, pcov = curve_fit(fit, t, rad, 10, err, True)
cov2 = np.diag(pcov)
plt.figure(figsize=(10, 10))
plt.errorbar(t, rad, yerr=err, xerr=Terr, fmt='.', label='Original data')
plt.plot(t, fit(t, *popt), color='orange', marker='|',
         label='fit: tau=%.3e +- ' % tuple(popt) + '%.3e' % tuple(np.sqrt(cov2)))
popt, pcov = curve_fit(refit, t, rad, (1, 1, 1, 1), err, True)
pvar = np.diag(pcov)
plt.plot(t, refit(t, *popt), color='red', marker='|',
         label='Refit: tau=%.3e +- ' % popt[1] + '%.3e' % np.sqrt(pvar[1]))

plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title('Figure.9 ')
plt.legend(loc='best')
plt.show()
print(popt, ' +- ', np.sqrt(pvar))
ys = np.zeros(len(rad))
for i in range(0, len(rad)):
    ys[i] = refit(t[i], *popt)
v = len(t) - len(popt)
r2 = chi_square(ys, rad, err)
print(r2 / 2 / v)
result.append([*popt])

data9 = pd.read_csv(r"/pendulum/Data9.csv", delimiter=',', skiprows=0)
t = data9['t'] - data9['t'].iloc[0]  # set the beginning of timeline to 0
x = data9['x']
y = data9['y']
rad = radian(data9['deg'])
err = radian(np.zeros(len(y)) + 0.5)
period = T(0.70)
Terr = T(0.71) - period
print('Time err: ', Terr)

print("Period from formula: " + str(period))
print("Period observed: " + str(30 / 18))

popt, pcov = curve_fit(fit, t, rad, 10, err, True)
cov2 = np.diag(pcov)
plt.figure(figsize=(10, 10))
plt.errorbar(t, rad, yerr=err, xerr=Terr, fmt='.', label='Original data')
plt.plot(t, fit(t, *popt), color='orange', marker='|',
         label='fit: tau=%.3e +- ' % tuple(popt) + '%.3e' % tuple(np.sqrt(cov2)))
popt, pcov = curve_fit(refit, t, rad, (1, 1, 1, 1), err, True)
pvar = np.diag(pcov)
plt.plot(t, refit(t, *popt), color='red', marker='|',
         label='Refit: tau=%.3e +- ' % popt[1] + '%.3e' % np.sqrt(pvar[1]))

plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title('Figure.10 ')
plt.legend(loc='best')
plt.show()
print(popt, ' +- ', np.sqrt(pvar))
ys = np.zeros(len(rad))
for i in range(0, len(rad)):
    ys[i] = refit(t[i], *popt)
v = len(t) - len(popt)
r2 = chi_square(ys, rad, err)
print(r2 / 2 / v)
result.append([*popt])

data10 = pd.read_csv(r"/pendulum/Data10.csv", delimiter=',', skiprows=0)
t = data10['t'] - data10['t'].iloc[0]  # set the beginning of timeline to 0
x = data10['x']
y = data10['y']
rad = radian(data10['deg'])
err = radian(np.zeros(len(y)) + 0.5)
period = T(0.48)
Terr = T(0.49) - period
print('Time err: ', Terr)

print("Period from formula: " + str(period))
print("Period observed: " + str(30 / 22))

popt, pcov = curve_fit(fit, t, rad, 10, err, True)
cov2 = np.diag(pcov)
plt.figure(figsize=(10, 10))
plt.errorbar(t, rad, yerr=err, xerr=Terr, fmt='.', label='Original data')
plt.plot(t, fit(t, *popt), color='orange', marker='|',
         label='fit: tau=%.3e +- ' % tuple(popt) + '%.3e' % tuple(np.sqrt(cov2)))
popt, pcov = curve_fit(refit, t, rad, (1, 1, 1, 1), err, True)
pvar = np.diag(pcov)
plt.plot(t, refit(t, *popt), color='red', marker='|',
         label='Refit: tau=%.3e +- ' % popt[1] + '%.3e' % np.sqrt(pvar[1]))

plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title('Figure.8 ')
plt.legend(loc='best')
plt.show()
print(popt, ' +- ', np.sqrt(pvar))
ys = np.zeros(len(rad))
for i in range(0, len(rad)):
    ys[i] = refit(t[i], *popt)
v = len(t) - len(popt)
r2 = chi_square(ys, rad, err)
print(r2 / 2 / v)
result.append([*popt])

with open('../result.csv', 'w', encoding='UTF8') as f:
    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    for row in result:
        writer.writerow(row)
