import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from scipy import stats

k = 0.10055  # meter constant
m_earth = 5.972e24  # mass of the earth kg
m_sun = 2e30  # mass of the sun kg
m_floor = 10e6  # mass of the floor kg
d_earth_sun = 1.50e11  # distance between the earth and the sun m
G = 6.67e-11  # gravitational constant
r_earth = 6.371009e6  # reference value for radius earth m
g_earth = 9.804253  # reference value for g of earth m/s^2
r_delta = 3.95  # change in height between floors m
sea_level = 115  # ground floor height above sea level
r_earth += sea_level


data = pd.read_csv('data.csv')
floor = data['floor']



# Converts from Div to Gal, cm/s^2
gal1 = data['value'] * k * 1000

# Converts from Gal to m/^2
g_delta = gal1 / 10000

Error = np.ones(len(g_delta)) * k * np.std(g_delta)


def radius(g_delta):
    return -2 * (r_delta) * (g_earth / g_delta)


def model(x, a, b):
    return a * x + b


def chi_square(ys, fs, sigma):
    chi_sqr = 0
    for i in range(0, len(ys)):
        chi_sqr += ((ys[i] - fs[i]) / sigma[i]) ** 2
    return chi_sqr




popt1, pcov1 = curve_fit(model, floor, g_delta, p0=[0, 0], sigma=Error,
                         absolute_sigma=True)  # Call linear best fit function
pvar1 = np.diag(pcov1)

# # plot g_delta vs. r_delta
#
plt.figure(figsize=(10, 10))
plt.plot(floor, model(floor, *popt1), c='r', label='fit')
plt.errorbar(floor, g_delta, yerr=Error, fmt='.', label='Original data')
plt.legend()
plt.xlabel('Floor number')
plt.ylabel('Rescaled Gravitimeter reading (m/s^2)')
plt.show()

R1 = radius(popt1[0] / 10000)  # in m
R1U = np.abs(((np.sqrt(pvar1[0]) / popt1[0])) * R1)  # unceratanty calculation

print('The Radius of the Earth from the first days measurement is:', R1, '±', R1U)
print('Difference is ', r_earth - R1, 'm')
print(*popt1, *pvar1)



# accomodate error from floor mass

floor_delta = abs(G * (m_earth + 10e6 * (15 - floor)) / (r_earth + r_delta * (15 - floor))**2 - G * m_earth / r_earth**2)
g_delta -= floor_delta

popt1, pcov1 = curve_fit(model, floor, g_delta, p0=[0, 0], sigma=Error,
                         absolute_sigma=True)  # Call linear best fit function
pvar1 = np.diag(pcov1)

# # plot g_delta vs. r_delta
#
plt.figure(figsize=(10, 10))
plt.plot(floor, model(floor, *popt1), c='r', label='fit')
plt.errorbar(floor, g_delta, yerr=Error, fmt='.', label='Original data')
plt.legend()
plt.xlabel('Floor number')
plt.ylabel('Rescaled Gravitimeter reading (m/s^2)')
plt.show()

R1 = radius(popt1[0] / 10000)  # in m
R1U = np.abs(((np.sqrt(pvar1[0]) / popt1[0])) * R1)  # unceratanty calculation
print('Accomodation for floor mass uncertainty the first days measurement is:', R1, '±', R1U)
print('Difference is ', r_earth - R1, 'm')
print(*popt1)
result = []
ys = np.zeros(len(g_delta))
for i in range(0, len(g_delta)):
    ys[i] = model(floor[i], *popt1)
v = len(floor) - len(popt1)
r2 = chi_square(ys, g_delta, Error)
print('Reduced Chi-Square: ', r2 / v)
#
# popt, pcov = curve_fit(model, range(2, 14), data.groupby('floor').mean().dif, sigma=data.groupby('floor').std().dif,
#                        absolute_sigma=True)
# pvar = np.diag(pcov)
# plt.figure(figsize=(10, 10))
# plt.plot(data.floor, model(data.floor, *popt), c='r', label='fit')
# plt.errorbar(floor, g_change, yerr=err, fmt='.', label='Original data')
# plt.xlabel("Floor")
# plt.ylabel("Change in Gravimeter reading (m/s^2)")
# plt.legend(loc='best')
# plt.show()

# #Getting a new data set of the averages and setting error to standard deviation
# gal_mean = np.empty(11,)
# gal_std = np.empty(11,)
# for i in range(len(gal_mean)):
#     a = np.array([gal1[i]])
#     gal_mean[i] = np.mean(a)
#     gal_std[i] = np.std(a)
#
# g = 9.81 - gal_mean/100000
# delg = np.empty(11,)
# for i in range(len(delg)):
#     delg[i] = g[i] - g[0]
# g_error = 1000*gal_std * k/100000
# print(delg)
# print(g_error)


data = pd.read_csv('data2.csv')


# Converts from Div to Gal, cm/s^2
gal1 = data['value'] * k * 1000

# Converts from Gal to m/^2
g_delta2 = gal1 / 10000

Error = np.ones(len(g_delta2)) * k * np.std(g_delta2)

popt2, pcov2 = curve_fit(model, floor, g_delta2, p0=[0, 0], sigma=Error,
                         absolute_sigma=True)  # Call linear best fit function
pvar2 = np.diag(pcov2)

# # plot g_delta vs. r_delta
#
plt.figure(figsize=(10, 10))
plt.plot(floor, model(floor, *popt2), c='r', label='fit')
plt.errorbar(floor, g_delta2, yerr=Error, fmt='.', label='Original data')
plt.legend()
plt.xlabel('Floor number')
plt.ylabel('Rescaled Gravitimeter reading (m/s^2)')
plt.show()

R1 = radius(popt2[0] / 10000)  # in m
R1U = np.abs(((np.sqrt(pvar2[0]) / popt2[0])) * R1)  # unceratanty calculation

print('The Radius of the Earth from the Second day measurement is:', R1, '±', R1U)
print('Difference is ', r_earth - R1, 'm')
print(*popt2, *pvar2)


# accomodate error from floor mass

floor_delta = abs(G * (m_earth + 10e6 * (15 - floor)) / (r_earth + r_delta * (15 - floor))**2 - G * m_earth / r_earth**2)
g_delta2 -= floor_delta

popt2, pcov2 = curve_fit(model, floor, g_delta2, p0=[0, 0], sigma=Error,
                         absolute_sigma=True)  # Call linear best fit function
pvar2 = np.diag(pcov2)

# # plot g_delta vs. r_delta
#
plt.figure(figsize=(10, 10))
plt.plot(floor, model(floor, *popt2), c='r', label='fit')
plt.errorbar(floor, g_delta2, yerr=Error, fmt='.', label='Original data')
plt.legend()
plt.xlabel('Floor number')
plt.ylabel('Rescaled Gravitimeter reading (m/s^2)')
plt.show()

R1 = radius(popt2[0] / 10000)  # in m
R1U = np.abs(((np.sqrt(pvar2[0]) / popt2[0])) * R1)  # unceratanty calculation
print('Accomodation for floor mass uncertainty Second day measurement is:', R1, '±', R1U)
print('Difference is ', r_earth - R1, 'm')
print(*popt2)
result = []
ys = np.zeros(len(g_delta2))
for i in range(0, len(g_delta2)):
    ys[i] = model(floor[i], *popt2)
v = len(floor) - len(popt2)
r2 = chi_square(ys, g_delta2, Error)
print('Reduced Chi-Square: ', r2 / v)


data = pd.read_csv('data3.csv')


# Converts from Div to Gal, cm/s^2
gal1 = data['value'] * k * 1000

# Converts from Gal to m/^2
g_delta3 = gal1 / 10000

Error = np.ones(len(g_delta3)) * k * np.std(g_delta3)

popt3, pcov3 = curve_fit(model, floor, g_delta3, p0=[0, 0], sigma=Error,
                         absolute_sigma=True)  # Call linear best fit function
pvar3 = np.diag(pcov1)

# # plot g_delta vs. r_delta
#
plt.figure(figsize=(10, 10))
plt.plot(floor, model(floor, *popt3), c='r', label='fit')
plt.errorbar(floor, g_delta3, yerr=Error, fmt='.', label='Original data')
plt.legend()
plt.xlabel('Floor number')
plt.ylabel('Rescaled Gravitimeter reading (m/s^2)')
plt.show()

R1 = radius(popt3[0] / 10000)  # in m
R1U = np.abs(((np.sqrt(pvar3[0]) / popt3[0])) * R1)  # unceratanty calculation

print('The Radius of the Earth from the Second day measurement is:', R1, '±', R1U)
print('Difference is ', r_earth - R1, 'm')
print(*popt3, pvar3)


# accomodate error from floor mass

floor_delta = abs(G * (m_earth + 10e6 * (floor)) / (r_earth + r_delta * (floor))**2 - G * m_earth / r_earth**2)
g_delta3 -= floor_delta
Error = np.ones(len(g_delta3)) * k * np.std(g_delta3)
popt3, pcov3 = curve_fit(model, floor, g_delta3, p0=[0, 0], sigma=Error,
                         absolute_sigma=True)  # Call linear best fit function
pvar3 = np.diag(pcov3)
# # plot g_delta vs. r_delta
#
plt.figure(figsize=(10, 10))
plt.plot(floor, model(floor, *popt3), c='r', label='fit')
plt.errorbar(floor, g_delta3, yerr=Error, fmt='.', label='Original data')
plt.legend()
plt.xlabel('Floor number')
plt.ylabel('Rescaled Gravitimeter reading (m/s^2)')
plt.show()

R1 = radius(popt3[0] / 10000)  # in m
R1U = np.abs(((np.sqrt(pvar3[0]) / popt3[0])) * R1)  # unceratanty calculation
print('Accomodation for floor mass uncertainty Second day measurement is:', R1, '±', R1U)
print('Difference is ', r_earth - R1, 'm')
print(*popt3)
result = []
ys = np.zeros(len(g_delta3))
for i in range(0, len(g_delta3)):
    ys[i] = model(floor[i], *popt3)
v = len(floor) - len(popt3)
r2 = chi_square(ys, g_delta3, Error)
print('Reduced Chi-Square: ', r2 / v)


data = pd.read_csv('data4.csv')


# Converts from Div to Gal, cm/s^2
gal1 = data['value'] * k * 1000

# Converts from Gal to m/^2
g_delta4 = gal1 / 10000

Error = np.ones(len(g_delta3)) * k * np.std(g_delta3)

popt4, pcov4 = curve_fit(model, floor, g_delta4, p0=[0, 0], sigma=Error,
                         absolute_sigma=True)  # Call linear best fit function
pvar4 = np.diag(pcov1)

# # plot g_delta vs. r_delta
#
plt.figure(figsize=(10, 10))
plt.plot(floor, model(floor, *popt4), c='r', label='fit')
plt.errorbar(floor, g_delta4, yerr=Error, fmt='.', label='Original data')
plt.legend()
plt.xlabel('Floor number')
plt.ylabel('Rescaled Gravitimeter reading (m/s^2)')
plt.show()

R1 = radius(popt4[0] / 10000)  # in m
R1U = np.abs(((np.sqrt(pvar4[0]) / popt4[0])) * R1)  # unceratanty calculation

print('The Radius of the Earth from the Second day measurement is:', R1, '±', R1U)
print('Difference is ', r_earth - R1, 'm')
print(*popt4,pvar4)


# accomodate error from floor mass

floor_delta = abs(G * (m_earth + 10e6 * (floor)) / (r_earth + r_delta * (floor))**2 - G * m_earth / r_earth**2)
print(floor_delta)
g_delta4 -= floor_delta
Error = np.ones(len(g_delta4)) * k * np.std(g_delta4)
popt4, pcov4 = curve_fit(model, floor, g_delta4, p0=[0, 0], sigma=Error,
                         absolute_sigma=True)  # Call linear best fit function
pvar4 = np.diag(pcov4)
# # plot g_delta vs. r_delta
#
plt.figure(figsize=(10, 10))
plt.plot(floor, model(floor, *popt4), c='r', label='fit')
plt.errorbar(floor, g_delta4, yerr=Error, fmt='.', label='Original data')
plt.legend()
plt.xlabel('Floor number')
plt.ylabel('Rescaled Gravitimeter reading (m/s^2)')
plt.show()

R1 = radius(popt4[0] / 10000)  # in m
R1U = np.abs(((np.sqrt(pvar4[0]) / popt4[0])) * R1)  # unceratanty calculation
print('Accomodation for floor mass uncertainty Second day measurement is:', R1, '±', R1U)
print('Difference is ', r_earth - R1, 'm')
print(*popt4)
result = []
ys = np.zeros(len(g_delta4))
for i in range(0, len(g_delta4)):
    ys[i] = model(floor[i], *popt4)
v = len(floor) - len(popt4)
r2 = chi_square(ys, g_delta4, Error)
print('Reduced Chi-Square: ', r2 / v)


avg = (g_delta + g_delta2 + g_delta3 + g_delta4)/4
Error = np.ones(len(avg)) * k * np.std(avg)
popt5, pcov5 = curve_fit(model, floor, avg, p0=[0, 0], sigma=Error,
                         absolute_sigma=True)  # Call linear best fit function
pvar5 = np.diag(pcov4)
plt.figure(figsize=(7, 7))
# plt.plot(floor, model(floor, *popt1), c='b', label='fit-week1')
plt.errorbar(floor, g_delta, yerr=Error, fmt='.', label='Week1 data')
# plt.plot(floor, model(floor, *popt2), c='r', label='fit-week2')
plt.errorbar(floor, g_delta2, yerr=Error, fmt='.', label='Week2 data')
# plt.plot(floor, model(floor, *popt3), c='g', label='fit-week3')
plt.errorbar(floor, g_delta3, yerr=Error, fmt='.', label='Week3 data')
# plt.plot(floor, model(floor, *popt4), c='g', label='fit-week4')
plt.errorbar(floor, g_delta4, yerr=Error, fmt='.', label='Week4 data')
plt.plot(floor, model(floor, *popt5), c='b', label='fit-avg')
plt.errorbar(floor, avg, yerr=Error, fmt='.', label='Avg data')
plt.legend()
plt.xlabel('Floor number')
plt.ylabel('Rescaled Gravitimeter reading (m/s^2)')
plt.title('Rescaled Gravitimeter reading v.s. floor number')
plt.show()

R1 = radius(popt5[0] / 10000)  # in m
R1U = np.abs(((np.sqrt(pvar5[0]) / popt5[0])) * R1)  # unceratanty calculation
print('Accomodation for floor mass uncertainty Second day measurement is:', R1, '±', R1U)
print('Difference is ', r_earth - R1, 'm')
print(*popt5)
result = []
ys = np.zeros(len(avg))
for i in range(0, len(avg)):
    ys[i] = model(floor[i], *popt5)
v = len(floor) - len(popt5)
r2 = chi_square(ys, avg, Error)
print('Reduced Chi-Square: ', r2 / v)
