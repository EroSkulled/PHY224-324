import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


with open('calibration_p3.pkl', 'rb') as file:
    data_from_file = pickle.load(file)
xx = np.linspace(0, 4095 / 1e3, 4096)
plt.figure(figsize=(10, 5))
plt.plot(xx, data_from_file['evt_2'], color='k', lw=0.5)
plt.ylabel('Readout Voltage (V)')
plt.xlabel('Time (ms)')
plt.show()

# with open('signal_p3.pkl', 'rb') as file:
#     data_from_file=pickle.load(file)
# plt.plot(xx, data_from_file['evt_2'], color='k', lw=0.5)
# plt.ylabel('Readout Voltage (V)')
# plt.xlabel('Time (ms)')
# plt.show()


y = []
for key in data_from_file:
    y.append(max(data_from_file[key]) * 1000)
sd = np.std(y)

def energy_plot(ydata, ca_factor):
    fig, ax = plt.subplots()
    ydata = ydata * ca_factor
    bin_heights, bin_borders, _ = plt.hist(ydata, bins='auto', histtype='step', color='black')
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    popt, pcov = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
    ax.plot(bin_centers, gaussian(bin_centers, *popt), 'r-', label='fit')
    ax.errorbar(bin_centers, bin_heights, yerr=sd, marker='.', drawstyle='steps-mid', label='Data', color='black')
    err = np.ones(len(bin_centers)) * sd
    rr = chi_square(gaussian(bin_centers, *popt), bin_heights, err) / (len(bin_centers) - 1)
    prob = 1 - stats.chi2.cdf(rr, len(bin_centers) - 1)
    textstr = '\n'.join((
        r'$\mu=%.2f$ keV' % (popt[0],),
        r'$\sigma=%.2f$ keV' % (popt[2],),
        r'$Amplitude=%.2f$' % (popt[1],),
        r'$X^2/DOF=%.2f$' % (rr,),
        r'$X^2prob.=%.2f$' % (prob,),))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Events / 0.05keV')
    plt.legend(loc='best')
    plt.show()


def energy(t, A):
    C = (0.00008 / 0.00002) ** (-(0.00002 / (0.00008 - 0.00002))) * ((0.00002 - 0.00008) / 0.00008)
    return A * C * (np.e**(-t/0.00002) - np.e**(-t/0.00008))


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- (x - mean) ** 2 / (2 * standard_deviation ** 2))


def chi_square(ys, fs, sigma):
    chi_sqr = 0
    for i in range(0, len(ys)):
        chi_sqr += ((ys[i] - fs[i]) / sigma[i]) ** 2
    chi_sqr = (1/(len(ys) - 2))*chi_sqr
    return chi_sqr


fig, ax = plt.subplots()
x = np.random.normal(min(y), max(y), size=1000)
ydata = np.array(y)


bin_heights, bin_borders, _ = plt.hist(ydata, bins='auto', histtype='step', color='black')
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
popt, pcov = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
ax.plot(bin_centers, gaussian(bin_centers, *popt), 'r-', label='fit')
ax.errorbar(bin_centers, bin_heights, yerr=sd, marker='.', drawstyle='steps-mid', label='Data', color='black')
err = np.ones(len(bin_centers)) * sd
rr = chi_square(gaussian(bin_centers, *popt), bin_heights, err) / (len(bin_centers) - 1)
prob = 1 - stats.chi2.cdf(rr, len(bin_centers) - 1)
textstr = '\n'.join((
    r'$\mu=%.2f$ mV' % (popt[0],),
    r'$\sigma=%.2f$ mV' % (popt[2],),
    r'$Amplitude=%.2f$' % (popt[1],),
    r'$X^2/DOF=%.2f$' % (rr,),
    r'$X^2prob.=%.2f$' % (prob,),))
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top')
plt.xlabel('Amplitude (mV)')
plt.ylabel('Events / 0.01mV')
plt.legend(loc='best')
plt.show()
ca_factor = 10/popt[0]
print('Calibration factor: ', ca_factor)
res = ca_factor*popt[2]
print('Resolution: ', res)
energy_plot(ydata, ca_factor)







# max-min
y = []
for key in data_from_file:
    y.append((max(data_from_file[key]) - min(data_from_file[key])) * 1000)

ydata = np.array(y)

fig, ax = plt.subplots()
bin_heights, bin_borders, _ = plt.hist(ydata, bins='auto', histtype='step', color='black')
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
popt, pcov = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
ax.plot(bin_centers, gaussian(bin_centers, *popt), 'r-', label='fit')
ax.errorbar(bin_centers, bin_heights, yerr=sd, marker='.', drawstyle='steps-mid', label='Data', color='black')
err = np.ones(len(bin_centers)) * sd
rr = chi_square(gaussian(bin_centers, *popt), bin_heights, err) / (len(bin_centers) - 1)
prob = 1 - stats.chi2.cdf(rr, len(bin_centers) - 1)
textstr = '\n'.join((
    r'$\mu=%.2f$ mV' % (popt[0],),
    r'$\sigma=%.2f$ mV' % (popt[2],),
    r'$Amplitude=%.2f$' % (popt[1],),
    r'$X^2/DOF=%.2f$' % (rr,),
    r'$X^2prob.=%.2f$' % (prob,),))
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top')
plt.xlabel('Amplitude (mV)')
plt.ylabel('Events / 0.01mV')
plt.legend(loc='best')
plt.show()
ca_factor = 10/popt[0]
print('Calibration factor: ', ca_factor)
res = ca_factor*popt[2]
print('Resolution: ', res)
energy_plot(ydata, ca_factor)

# max-baseline
y = []
for key in data_from_file:
    y.append((max(data_from_file[key]) - np.average(data_from_file[key][:999])) * 1000)
ydata = np.array(y)

fig, ax = plt.subplots()
bin_heights, bin_borders, _ = plt.hist(ydata, bins='auto', histtype='step', color='black')
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
popt, pcov = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
ax.plot(bin_centers, gaussian(bin_centers, *popt), 'r-', label='fit')
ax.errorbar(bin_centers, bin_heights, yerr=sd, marker='.', drawstyle='steps-mid', label='Data', color='black')
err = np.ones(len(bin_centers)) * sd
rr = chi_square(gaussian(bin_centers, *popt), bin_heights, err) / (len(bin_centers) - 1)
prob = 1 - stats.chi2.cdf(rr, len(bin_centers) - 1)
textstr = '\n'.join((
    r'$\mu=%.2f$ mV' % (popt[0],),
    r'$\sigma=%.2f$ mV' % (popt[2],),
    r'$Amplitude=%.2f$' % (popt[1],),
    r'$X^2/DOF=%.2f$' % (rr,),
    r'$X^2prob.=%.2f$' % (prob,),))
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top')
plt.xlabel('Amplitude (mV)')
plt.ylabel('Events / 0.01mV')
plt.legend(loc='best')
plt.show()
ca_factor = 10/popt[0]
print('Calibration factor: ', ca_factor)
res = ca_factor*popt[2]
print('Resolution: ', res)
energy_plot(ydata, ca_factor)

# sum
y = []
for key in data_from_file:
    y.append(sum(data_from_file[key]))
ydata = np.array(y)

fig, ax = plt.subplots()
bin_heights, bin_borders, _ = plt.hist(ydata, bins='auto', histtype='step', color='black')
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
popt, pcov = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
ax.plot(bin_centers, gaussian(bin_centers, *popt), 'r-', label='fit')
ax.errorbar(bin_centers, bin_heights, yerr=sd, marker='.', drawstyle='steps-mid', label='Data', color='black')
err = np.ones(len(bin_centers)) * sd
rr = chi_square(gaussian(bin_centers, *popt), bin_heights, err) / (len(bin_centers) - 1)
prob = 1 - stats.chi2.cdf(rr, len(bin_centers) - 1)
textstr = '\n'.join((
    r'$\mu=%.2f$ V' % (popt[0],),
    r'$\sigma=%.2f$ V' % (popt[2],),
    r'$Amplitude=%.2f$' % (popt[1],),
    r'$X^2/DOF=%.2f$' % (rr,),
    r'$X^2prob.=%.2f$' % (prob,),))
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top')
plt.xlabel('Sum of Amplitude (V)')
plt.ylabel('Events / 0.01V')
plt.legend(loc='best')
plt.show()
ca_factor = 10/popt[0]
print('Calibration factor: ', ca_factor)
res = ca_factor*popt[2]
print('Resolution: ', res)
energy_plot(ydata, ca_factor)

# sum-baseline
y = []
for key in data_from_file:
    y.append(sum(data_from_file[key]) - np.average(sum(data_from_file[key][:999])))
ydata = np.array(y)

fig, ax = plt.subplots()
bin_heights, bin_borders, _ = plt.hist(ydata, bins='auto', histtype='step', color='black')
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
popt, pcov = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
ax.plot(bin_centers, gaussian(bin_centers, *popt), 'r-', label='fit')
ax.errorbar(bin_centers, bin_heights, yerr=sd, marker='.', drawstyle='steps-mid', label='Data', color='black')
err = np.ones(len(bin_centers)) * sd
rr = chi_square(gaussian(bin_centers, *popt), bin_heights, err) / (len(bin_centers) - 1)
prob = 1 - stats.chi2.cdf(rr, len(bin_centers) - 1)
textstr = '\n'.join((
    r'$\mu=%.2f$ V' % (popt[0],),
    r'$\sigma=%.2f$ V' % (popt[2],),
    r'$Amplitude=%.2f$' % (popt[1],),
    r'$X^2/DOF=%.2f$' % (rr,),
    r'$X^2prob.=%.2f$' % (prob,),))
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top')
plt.xlabel('Sum of Amplitude (V)')
plt.ylabel('Events / 0.01V')
plt.legend(loc='best')
plt.show()
ca_factor = 10/popt[0]
print('Calibration factor: ', ca_factor)
res = ca_factor*popt[2]
print('Resolution: ', res)
energy_plot(ydata, ca_factor)

# sum limited
y = []
for key in data_from_file:
    y.append(sum(data_from_file[key][1000:2000]))
ydata = np.array(y)

fig, ax = plt.subplots()
bin_heights, bin_borders, _ = plt.hist(ydata, bins='auto', histtype='step', color='black')
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
popt, pcov = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])
ax.plot(bin_centers, gaussian(bin_centers, *popt), 'r-', label='fit')
ax.errorbar(bin_centers, bin_heights, yerr=sd, marker='.', drawstyle='steps-mid', label='Data', color='black')
err = np.ones(len(bin_centers)) * sd
rr = chi_square(gaussian(bin_centers, *popt), bin_heights, err) / (len(bin_centers) - 1)
prob = 1 - stats.chi2.cdf(rr, len(bin_centers) - 1)
textstr = '\n'.join((
    r'$\mu=%.2f$ V' % (popt[0],),
    r'$\sigma=%.2f$ V' % (popt[2],),
    r'$Amplitude=%.2f$' % (popt[1],),
    r'$X^2/DOF=%.2f$' % (rr,),
    r'$X^2prob.=%.2f$' % (prob,),))
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top')
plt.xlabel('Sum of Amplitude (V)')
plt.ylabel('Events / 0.01V')
plt.legend(loc='best')
plt.show()
ca_factor = 10/popt[0]
print('Calibration factor: ', ca_factor)
res = ca_factor*popt[2]
print('Resolution: ', res)
energy_plot(ydata, ca_factor)