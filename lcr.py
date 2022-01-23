import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd


def rcs(pred, target, uncertainty, n_params):
    return np.square((pred - target) / uncertainty).sum() / (pred.size - n_params)


def rc(x, r, c):
    return np.exp(-x / (r * c))


def rc_(x, tao):
    return np.exp(-x / tao)


# noinspection PyShadowingNames
def lr(x, R, L):
    return np.exp(-x * R / L)


# RC circuit on battery


data = pd.read_csv(r"C:\Users\Walter\IdeaProjects\PHY224\LCR\exp1RC.csv", delimiter=',', skiprows=1)
data.plot(x='second')
plt.title('RC circuit on Battery')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
# choose a segment that contains one decay
data_seg = data[:1880]
xdata = data_seg['second'] - data_seg['second'].iloc[0]  # set the beginning of timeline to 0
ydata = data_seg['Volt'] / data_seg['Volt'].iloc[0]  # set it as a ratio over the initial value

sigma_y = np.ones_like(ydata) * 0.000001  # error
sigma_x = np.ones_like(xdata) * 0.000001

p_opt, p_cov = curve_fit(rc, xdata, ydata, (1e5, 2.2e-8), sigma_y, True)
fig, ax = plt.subplots()
ax.plot(xdata, rc(xdata, *p_opt), 'r-', label='fit: R=%.3e, C=%.3e' % tuple(p_opt))
ax.errorbar(xdata, ydata, yerr=sigma_y, xerr=sigma_x, label="error", linestyle='None', marker=".")
ax.set_title('RC circuit')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (ratio)')
ax.legend(loc="best")
time_constant = p_opt[0] * p_opt[1]
time_constant_sd = time_constant * np.sqrt((p_cov[0, 0] / p_opt[0]) ** 2 + (p_cov[1, 1] / p_opt[1]) ** 2)
print(
    "[RC] The time constant estimated is {:.3e} with standard deviation {:.3e}".format(time_constant,

                                                                                       time_constant_sd))
# extreme high reduced chi square value due to small uncertainties
print(rcs(rc(xdata, *p_opt), ydata, sigma_y, 1))
# LR circuit on wave gen

data = pd.read_csv(r"C:\Users\Walter\IdeaProjects\PHY224\LCR\exp1LR.csv", delimiter=',', skiprows=1)
data.plot(x='second')
plt.title('LR circuit on Wavegen')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')

# choose a segment that contains one decay
data_seg = data[1000:1800]
y_corr = abs(min(data_seg['Volt'])) + data_seg['Volt']
xdata = data_seg['second'] - data_seg['second'].iloc[0]  # set the beginning of timeline to 0
ydata = y_corr / y_corr.iloc[1]  # set it as a ratio over the initial value

sigma_y = np.ones_like(ydata) * 0.000001  # error
sigma_x = np.ones_like(xdata) * 0.000001

p_opt, p_cov = curve_fit(lr, xdata, ydata, (1, 0.0426), sigma_y, True)

# noinspection PyRedeclaration
fig, ax = plt.subplots()
ax.plot(xdata, lr(xdata, *p_opt), 'r-',
        label='fit: R=%.3e, L=%.3e' % tuple(p_opt))
ax.errorbar(xdata, ydata, yerr=sigma_y, xerr=sigma_x, label="error", linestyle='None', marker=".")
ax.set_title('LR circuit')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (ratio)')
ax.legend()

# extreme high reduced chi square value due to small uncertainties
print(rcs(lr(xdata, *p_opt), ydata, sigma_y, 1))
# LR circuit on wave gen

time_constant = p_opt[1] / p_opt[0]
time_constant_sd = time_constant * np.sqrt((p_cov[0, 0] / p_opt[0]) ** 2 + (p_cov[1, 1] / p_opt[1]) ** 2)
print("[LR] The time constant estimated is {:.3e} with standard deviation {:.3e}".format(time_constant,
                                                                                         time_constant_sd))

# LC circuit on wave gen, VL measured

# Theoretical frequency is:
# 1/(2*np.pi*np.sqrt(22e-9*42.6e-3))
# =5198.8105

data = pd.read_csv(r"C:\Users\Walter\IdeaProjects\PHY224\LCR\exp1LC.csv", delimiter=',', skiprows=1)
data.plot(x='second')
plt.title("LC circuit")
plt.ylabel("Voltage (V)")
plt.xlabel('Time (s)')
plt.show()

################# experiment 2 #############

data = pd.read_csv(r"C:\Users\Walter\IdeaProjects\PHY224\LCR\scope_5.csv", delimiter=',', skiprows=1)
data.plot(x='second')
plt.title("RC circuit")
plt.ylabel("Voltage (V)")
plt.xlabel('Time (s)')
plt.show()
# impedance RC
R = 10000
C = 2.2e-8

data = pd.read_csv(r"C:\Users\Walter\IdeaProjects\PHY224\LCR\exp2RC.csv", delimiter=',', skiprows=1)
plt.semilogx(data["freq"], data["phase"])
plt.title("RC circuit semi log plot")
plt.ylabel("Phases (degree)")
plt.xlabel('log Frequency (log(Hz))')

data['angular_freq'] = 2 * np.pi * data['freq']

# TODO: Check formula Z_RC
data['Z_RC'] = data.V1 / data.V2 * R + R
# noinspection PyRedeclaration
fig, ax = plt.subplots()
ax.plot(data['angular_freq'], data['Z_RC'], label="Observed")

freq = np.linspace(100, 10000, 100)
angular_freq = 2 * np.pi * freq
Z_RC = np.sqrt(R ** 2 + (1 / (angular_freq * C)) ** 2)
ax.plot(angular_freq, Z_RC, label="Theoretical")
ax.legend()
ax.set_xlabel('Angular Frequency (rad/s)')
ax.set_ylabel('Z')
plt.show()

sub_data = data[:]


# noinspection PyShadowingNames
def impedance_rc(angular_freq, C):
    R = 10000
    Z_RC = np.sqrt(R ** 2 + (1 / (angular_freq * C)) ** 2)
    return Z_RC


xdata = sub_data['angular_freq']
ydata = sub_data['Z_RC']
reading_error = 0.03  #error
sigma_x = np.ones_like(xdata) * 0.000001
sigma_y = ydata * np.sqrt(
    (sub_data['V1'] * reading_error / sub_data['V1']) ** 2 + (sub_data['V2'] * reading_error / sub_data['V2']) ** 2)

p_opt, p_cov = curve_fit(impedance_rc, xdata, ydata, 2.2e-8, sigma_y, True)

# noinspection PyRedeclaration
fig, ax = plt.subplots()
ax.plot(xdata, impedance_rc(xdata, *p_opt), 'r-',
        label='fit: C=%.3e (F)' % tuple(p_opt))
ax.errorbar(xdata, ydata, yerr=sigma_y, label="error", linestyle='None', marker=".")
ax.set_title('RC circuit fit')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Z')
ax.legend()
plt.show()
print("The capacitance estimated is {:.3e} (F) with standard deviation {:.3e}".format(p_opt[0], p_cov[0, 0]))

# extreme high reduced chi square value due to small uncertainties
print(rcs(impedance_rc(xdata, *p_opt), ydata, sigma_y, 1))
# LR circuit on wave gen

# impedance RL

# theoretical curve
R = 512.4
L = 4.26e-2
freq = np.linspace(1, 25000, 100)
angular_freq = 2 * np.pi * freq
Z_RL = np.sqrt(R ** 2 + (angular_freq * L) ** 2)

# actual curve
data = pd.read_csv(r"C:\Users\Walter\IdeaProjects\PHY224\LCR\exp2LR(V_Vr).csv", delimiter=',',
                   skiprows=1)

plt.semilogx(data["freq"], data["phase"])
plt.title("RL circuit semi log plot")
plt.ylabel("Phases (degree)")
plt.xlabel('log Frequency (log(Hz))')
data['angular_freq'] = 2 * np.pi * data['freq']

# TODO: Check formula Z_RL
data['Z_RL'] = data.V1 / data.V2 * R
# noinspection PyRedeclaration
fig, ax = plt.subplots()
ax.plot(data['angular_freq'], data['Z_RL'], label="observed")

ax.plot(angular_freq, Z_RL, label="theoretical")
ax.legend()
ax.set_xlabel('Angular Frequency (rad/s)')
ax.set_ylabel('Z')
plt.show()

sub_data = data[0:6]


# noinspection PyShadowingNames
def impedance_rl(angular_freq, L):
    R = 512.4
    Z_RL = np.sqrt(R ** 2 + (angular_freq * L) ** 2)
    return Z_RL


xdata = sub_data['angular_freq']
ydata = sub_data['Z_RL']
reading_error = 0.03  # the error
# noinspection PyRedeclaration
sigma_x = np.ones_like(xdata) * 0.000001
sigma_y = ydata * np.sqrt(
    (sub_data['V1'] * reading_error / sub_data['V1']) ** 2 + (sub_data['V2'] * reading_error / sub_data['V2']) ** 2)

p_opt, p_cov = curve_fit(impedance_rl, xdata, ydata, 0.04, sigma_y, True)

# noinspection PyRedeclaration
fig, ax = plt.subplots()
ax.plot(xdata, impedance_rl(xdata, *p_opt), 'r-',
        label='fit: L=%.3e (H)' % tuple(p_opt))
ax.errorbar(xdata, ydata, yerr=sigma_y, label="error", linestyle='None', marker=".")
ax.set_title('RL circuit fit')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Z')
ax.legend()
plt.show()
print("The Inductance estimated is {:.3e} (H) with standard deviation {:.3e}".format(p_opt[0], p_cov[0, 0]))
# extreme high reduced chi square value due to small uncertainties
print(rcs(impedance_rl(xdata, *p_opt), ydata, sigma_y, 1))
# LR circuit on wave gen

# impedance RCL

# theoretical curve
R = 512.4
C = 2.2e-8
L = 4.26e-2
freq = np.linspace(100, 25000, 1000)
angular_freq = 2 * np.pi * freq
Z_RCL = np.sqrt(R ** 2 + (angular_freq * L - 1 / (angular_freq * C)) ** 2)

# actual curve
data = pd.read_csv(r"C:\Users\Walter\IdeaProjects\PHY224\LCR\exp2LR(V_Vr).csv", delimiter=',',
                   skiprows=1)

plt.semilogx(data["freq"], data["phase"])
plt.title("RCL circuit semi log plot")
plt.ylabel("Phases (degree)")
plt.xlabel('log Frequency (log(Hz))')
data['angular_freq'] = 2 * np.pi * data['freq']
# TODO: Check formula Z_RCL
data['Z_RCL'] = data.V1 / data.V2 * R + R
# noinspection PyRedeclaration
fig, ax = plt.subplots()
ax.plot(data['angular_freq'], data['Z_RCL'], label="observed")
ax.plot(angular_freq, Z_RCL, label="theoretical")
ax.legend()
ax.set_xlabel('Angular Frequency (rad/s)')
ax.set_ylabel('Z')
plt.show()

sub_data = data[0:6]


# noinspection PyShadowingNames
def impedance_rcl(angular_freq, L, C):
    R = 512.4
    Z_RCL = np.sqrt(R ** 2 + ((angular_freq * L) + (1 / (angular_freq * C))) ** 2)
    return Z_RCL


# problematic

xdata = sub_data['angular_freq']
ydata = sub_data['Z_RCL']
reading_error = 0.03
sigma_y = ydata * np.sqrt(
    (sub_data['V1'] * reading_error / sub_data['V1']) ** 2 + (sub_data['V2'] * reading_error / sub_data['V2']) ** 2)
p0 = [0.04, 0.04]
sigmas = np.array(sigma_x, sigma_y)
p_opt, p_cov = curve_fit(impedance_rcl, xdata, ydata, p0, sigmas, True)

# noinspection PyRedeclaration
fig, ax = plt.subplots()
ax.plot(xdata, impedance_rcl(xdata, *p_opt), 'r-',
        label='fit: L=%.3e (H), C=%.3e (F)' % tuple(p_opt))
ax.errorbar(xdata, ydata, yerr=sigma_y, label="error", linestyle='None', marker=".")
ax.set_title('RCL circuit fit')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Z')
ax.legend()
plt.show()
print("The Inductance estimated is {:.3e} (H) with standard deviation {:.3e}".format(p_opt[0], p_cov[0, 0]))
print("The Capacitance estimated is {:.3e} (F) with standard deviation {:.3e}".format(p_opt[1], p_cov[1, 1]))
# extreme high reduced chi square value due to small uncertainties
print(rcs(impedance_rcl(xdata, *p_opt), ydata, sigma_y, 1))
# LR circuit on wave gen
