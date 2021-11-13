import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import os
from uncertainties import ufloat


def v_corr(v: float, d: float, D: float):
    return v / (1 - 2.104 * (d / D) + 2.089 * (d / D) ** 2)


def re(p: float, l: float, v: float, n: float):
    return p * l * v / n


def model(x, a, b):
    return a * x ** b


# eq. 12 for water
def model1(x, a):
    return a * x ** 0.5


# color function for the figure
def get_colors(palette, n):
    col = []
    cmap = matplotlib.cm.get_cmap(palette, n)
    for i in range(cmap.N):
        rgb = cmap(i)[:3]
        col.append(matplotlib.colors.rgb2hex(rgb))
    return col


def rcs(pred, target, uncertainty, n_params):
    return np.square((pred - target) / uncertainty).sum() / (pred.size - n_params)


diameter = [2.2, 3.0, 4.0, 4.9, 6.2]
radius = [1.1, 1.5, 2.0, 2.45, 3.1]
mean_v = [0] * 5
color = get_colors('Accent', 5)
path = 'C:\\Users\\Walter\\IdeaProjects\\PHY224\\fluid'

for size in sorted(os.listdir(path)):
    avg_v = []
    for trial in os.listdir(f'{path}\\{size}'):
        if trial.endswith('.txt'):
            time, pos = np.loadtxt(f'{path}\\{size}\\{trial}', unpack=True, skiprows=2)
            # Add uncertainty
            time = np.array([ufloat(i, 0.0005) for i in time.tolist()])
            pos = np.array([ufloat(j, 0.001) for j in pos.tolist()])
            time, pos = time[pos > 1e-6], pos[pos > 1e-6]
            vel = np.diff(pos)[10:]
            plt.plot([k.nominal_value for k in vel], c=color[int(size) - 1])
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity (cm/s)')
            plt.title('Velocities vs. Time plot \n for different sizes of Teflon beads in water')
            avg_v.append(vel.mean())  # mean velocity for each trial
    plt.plot([], [], c=color[int(size) - 1], label=f'r={radius[(int(size) - 1)]:.3f} mm')  # grouped labels
    plt.legend(loc='best')
    avg_v = np.array(avg_v)
    mean_v[int(size) - 1] = avg_v.mean()  # mean v
plt.show()
print("raw velocities")
for v in mean_v:
    print(v)

print("corrected velocities")
corr_v = [(v_corr(v / 100, d / 1000, 0.095) * 100) for d, v in zip(diameter, mean_v)]
for v in corr_v:
    print(v)

print("Reynolds number")
ren = [(re(1, v, d / 10, 0.01)) for d, v in zip(diameter, corr_v)]
for r in ren:
    print(r)

popt, pcov = curve_fit(model, radius, [i.nominal_value for i in corr_v], sigma=[j.std_dev for j in corr_v],
                       absolute_sigma=True)
popt2, pcov2 = curve_fit(model1, radius, [i.nominal_value for i in corr_v], sigma=[j.std_dev for j in corr_v],
                         absolute_sigma=True)
print(popt)
print(popt2)
xdat = np.linspace(min(radius) * 0.9, max(radius) * 1.05)
plt.scatter(radius, [i.nominal_value for i in corr_v], label='data')
plt.plot(xdat, model(xdat, *popt), c='b', label=f'fit with y={popt[0]:.1f}x^{popt[1]:.2f}')
plt.plot(xdat, model1(xdat, *popt2), c='r', label=f'fit with eq. 18 (y={popt2[0]:.1f}x^2)')
plt.xlabel('radius (mm)')
plt.ylabel('mean terminal velocity (cm/s), corrected for wall effect')
plt.title(f'radius vs terminal velocity (corrected) for water')
plt.legend(loc='best')
plt.show()
radius = np.array(radius)

# extreme high reduced chi square value due to small uncertainties
print(rcs(model1(radius, *popt2), [i.nominal_value for i in corr_v], [j.std_dev for j in corr_v], 1))
