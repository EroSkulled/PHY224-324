import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def v_corr(v: float, d: float, D: float):
    return v / (1 - 2.104 * (d / D) + 2.089 * (d / D) ** 2)


def re(p: float, l: float, v: float, n: float):
    return p * l * v / n


def model_function(x, a):
    return x ** a


dat = np.loadtxt('../water/1-1.txt', delimiter='\t', dtype=None, encoding=None, skiprows=2)

time1 = dat.transpose()[0][3:85]
dis1 = dat.transpose()[1][3:85]

dat = np.loadtxt('../water/1-2.txt', delimiter='\t', dtype=None, encoding=None, skiprows=2)

time2 = dat.transpose()[0][3:74]
dis2 = dat.transpose()[1][3:74]

dat = np.loadtxt('../water/1-3.txt', delimiter='\t', dtype=None, encoding=None, skiprows=3)
time3 = dat.transpose()[0][:68]
dis3 = dat.transpose()[1][:68]

dat = np.loadtxt('../water/1-4.txt', delimiter='\t', dtype=None, encoding=None, skiprows=3)
time4 = dat.transpose()[0][:71]
dis4 = dat.transpose()[1][:71]

dat = np.loadtxt('../water/1-5.txt', delimiter='\t', dtype=None, encoding=None, skiprows=7)
time5 = dat.transpose()[0][:70]
dis5 = dat.transpose()[1][:70]

dat = np.loadtxt('../water/2-1.txt', delimiter='\t', dtype=None, encoding=None, skiprows=7)
time6 = dat.transpose()[0][:55]
dis6 = dat.transpose()[1][:55]

dat = np.loadtxt('../water/2-2.txt', delimiter='\t', dtype=None, encoding=None, skiprows=3)
time7 = dat.transpose()[0][:58]
dis7 = dat.transpose()[1][:58]

dat = np.loadtxt('../water/2-3.txt', delimiter='\t', dtype=None, encoding=None, skiprows=6)
time8 = dat.transpose()[0][:56]
dis8 = dat.transpose()[1][:56]

dat = np.loadtxt('../water/2-4.txt', delimiter='\t', dtype=None, encoding=None, skiprows=6)
time9 = dat.transpose()[0][:55]
dis9 = dat.transpose()[1][:55]

dat = np.loadtxt('../water/2-5.txt', delimiter='\t', dtype=None, encoding=None, skiprows=5)
time10 = dat.transpose()[0][:47]
dis10 = dat.transpose()[1][:47]

dat = np.loadtxt('../water/3-1.txt', delimiter='\t', dtype=None, encoding=None, skiprows=6)
time11 = dat.transpose()[0][:48]
dis11 = dat.transpose()[1][:48]

dat = np.loadtxt('../water/3-2.txt', delimiter='\t', dtype=None, encoding=None, skiprows=7)
time12 = dat.transpose()[0][:47]
dis12 = dat.transpose()[1][:47]

dat = np.loadtxt('../water/3-3.txt', delimiter='\t', dtype=None, encoding=None, skiprows=5)
time13 = dat.transpose()[0][:49]
dis13 = dat.transpose()[1][:49]

dat = np.loadtxt('../water/3-4.txt', delimiter='\t', dtype=None, encoding=None, skiprows=5)
time14 = dat.transpose()[0][:48]
dis14 = dat.transpose()[1][:48]

dat = np.loadtxt('../water/3-5.txt', delimiter='\t', dtype=None, encoding=None, skiprows=5)
time15 = dat.transpose()[0][:54]
dis15 = dat.transpose()[1][:54]

dat = np.loadtxt('../water/4-1.txt', delimiter='\t', dtype=None, encoding=None, skiprows=9)
time16 = dat.transpose()[0][:42]
dis16 = dat.transpose()[1][:42]

dat = np.loadtxt('../water/4-2.txt', delimiter='\t', dtype=None, encoding=None, skiprows=7)
time17 = dat.transpose()[0][:33]
dis17 = dat.transpose()[1][:33]

dat = np.loadtxt('../water/4-3.txt', delimiter='\t', dtype=None, encoding=None, skiprows=6)
time18 = dat.transpose()[0][:43]
dis18 = dat.transpose()[1][:43]

dat = np.loadtxt('../water/4-4.txt', delimiter='\t', dtype=None, encoding=None, skiprows=7)
time19 = dat.transpose()[0][:41]
dis19 = dat.transpose()[1][:41]

dat = np.loadtxt('../water/4-5.txt', delimiter='\t', dtype=None, encoding=None, skiprows=5)
time20 = dat.transpose()[0][:44]
dis20 = dat.transpose()[1][:44]

dat = np.loadtxt('../water/5-1.txt', delimiter='\t', dtype=None, encoding=None, skiprows=3)
time21 = dat.transpose()[0][:34]
dis21 = dat.transpose()[1][:34]

dat = np.loadtxt('../water/5-2.txt', delimiter='\t', dtype=None, encoding=None, skiprows=5)
time22 = dat.transpose()[0][:36]
dis22 = dat.transpose()[1][:36]

dat = np.loadtxt('../water/5-3.txt', delimiter='\t', dtype=None, encoding=None, skiprows=9)
time23 = dat.transpose()[0][:35]
dis23 = dat.transpose()[1][:35]

dat = np.loadtxt('../water/5-4.txt', delimiter='\t', dtype=None, encoding=None, skiprows=5)
time24 = dat.transpose()[0][:33]
dis24 = dat.transpose()[1][:33]

dat = np.loadtxt('../water/5-5.txt', delimiter='\t', dtype=None, encoding=None, skiprows=4)
time25 = dat.transpose()[0][:34]
dis25 = dat.transpose()[1][:34]

avg_v4 = v_corr(dis4[-1] / (time4[-1] - time4[0]), 2.2, 95)
avg_v5 = v_corr(dis5[-1] / (time5[-1] - time5[0]), 2.2, 95)
avg_v3 = v_corr(dis3[-1] / (time3[-1] - time3[0]), 2.2, 95)
avg_v2 = v_corr(dis2[-1] / (time2[-1] - time2[0]), 2.2, 95)
avg_v1 = v_corr(dis1[-1] / (time1[-1] - time1[0]), 2.2, 95)

avg_1 = (avg_v1 + avg_v2 + avg_v3 + avg_v4 + avg_v5) / 5

avg_v6 = v_corr(dis6[-1] / (time6[-1] - time6[0]), 3, 95)
avg_v7 = v_corr(dis7[-1] / (time7[-1] - time7[0]), 3, 95)
avg_v8 = v_corr(dis8[-1] / (time8[-1] - time8[0]), 3, 95)
avg_v9 = v_corr(dis9[-1] / (time9[-1] - time9[0]), 3, 95)
avg_v10 = v_corr(dis10[-1] / (time10[-1] - time10[0]), 3, 95)

avg_2 = (avg_v6 + avg_v7 + avg_v8 + avg_v9 + avg_v10) / 5

avg_v11 = v_corr(dis11[-1] / (time11[-1] - time11[0]), 4, 95)
avg_v12 = v_corr(dis12[-1] / (time12[-1] - time12[0]), 4, 95)
avg_v13 = v_corr(dis13[-1] / (time13[-1] - time13[0]), 4, 95)
avg_v14 = v_corr(dis14[-1] / (time14[-1] - time14[0]), 4, 95)
avg_v15 = v_corr(dis15[-1] / (time15[-1] - time15[0]), 4, 95)

avg_3 = (avg_v11 + avg_v12 + avg_v13 + avg_v14 + avg_v15) / 5

avg_v16 = v_corr(dis16[-1] / (time16[-1] - time16[0]), 4.9, 95)
avg_v17 = v_corr(dis17[-1] / (time17[-1] - time17[0]), 4.9, 95)
avg_v18 = v_corr(dis18[-1] / (time18[-1] - time18[0]), 4.9, 95)
avg_v19 = v_corr(dis19[-1] / (time19[-1] - time19[0]), 4.9, 95)
avg_v20 = v_corr(dis20[-1] / (time20[-1] - time20[0]), 4.9, 95)

avg_4 = (avg_v16 + avg_v17 + avg_v18 + avg_v19 + avg_v20) / 5

avg_v24 = v_corr(dis24[-1] / (time24[-1] - time24[0]), 6.2, 95)
avg_v25 = v_corr(dis25[-1] / (time25[-1] - time25[0]), 6.2, 95)
avg_v23 = v_corr(dis23[-1] / (time23[-1] - time23[0]), 6.2, 95)
avg_v22 = v_corr(dis22[-1] / (time22[-1] - time22[0]), 6.2, 95)
avg_v21 = v_corr(dis21[-1] / (time21[-1] - time21[0]), 6.2, 95)

avg_5 = (avg_v21 + avg_v22 + avg_v23 + avg_v24 + avg_v25) / 5

r = re(1, 15, avg_5, 0.01)
print("High Re: ", np.round(r, 4))
print(np.round(avg_5, 4), " mm / s")
print("Theoretical terminal velocity is ", np.round(6.2 ** 0.5, 4), " mm / s")
avg = np.array(
    [avg_v1, avg_v2, avg_v3, avg_v4, avg_v5, avg_v6, avg_v7, avg_v8, avg_v9, avg_v10, avg_v11, avg_v12, avg_v13,
     avg_v14, avg_v15, avg_v16, avg_v17, avg_v18, avg_v19, avg_v20, avg_v21, avg_v22, avg_v23, avg_v24, avg_v25])
radius = np.array([2.2, 2.2,2.2, 2.2, 2.2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,4.9, 4.9, 4.9, 4.9, 4.9, 6.2, 6.2, 6.2, 6.2, 6.2])
popt, pcov = curve_fit(model_function, radius, avg)
pstd = np.sqrt(np.diag(pcov))
print(popt, pstd)
plt.scatter(radius, avg, label='Experiment data in Water')
plt.plot(radius, model_function(radius, popt), label='fitted by eq.12')
plt.ylabel('Mean velocity(mm / s)')
plt.xlabel('radius of the sphere(mm)')
plt.legend(loc='best')
plt.title('Velocity vs. Particle radius Plot')
plt.show()
