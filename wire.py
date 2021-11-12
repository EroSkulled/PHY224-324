import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

dat = np.loadtxt('../PHY224/wire.csv', delimiter=',', dtype=None, encoding=None, skiprows=1)
v = dat.transpose()[1]
a = dat.transpose()[2] / 1000
rl = dat.transpose()[3]


vuncertain = v * 0.0025
auncertain = a * 0.0075

vb = v[8:12]
ab = a[8:12]
vbuncertain = vuncertain[8:12]
abuncertain = auncertain[8:12]


vc = v[36:]
ac = a[36:]
vcuncertain = vuncertain[36:]
acuncertain = auncertain[36:]


vd = v[12:16]
ad = a[12:16]
vduncertain = vuncertain[12:16]
aduncertain = auncertain[12:16]


ve = v[24:28]
ae = a[24:28]
veuncertain = vuncertain[24:28]
aeuncertain = auncertain[24:28]

vf = v[16:20] # 15v
af = a[16:20]
vfuncertain = vuncertain[16:20]
afuncertain = auncertain[16:20]

vg = v[28:32]
ag = a[28:32]
vguncertain = vuncertain[28:32]
aguncertain = auncertain[28:32]

vh = v[20:24] # 20v
ah = a[20:24]
vhuncertain = vuncertain[20:24]
ahuncertain = auncertain[20:24]

vi = v[32:36]
ai = a[32:36]
viuncertain = vuncertain[32:36]
aiuncertain = auncertain[32:36]




def model_function(x, a, b):
    return -a * x + b


popt, pcov = curve_fit(model_function, ab, vb, sigma=abuncertain, absolute_sigma=True)
pstd = np.sqrt(np.diag(pcov))
a, b = popt


popt2, pcov2 = curve_fit(model_function, ac, vc, sigma=acuncertain, absolute_sigma=True)
pstd2 = np.sqrt(np.diag(pcov2))
a2, b2 = popt2


print("6 V PSU Option1 output resistance (R_ps): ", np.round(a, 5), " +-", np.round(pstd[0], 5) , " Ohm, open-circuit voltage: ", np.round(b, 5),  " +-", np.round(pstd[1], 5), " V")
print("6 V PSU Option2 output resistance (R_ps): ", np.round(a2, 5), " +-", np.round(pstd2[0], 5) , " Ohm, open-circuit voltage: ", np.round(b2, 5),  " +-", np.round(pstd2[1], 5), " V")


plt.errorbar(ab, vb, yerr=vbuncertain, xerr=abuncertain, label='PSU Option1 data with uncertainty', ls='-')
plt.errorbar(ac, vc, yerr=vcuncertain, xerr=acuncertain, label='PSU Option2 data with uncertainty', ls='-')
plt.xlabel('Current(A)')
plt.ylabel('Voltage(V)')
plt.legend(loc='best')
plt.title('Voltage vs. Current Plot for PSU 6V')
plt.show()

poptd, pcovd = curve_fit(model_function, ad, vd, sigma=aduncertain, absolute_sigma=True)
pstdd = np.sqrt(np.diag(pcovd))
a3, b3 = poptd


popte, pcove = curve_fit(model_function, ae, ve, sigma=aeuncertain, absolute_sigma=True)
pstde = np.sqrt(np.diag(pcove))
a4, b4 = popte
print("10 V PSU Option1 output resistance (R_ps): ", np.round(a3, 5), " +-", np.round(pstdd[0], 5) , " Ohm, open-circuit voltage: ", np.round(b3, 5),  " +-", np.round(pstdd[1], 5), " V")
print("10 V PSU Option2 output resistance (R_ps): ", np.round(a4, 5), " +-", np.round(pstde[0], 5) , " Ohm, open-circuit voltage: ", np.round(b4, 5),  " +-", np.round(pstde[1], 5), " V")


plt.errorbar(ad, vd, yerr=vduncertain, xerr=aduncertain, label='PSU Option1 data with uncertainty', ls='-')
plt.errorbar(ae, ve, yerr=veuncertain, xerr=aeuncertain, label='PSU Option2 data with uncertainty', ls='-')
plt.xlabel('Current(A)')
plt.ylabel('Voltage(V)')
plt.legend(loc='best')
plt.title('Voltage vs. Current Plot for PSU 10V')
plt.show()

poptf, pcovf = curve_fit(model_function, af, vf, sigma=afuncertain, absolute_sigma=True)
pstdf = np.sqrt(np.diag(pcovf))
a5, b5 = poptf


poptg, pcovg = curve_fit(model_function, ag, vg, sigma=aguncertain, absolute_sigma=True)
pstdg = np.sqrt(np.diag(pcovg))
a6, b6 = poptg


print("15 V PSU Option1 output resistance (R_ps): ", np.round(a5, 5), " +-", np.round(pstdf[0], 5) , " Ohm, open-circuit voltage: ", np.round(b5, 5),  " +-", np.round(pstdf[1], 5), " V")
print("15 V PSU Option2 output resistance (R_ps): ", np.round(a6, 5), " +-", np.round(pstdg[0], 5) , " Ohm, open-circuit voltage: ", np.round(b6, 5),  " +-", np.round(pstdg[1], 5), " V")



plt.errorbar(af, vf, yerr=vfuncertain, xerr=afuncertain, label='PSU Option1 data with uncertainty', ls='-')
plt.errorbar(ag, vg, yerr=vguncertain, xerr=aguncertain, label='PSU Option2 data with uncertainty', ls='-')
plt.xlabel('Current(A)')
plt.ylabel('Voltage(V)')
plt.legend(loc='best')
plt.title('Voltage vs. Current Plot for PSU 15V')
plt.show()


popth, pcovh = curve_fit(model_function, ah, vh, sigma=ahuncertain, absolute_sigma=True)
pstdh = np.sqrt(np.diag(pcovh))
a7, b7 = popth


popti, pcovi = curve_fit(model_function, ai, vi, sigma=aiuncertain, absolute_sigma=True)
pstdi = np.sqrt(np.diag(pcovi))
a8, b8 = popti


print("20 V PSU Option1 output resistance (R_ps): ", np.round(a7, 5), " +-", np.round(pstdh[0], 5) , " Ohm, open-circuit voltage: ", np.round(b7, 5),  " +-", np.round(pstdh[1], 5), " V")
print("20 V PSU Option2 output resistance (R_ps): ", np.round(a8, 5), " +-", np.round(pstdi[0], 5) , " Ohm, open-circuit voltage: ", np.round(b8, 5),  " +-", np.round(pstdi[1], 5), " V")



plt.errorbar(ah, vh, yerr=vhuncertain, xerr=ahuncertain, label='PSU Option1 data with uncertainty', ls='-')
plt.errorbar(ai, vi, yerr=viuncertain, xerr=aiuncertain, label='PSU Option2 data with uncertainty', ls='-')
plt.xlabel('Current(A)')
plt.ylabel('Voltage(V)')
plt.legend(loc='best')
plt.title('Voltage vs. Current Plot for PSU 20V')
plt.show()


plt.errorbar(ab, vb, yerr=vbuncertain, xerr=abuncertain, label='PSU Option1 6V', ls='-')
plt.errorbar(ac, vc, yerr=vcuncertain, xerr=acuncertain, label='PSU Option2 6V', ls='-')

plt.errorbar(ad, vd, yerr=vduncertain, xerr=aduncertain, label='PSU Option1 10V', ls='-')
plt.errorbar(ae, ve, yerr=veuncertain, xerr=aeuncertain, label='PSU Option2 10V', ls='-')

plt.errorbar(af, vf, yerr=vfuncertain, xerr=afuncertain, label='PSU Option1 15V', ls='-')
plt.errorbar(ag, vg, yerr=vguncertain, xerr=aguncertain, label='PSU Option2 15V', ls='-')

plt.errorbar(ah, vh, yerr=vhuncertain, xerr=ahuncertain, label='PSU Option1 20V', ls='-')
plt.errorbar(ai, vi, yerr=viuncertain, xerr=aiuncertain, label='PSU Option2 20V', ls='-')
plt.xlabel('Current(A)')
plt.ylabel('Voltage(V)')
plt.legend(loc='best')
plt.title('Voltage vs. Current Plot for all test results')
plt.show()

