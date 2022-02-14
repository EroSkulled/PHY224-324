# -*- coding: utf-8 -*-
"""
Created on Jan 25 2022

@author: Walter
"""

save = True  # if True then we save images as files

from random import gauss
import matplotlib.pyplot as plt
import numpy as np

"""
Ex.1
"""
N = 200  # N is how many data points we will have in our sine wave


def add_func(A1: float, T1: float, A2: float, T2: float):
    # wave1 amplitude,  # wave1 period, # wave2 amplitude,  # wave2 period
    y1 = A1 * np.sin(2. * np.pi * time / T1)
    y2 = A2 * np.sin(2. * np.pi * time / T2)
    return y1 + y2


time = np.arange(N)
y = add_func(5., 16., 8., 12.)
z = np.fft.fft(y)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='col')

# noise_amp=A1/2.
# set the amplitude of the noise relative to sine's amp

"""
i=0
noise=[]
while i < N:
    noise.append(gauss(0,noise_amp))
    i+=1
"""
# noise=[gauss(0,noise_amp) for _usused_variable in range(len(y))]
# this line, and the commented block above, do exactly the same thing

# x=y+noise
# x=y
# y is our pure sine wave, x is y with noise added
#
# z1=np.fft.fft(y)
# z2=np.fft.fft(x)
# take the Fast Fourier Transforms of both x and y

# fig, ( (ax1,ax2), (ax3,ax4) ) = plt.subplots(2,2,sharex='col',sharey='col')
""" 
this setups up a 2x2 array of graphs, based on the first two arguments
of plt.subplots()

the sharex and sharey force the x- and y-axes to be the same for each 
column
"""
# y1 = 5. * np.sin(2. * np.pi * time / 17.)
# y2 = 9. * np.sin(2. * np.pi * time / 12.)
ax1.plot(time / N, y)
ax2.plot(np.abs(z))
# ax1.plot(time / N, y1)
# ax2.plot(time / N, y2)
""" 
our graphs are now plotted

(ax1,ax2) is a list of figures which are the top row of figures

therefore ax1 is top-left and ax2 is top-right

we plot the position-time graphs rescaled by a factor of N so that
the FFT x-axis agrees with the frequency we could measure from the
position-time graph. by default, both graphs use "data-point number"
on their x-axes, so would go 0 to 200 since N=200.
"""

fig.subplots_adjust(hspace=0)
# remove the horizontal space between the top and bottom row
ax1.set_xlabel('Position-Time')
ax2.set_xlabel('Absolute value of FFT of Position-Time\n(Amplitude-Frequency)')
ax1.set_ylim(-15, 15)
ax2.set_ylim(0, 900)
ax1.set_ylabel('2 Sine Wave Added Together\nWith Different Amplitude and Frequency')

mydpi = 300
plt.tight_layout()

if (save): plt.savefig('SingleWaveAndNoiseWithFFT.png', dpi=mydpi)
plt.show()
"""
plt.show() displays the graph on your computer

plt.savefig will save the graph as a .png file, useful for including
in your report so you don't have to cut-and-paste
"""

M = len(z)
freq = np.arange(M)  # frequency values, like time is the time values
width = 8  # width=2*sigma**2 where sigma is the standard deviation
peak = 12.3  # ideal value is approximately N/T1

filter_function = (np.exp(-(freq - peak) ** 2 / width) + np.exp(-(freq + peak - M) ** 2 / width))
z_filtered = z * filter_function
"""
we choose Gaussian filter functions, fairly wide, with
one peak per spike in our FFT graph

we eyeballed the FFT graph to figure out decent values of 
peak and width for our filter function

a larger width value is more forgiving if your peak value
is slightly off

making width a smaller value, and fixing the value of peak,
will give us a better final result
"""

# fig, (ax1,ax2,ax3)=plt.subplots(3,1,sharex='col')
# # this gives us an array of 3 graphs, vertically aligned
# ax1.plot(np.abs(z2))
# ax2.plot(np.abs(filter_function))
# ax3.plot(np.abs(z_filtered))
# """
# note that in general, the fft is a complex function, hence we plot
# the absolute value of it. in our case, the fft is real, but the
# result is both positive and negative, and the absolute value is still
# easier to understand
#
# if we plotted (abs(fft))**2, that would be called the power spectra
# """
#
# fig.subplots_adjust(hspace=0)
# ax1.set_ylim(0,480)
# ax2.set_ylim(0,1.2)
# ax3.set_ylim(0,480)
# ax1.set_ylabel('Noisy FFT')
# ax2.set_ylabel('Filter Function')
# ax3.set_ylabel('Filtered FFT')
# ax3.set_xlabel('Absolute value of FFT of Position-Time\n(Amplitude-Frequency)')
#
# plt.tight_layout()
# """
# the \n in our xlabel does not save to file well without the
# tight_layout() command
# """
#
# if(save): plt.savefig('FilteringProcess.png',dpi=mydpi)
# plt.show()
#
# cleaned=np.fft.ifft(z_filtered)
# """
# ifft is the inverse FFT algorithm
#
# it converts an fft graph back into a sinusoidal graph
#
# we took the data, took the fft, used a filter function
# to eliminate most of the noise, then took the inverse fft
# to get our "cleaned" version of the original data
# """
#
# fig, (ax1,ax2,ax3)=plt.subplots(3,1,sharex='col',sharey='col')
# ax1.plot(time/N,x)
# ax2.plot(time/N,np.real(cleaned))
# ax3.plot(time/N,y-np.real(cleaned))
# """
# we plot the real part of our cleaned data - but since the
# original data was real, the result of our tinkering should
# be real so we don't lose anything by doing this
#
# if you don't explicitly plot the real part, python will
# do it anyway and give you a warning message about only
# plotting the real part of a complex number. so really,
# it's just getting rid of a pesky warning message
# """
#
# fig.subplots_adjust(hspace=0)
# ax1.set_ylim(-13,13)
# ax1.set_ylabel('Original Data')
# ax2.set_ylabel('Filtered Data')
# ax3.set_ylabel('Ideal Result')
# ax3.set_xlabel('Position-Time')
#
# if(save): plt.savefig('SingleWaveAndNoiseFFT.png',dpi=mydpi)
# plt.show()
