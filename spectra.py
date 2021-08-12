#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:57:37 2021

@author: thkam
"""

from libow2 import WHITE_LED_SPECTRUM, TSFF5210_SPECTRUM, VLC_DROP_FILTER, INFRARED_DROP_FILTER, RESPONSIVITY
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 14,
    "lines.linewidth" : 2,
})

l = np.linspace(400e-9, 1400e-9, 1000)
ST_white = WHITE_LED_SPECTRUM(l)
ST_white = ST_white / np.max(ST_white)

ST_infrared = TSFF5210_SPECTRUM(l)

fig = plt.figure()

plt.plot(l/1e-9, ST_white, '-b', label = 'white LED')
plt.plot(l/1e-9, ST_infrared, '--r', label = 'infrared LED')
plt.ylabel('$S_\mathrm{T}(\lambda)$ [a.u.]')
plt.xlabel('$\lambda$ [nm]')
plt.legend()
plt.savefig('/home/thkam/Documents/ow_iot/jocn_paper/figures/transmission.png')

l = np.linspace(200e-9, 1200e-9, 1000)
SR_vlc_drop = VLC_DROP_FILTER(l)
SR_ir_drop = INFRARED_DROP_FILTER(l)

plt.figure(2)
plt.plot(l/1e-9, SR_vlc_drop, '-b', label = 'VLC drop filter')
plt.plot(l/1e-9, SR_ir_drop, '--r', label = 'IR drop filter')
plt.ylabel('$S_\mathrm{R}(\lambda)$')
plt.xlabel('$\lambda$ [nm]')
plt.legend()
plt.savefig('/home/thkam/Documents/ow_iot/jocn_paper/figures/recfilter.png')

data = np.genfromtxt('R.csv', delimiter = ',')
l = data[:,0]
lmax = np.max(l)
lmin = np.min(l)
scale = (lmax + lmin) * 0.5
R = data[:,1]
plt.figure(3)
p = np.polyfit(l/scale, R, 5)
Rp = np.polyval(p, l/scale)
Rp = RESPONSIVITY(l*1e-9)
plt.plot(l, Rp, 'r', label = 'polynomial fitting')
plt.plot(l, R, 'b--', label = 'original curve') 
plt.ylabel('$\mathcal{R}(\lambda)$ [A/W]')
plt.xlabel('$\lambda$ [nm]')

plt.legend()
plt.savefig('/home/thkam/Documents/ow_iot/jocn_paper/figures/responsivity.png')



    