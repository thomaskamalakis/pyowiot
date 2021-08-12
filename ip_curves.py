#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 08:56:17 2021

@author: thkam
"""
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 14,
    "lines.linewidth" : 2,
})
MIN = 1
MAX = 100
N = 10

data = np.genfromtxt('ir_ip2.csv', delimiter = ',')
i = data[:,0]
indx = np.argsort(i)
i = i[indx]
P = data[:,1]
P = P[indx]
r = np.where( (i >= MIN) & ( i<= MAX) )
i = i[r]
P = P[r]
scalex = 1000
scaley = 1000

#x = np.log10(i/scalex)
#y = np.log10(P/scaley)

## Fitting on logarithmic scale
#x = np.log10(i / scalex)
#y = np.log10(P / scaley)
#
#p = np.polyfit(x, y, 2)
#yp = np.polyval(p, x)
#
#plt.close('all')
#plt.figure(1)
#plt.plot(x, y, x, yp)
#
#Pp = 10 ** y * scaley 
#plt.figure(2)
#plt.loglog(i, P, 'rs', label = 'light/current')
#plt.loglog(i, Pp, 'r-', label = 'polynomial fitting')
#
#plt.xlabel('$I_\mathrm{D}$ [mA]')
#plt.ylabel('$I_\mathrm{e}$ [mW/sr]')
#
#plt.legend()
##plt.savefig('/home/thkam/Documents/ow_iot/jocn_paper/figures/ip_ir.png')
#print(p)

m = 45

P = P * 2 * np.pi / (m+1)
x = i / scalex
y = P / scaley 

p = np.polyfit(x, y, 2)
pinv = np.polyfit(y, x, 2)

plt.close('all')

yp = np.polyval(p, x)
Pp = yp * scaley 
plt.figure(1)
plt.loglog(i, P, 'bo', label = 'light/current')
plt.loglog(i, Pp, 'r-', label = 'polynomial fitting')

plt.xlabel('$I_\mathrm{D}$ [mA]')
plt.ylabel('$P_\mathrm{T}$ [mW]')

plt.legend()
plt.savefig('/home/thkam/Documents/ow_iot/jocn_paper/figures/ip_ir.png')
print(p)

xpi = np.polyval(pinv, y)
ipi = xpi * scalex 
plt.figure(2)
plt.loglog(P, i, 'bo', label = 'current/light')
plt.loglog(P, ipi, 'r-', label = 'polynomial fitting')

plt.ylabel('$I_\mathrm{D}$ [mA]')
plt.xlabel('$P_\mathrm{T}$ [mW]')

plt.legend()
plt.savefig('/home/thkam/Documents/ow_iot/jocn_paper/figures/ip_ir_inv.png')
print(pinv)