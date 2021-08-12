#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:57:37 2021

@author: thkam
"""

from libow2 import node
import numpy as np
import matplotlib.pyplot as plt

def plot_arrow(ax, x, y, narrow, darrow, fc):
    x_arrow = x[narrow]
    y_arrow = y[narrow]
    dx_arrow = f[darrow] - x_arrow 
    plt.annotate( '' , 
                  xy = (x_arrow + dx_arrow, y_arrow), 
                  xytext = (x_arrow, y_arrow),                  
                  arrowprops = dict(arrowstyle = '->',
                                    color = fc,
                                    connectionstyle = "angle,angleA=-90,angleB=180,rad=0"),
                )

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 14,
    "lines.linewidth" : 2,
})

n = node()
f = np.logspace(-1, 6, num=25)
SV = n.rx_elec.SV_psd(f)
SI = n.rx_elec.SI_psd(f)
SRF = n.rx_elec.RF_psd(f)
ZF = n.rx_elec.ZF(f)

plt.close('all')
fig, ax = plt.subplots()

p1, = ax.loglog(f, SI/1e-24, '-bo', label = '$S_\mathrm{I}$') 
p2, = ax.loglog(f, SV/1e-24, '-rs', label = '$S_\mathrm{V}/|Z_\mathrm{F}|^2$')
p3, = ax.loglog(f, SRF/1e-24, '-cd', label = '$S_\mathrm{RF}/R_\mathrm{F}^2$')

ax.set_xlabel('$f$ [Hz]')
ax.set_ylabel('PSD [$\mathrm{pA}^2$/$\mathrm{Hz}$]')
plot_arrow(ax, f, SI/1e-24, 8, 4, 'b')
plot_arrow(ax, f, SV/1e-24, 12, 8, 'r')


ax2=ax.twinx()
ax2.set_ylabel('$|Z(f)|$ [$\Omega$]')
p4, = ax2.loglog(f, np.abs(ZF), '--k', label = '$|Z_\mathrm{F}|$')

plot_arrow(ax2, f, np.abs(ZF), 18, 20, 'k')
plt.show()
lgd = plt.legend(handles = [p1, p2, p3, p4], bbox_to_anchor=(1.1, 1), loc='upper left')
plt.savefig('/home/thkam/Documents/ow_iot/jocn_paper/figures/psd.png',
            bbox_extra_artists=[lgd], bbox_inches='tight')


