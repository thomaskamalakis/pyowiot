#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 08:27:49 2021

@author: thkam
"""
from libow3 import sensor_consumption
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 14,
    "lines.linewidth" : 2,
})

ID = np.logspace(-6, -1, num = 50)
tbattery1 = np.zeros( ID.size )
tbattery10 = np.zeros( ID.size )
tbattery60 = np.zeros( ID.size )

for i, IDv in enumerate(ID):
    c = sensor_consumption(IWU = 1.3e-3,
                           tWU = 20e-3,
                           IRO = 1.3e-3,
                           tRO = 40e-3,
                           IRX = 1.3e-3,
                           Ldatau = 200,
                           Ldatad = 200,
                           Rbu = 1e3,
                           Rbd = 1e3,
                           ITX = IDv,
                           Tcycle = 10,
                           ISL = 400e-9,
                           QmAh = 220)
    
    tbattery10[i] = c.battery_life() / 3600 / 24
    
    c.Tcycle = 1
    tbattery1[i] = c.battery_life() / 3600 / 24
    
    c.Tcycle = 60
    tbattery60[i] = c.battery_life() / 3600 / 24
    
plt.close('all')
plt.figure(1)
plt.loglog( ID/1e-3, tbattery60, 'bs-', label = '$t_\mathrm{CY}=60\mathrm{s}$', markevery = 2)
plt.loglog( ID/1e-3, tbattery10, 'ko-', label = '$t_\mathrm{CY}=10\mathrm{s}$', markevery = 2)
plt.loglog( ID/1e-3, tbattery1, 'r-', label = '$t_\mathrm{CY}=1\mathrm{s}$', markevery = 2)
plt.xlabel('$I_\mathrm{TX}$ [mA]')
plt.ylabel('$t_\mathrm{BL}$ [days]')
plt.legend()
plt.grid()
plt.savefig('/home/thkam/Documents/ow_iot/jocn_paper/figures/tbattery.png')

    