#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:57:37 2021

@author: thkam
"""

from libow2 import node, EZ, sdefaults, WHITE_LED_SPECTRUM, VLC_DROP_FILTER, \
                   grid_of_points, aligned_to, node_array, sensor_net
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 14,
    "lines.linewidth" : 2,
})
N1 = 40
N2 = 40

r_vlc = np.array([2.5, 2.5, 3])
r_infrared = np.array([0.5, 1.0, 0])

vlc_node = node( r = r_vlc,
                 nt = -EZ,
                 nr = -EZ,
                 m = 1,
                 A = 1e-4,
                 tx_spec = WHITE_LED_SPECTRUM,
                 rx_filter = VLC_DROP_FILTER,
                 l_tx_start = sdefaults['vlc_lstart'],
                 l_tx_end = sdefaults['vlc_lend'],
                 l_rx_start = sdefaults['ir_lstart'],
                 l_rx_end = sdefaults['ir_lend'],
                 PT = 6)

#n = aligned_to(r_vlc, r_infrared)
n = +EZ
ir_node = node( r = r_infrared,
                 nt = n,
                 nr = n,
                 m = 1,
                 A = 1e-4,
                 tx_spec = WHITE_LED_SPECTRUM,
                 rx_filter = VLC_DROP_FILTER,
                 l_tx_start = sdefaults['vlc_lstart'],
                 l_tx_end = sdefaults['vlc_lend'],
                 l_rx_start = sdefaults['ir_lstart'],
                 l_rx_end = sdefaults['ir_lend'],
                 PT = 6)


na = node_array( [vlc_node, ir_node] )
s = sensor_net(na)
s.node_array.calc_h()