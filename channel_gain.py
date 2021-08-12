#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:57:37 2021

@author: thkam
"""

from libow2 import node, EZ, sdefaults, WHITE_LED_SPECTRUM, VLC_DROP_FILTER, \
                   grid_of_points, aligned_to, node_array, sensor_net, \
                   TSFF5210_SPECTRUM,  INFRARED_DROP_FILTER
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
r_ir =  np.array([2.5, 2.5, 0])
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

ir_node = node(r = r_ir, 
               nt = +EZ,
               nr = +EZ,
               m = 45,
               A = 1e-4,
               PT = 25e-3,
               tx_spec = TSFF5210_SPECTRUM,
               rx_filter = INFRARED_DROP_FILTER,
               l_tx_start = sdefaults['ir_lstart'],
               l_tx_end = sdefaults['ir_lend'],
               l_rx_start = sdefaults['vlc_lstart'],
               l_rx_end = sdefaults['vlc_lend'])

#r_infrared = grid_of_points(r0 = np.array([0, 0, 0]),
#                            dr1 = np.array([5, 0, 0]),
#                            dr2 = np.array([0, 5, 0]),
#                            N1 = N1,
#                            N2 = N2)
#
#n_infrared = aligned_to(r_vlc, r_infrared)
#
#vlc_array = node_array( [vlc_node] )
#ir_array = node_array(r = r_infrared, 
#                      nt = n_infrared,
#                      nr = n_infrared,
#                      m = 45,
#                      A = 1e-4,
#                      l_tx_start = sdefaults['ir_lstart'],
#                      l_tx_end = sdefaults['ir_lend'],
#                      l_rx_start = sdefaults['vlc_lstart'],
#                      l_rx_end = sdefaults['vlc_lend'])
#
#s = sensor_net(vlc_array + ir_array)
#s.plot_nodes()
#h_vlc = ir_array.calc_h_as_rx(vlc_array)
#x1 = np.linspace(0, 5, N1)
#x2 = np.linspace(0, 5, N2)
##
#plt.close('all')
#plt.figure(1)
#h_vlcdB = 10 * np.log10(h_vlc)
#ir_array.pcolor_on_grid(h_vlcdB, x1, x2)
#plt.colorbar()
#plt.xlabel('$x$ [m]')
#plt.ylabel('$y$ [m]')
#plt.title('Downlink $h$ [dB]')
#plt.savefig('/home/thkam/Documents/ow_iot/jocn_paper/figures/channelgaind.png')
#
#h_ir = ir_array.calc_h_as_tx(vlc_array)
#h_irdB = 10 * np.log10(h_ir)
#plt.figure(2)
#ir_array.pcolor_on_grid(h_irdB, x1, x2)
#plt.colorbar()
#plt.xlabel('$x$ [m]')
#plt.ylabel('$y$ [m]')
#plt.title('Uplink $h$ [dB]')
#plt.savefig('/home/thkam/Documents/ow_iot/jocn_paper/figures/channelgainu.png')

na = node_array([vlc_node, ir_node])
na.calc_powers()
Specn = na[0].tx_opt.Specn
l = np.linspace( sdefaults['vlc_lstart'], sdefaults['vlc_lend'], 1000)
plt.figure()
plt.plot(l, Specn(l))
