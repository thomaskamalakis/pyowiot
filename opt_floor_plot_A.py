#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 08:45:43 2022

@author: thkam
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 12,
    "lines.linewidth" : 2,
})

def make_scalar( w ):
    if not np.isscalar(w):
        return w[0]
    else:
        return w
    
def dict_to_np( d, fom, key ):
    rs = [ make_scalar( r[key] ) for r in results if r['fom'] == fom ]
    return np.array(rs)

def dict_params_to_np( d, fom, key ):
    rs = [ make_scalar( r['params'][key] ) for r in results if r['fom'] == fom ]
    return np.array(rs)

KEY = 'A'
FILENAME = 'diagonal_A.pickle'

with open(FILENAME,'rb') as f:
    results = pickle.load(f)
    
Nx = 10
Ny = 10

x = []
y = []
crossovers = []
fitness = []
uniformity = []


x_los = dict_to_np( results, 'tb_los', 'x')
y_los = dict_to_np( results, 'tb_los', 'y')
x_tot = dict_to_np( results, 'tb_tot', 'x')
x_tot = dict_to_np( results, 'tb_tot', 'y')
x_diff = dict_to_np( results, 'tb_diff', 'x')
x_diff = dict_to_np( results, 'tb_diff', 'y')    

f_los = dict_to_np( results, 'tb_los', 'fitness')
f_diff = dict_to_np( results, 'tb_diff', 'fitness')
f_tot = dict_to_np( results, 'tb_tot', 'fitness')

Rb_los = dict_params_to_np( results, 'tb_los', 2)
Rb_diff = dict_params_to_np( results, 'tb_diff', 2)
Rb_tot = dict_params_to_np( results, 'tb_tot', 2)

phi_los = dict_params_to_np( results, 'tb_los', 1)
phi_diff = dict_params_to_np( results, 'tb_diff', 1)
phi_tot = dict_params_to_np( results, 'tb_tot', 1)

theta_los = dict_params_to_np( results, 'tb_los', 0)
theta_diff = dict_params_to_np( results, 'tb_diff', 0)
theta_tot = dict_params_to_np( results, 'tb_tot', 0)


plt.close('all')

plt.figure()
plt.plot(x_los, Rb_los/1e3 , '-o' , label = 'LOS')
plt.plot(x_tot, Rb_tot/1e3 , '-s' , label = 'Total')
plt.plot(x_los, Rb_diff/1e3, '-x' , label = 'Diffuse')
plt.legend()
plt.xlabel('$x$ [m]')
plt.ylabel(r'$R_\mathrm{b}$ [Kb/s]')
plt.title('Conf. A')
plt.ylim([9.9, 10.1])
plt.savefig('Rb_A.png',bbox_inches='tight')


plt.figure()
plt.plot(x_los, f_los , '-o' , label = 'LOS')
plt.plot(x_tot, f_tot , '-s' , label = 'Total')
plt.plot(x_los, f_diff, '-x' , label = 'Diffuse')
plt.legend()
plt.xlabel('$x$ [m]')
plt.ylabel(r'$t_\mathrm{BL}$ [days]')
plt.title('Conf. A')
plt.savefig('tBL_A.png',bbox_inches='tight')

plt.figure()
plt.plot(x_los, theta_los / np.pi, '-o' , label = 'LOS')
plt.plot(x_tot, theta_tot / np.pi, '-s' , label = 'Total')
plt.plot(x_los, theta_diff / np.pi, '-x' , label = 'Diffuse')
plt.legend()
plt.xlabel('$x$ [m]')
plt.ylabel(r'$\theta_\mathrm{opt} / \pi$')
plt.title('Conf. A')
plt.savefig('theta_A.png',bbox_inches='tight')

