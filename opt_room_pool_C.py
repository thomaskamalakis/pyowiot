#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 11:44:26 2021

@author: thkam
"""
import numpy as np
from libow8 import sensor_net
import matplotlib.pyplot as plt
import pickle
import mixed_ga as ga
import owutils as ut
from designs import designs
from multiprocessing import Pool
from time import sleep

N_CPUS = 4
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 14,
    "lines.linewidth" : 2,
})
COMPUTE = True
DO_PLOTS = True
FOMs_latex = ['t_\mathrm{b}^{\mathrm{LOS}}', 't_\mathrm{b}^{\mathrm{TOT}}', 't_\mathrm{b}^{\mathrm{DIFF}}']
TITLES = ['Diffuse only']
PREFIX = 'diagonal'
KEY = 'C'
FILENAME = PREFIX + '_' + KEY + '.pickle'

map_type = 'R' * 3
mins = np.array([0, 0, 1e3])
maxs = np.array([np.pi/2.0, 2 * np.pi, 10e3])
Nparams = len(map_type)
Nx = 30
h_ww = None
L = designs[KEY]['room_L']
W = designs[KEY]['room_W']    
x = L * np.arange(1, Nx + 1) / (Nx+1)
y = W * np.arange(1, Nx + 1) / (Nx+1)

# Check if filename exists. If so this could mean the run was not completed

        
def sensor_ar(theta_t, phi_t, Rb, angle, FOM): 
    global h_ww
    global designs
    angle_rad = np.pi/180 * angle
    m = ut.half_angle_to_order(angle_rad)
    nS = ut.spher_to_cart_ar(1, theta_t, phi_t)
    designs[KEY]['nS_sensor'] = nS
    designs[KEY]['Rb_sensor'] = Rb 
    designs[KEY]['m_sensor'] = int(m)
    l = sensor_net( **designs[KEY] ) 
    l.calch(h_ww = h_ww)
    l.light_sim()
    l.calc_noise()
    l.calc_rq()
    l.calc_tbattery()    
    h_ww = l.h_ww
    fitness = getattr(l, FOM)[0]
    return fitness 

def ga_optimization(d):
    global h_ww
    global designs
    i = d['i']
    j = d['j']
    x = d['x']
    y = d['y']
    KEY = d['KEY']
    FOM = d['FOM']
    fitness_fun = lambda params: sensor_ar(params[0], params[1], params[2], 60, FOM)
    
    r_sensor = np.array([x, y, 0])
    designs[KEY]['r_sensor'] = r_sensor
    print('i = %d / %d, j = %d / %d sensor at %s, initiating pool.' %(i, x.size, j, y.size, r_sensor) )              
    
    g = ga.population( noChromosomes = 50,
                       noGenes = Nparams,
                       mins = mins,
                       maxs = maxs,
                       mapType = map_type,
                       fitnessFun = fitness_fun,
                       maxNoCrossovers = 20000,
                       reportEvery = 0,
                       mutationFactor = 0.3,
                       verboseLvl = 0,
                       reqUniformity = 0.001)
    print('Starting optimization i,j = ',i,j)
    g.simulate()
    params = g.chromosomes[0].variableValues()
    bf = g.chromosomes[0].fitness
    print('Optimization ended = ',i,j, 'optimal params: ', params, 'optimal fitness:', bf)

    result = {
        'x' : x,
        'y' : y,
        'i' : i,
        'j' : j,
        'key' : KEY,
        'fom' : FOM,
        'r_sensor' : r_sensor,
        'params' : params,
        'fitness' : bf,
        'fitnessEvaluations' : g.fitnessEvaluations,
        'crossovers' : g.crossovers,
        'uniformity' : g.avgUniformity(),
        'bestFitness' : g.bestFitnessRecords,
        'worstFitness' : g.worstFitnessRecords,
        'fitnesses' : g.getFitnesses()
        }
    
    return result

def dummy_ga(d):

    i = d['i']
    j = d['j']
    x = d['x']
    y = d['y']
    w = 1 - j / 10
    print('Starting GA i,j = ',i,j)
    sleep( w )
    print('GA ended = ',i,j)
    result = {'i': i, 'j': j, 'wait' : w, 'x' : x, 'y' : y}        
    return result
    
pool_args = []
results = []

for i, xc in enumerate(x):
    yc = y[i]   
    pool_args.append({'i' : i,
         'j' : i,
         'x' : xc,
         'y' : yc,
         'KEY' : 'C',
         'FOM' : 'tb_diff'
        })
    
    pool_args.append({'i' : i,
         'j' : i,
         'x' : xc,
         'y' : yc,
         'KEY' : 'C',
         'FOM' : 'tb_tot'
        })
    
    pool_args.append({'i' : i,
         'j' : i,
         'x' : xc,
         'y' : yc,
         'KEY' : 'C',
         'FOM' : 'tb_los'
        })
    
         
pool = Pool(processes = 16)
results = pool.map_async( ga_optimization, pool_args ).get()
with open(FILENAME, 'wb') as f:
    pickle.dump(results, f)
