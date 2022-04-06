#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 08:36:57 2021

@author: thkam
"""

import numpy as np

def normalize_dist(l, S):
    return S / np.trapz(S, l)

# constants to be used for simulations
class constants:
    O = np.array([0, 0, 0])
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])
    c0 = 3e8
    qe = 1.602e-19
    kB = 1.38e-23
    hP = 6.62e-34
    bK = 2.8977729e-3                 # Wien's constant

def WHITE_LED_SPECTRUM(l):
    lpeak1 = 470e-9
    Dl1 = 20e-9
    s1 = Dl1 / 2 / np.sqrt(np.log(2))
    
    lpeak2 = 600e-9
    Dl2 = 100e-9
    s2 = Dl2 / 2 / np.sqrt(np.log(2))
    
    return np.exp( -(l-lpeak1)**2.0 / s1**2.0 ) + np.exp( -(l-lpeak2)**2.0 / s2**2.0 ) 

def TSFF5210_SPECTRUM(l):
    lpeak = 870e-9
    Dl = 40e-9
    si = Dl / 2 / np.sqrt(np.log(2))
    return np.exp( -(l-lpeak) ** 2.0 / si ** 2.0 )

def RESPONSIVITY(l):
    p = np.array([ -6.39503882,  
                    27.47316339, 
                    -45.57791267,  
                    36.01964536, 
                    -12.8418451,
                    1.73076976 ])
    l = l / 1e-9
    lmin = 330
    lmax = 1090
    s = np.zeros(l.shape)
    i = np.where( (l >= lmin) & (l <= lmax))
    s[i] = np.polyval(p, 2*l[i]/(lmin + lmax) )
    return s

    
def blackbody(l , T):
    P = 2.0**constants.hP**2 *constants.c0 ** 2.0 / l**5/(
            np.exp( constants.hP * constants.c0 / l / constants.kB / T) - 1.0)
    return P 

# wavelength maximum for blackbody radiation
def blackbodymax(T):
    lmax = constants.bK / T
    Pmax = blackbody(lmax,T)
    return lmax, Pmax           

def sunirradiance(pmax, l, T):
    lmax , Pmax = blackbodymax(T)
    return blackbody(l,T) / Pmax * pmax

def SUN_SPECTRUM(l):
    return sunirradiance(1, l, 5800)

def INFRARED_SQUARE_DROP_FILTER(l):
    l = np.array(l)
    return ( ( 320e-9 < l) * (l < 720e-9 ) ).astype(float)

def INFRARED_DROP_FILTER(l):
    l = np.array(l)
    lpeak = (320e-9 + 720e-9) / 2
    l10 = 320e-9
    m = 6
    B = (lpeak - l10) / ( -np.log(0.1) ) ** (1/m)
    return np.exp( -(l-lpeak) **m / B ** m)

def VLC_SQUARE_DROP_FILTER(l):
    l = np.array(l)
    return ( ( 770e-9 < l) * (l < 1100e-9 ) ).astype(float)

def VLC_DROP_FILTER(l):
    l = np.array(l)
    lpeak = 900e-9
    l10 = 750e-9
    m = 6
    B = (lpeak - l10) / ( -np.log(0.1) ) ** (1/m)
    return np.exp( -(l-lpeak) **m / B ** m)

def ALL_PASS(l):
    return np.ones(l.shape)


class defaults:
    i_max = 100e-3                           # maximum driving current for sensor nodes
    P_max = 24.5e-3                          # maximum optical power for sensor nodes
    driver_pol = np.array([ 1.35376064e-01,  1.86846949e-01, -1.01789073e-04])
    driver_pol_inv = np.array([-1.74039667e+01, 5.32917840e+00, 5.61867428e-04])
    room_L = 5
    room_W = 5
    room_H = 3
    nS_master = -constants.ez
    nS_sensor = +constants.ez
    nR_master = -constants.ez
    nR_sensor = +constants.ez    
    A_master = 1e-4
    A_sensor = 1e-4
    m_master = 1
    m_sensor = 1
    RF_master = 1e6
    CF_master = 1e-9    
    RF_sensor = 1e6
    CF_sensor = 1e-9
    Vn_master = 15e-9
    In_master = 400e-15
    fncI_master = 1e3
    fncV_master = 1e3
    Vn_sensor = 15e-9
    In_sensor = 400e-15
    fncI_sensor = 1e3
    fncV_sensor = 1e3    
    temperature = 300
    sp_eff_master = 0.4
    sp_eff_sensor = 0.4
    FOV_master = np.pi / 2.0
    FOV_sensor = np.pi / 2.0
    pd_peak = 2e9
    room_N = 20                                # number of points for wall surfaces
    refl_floor = 0.3
    refl_ceiling = 0.8
    refl_north = 0.8
    refl_south = 0.8
    refl_east = 0.8
    refl_west = 0.8    
    l = np.linspace(200e-9, 1300e-9, 1000)
    amb_pos = 'west wall'
    amb_H = 1
    amb_L1 = 1
    amb_L2 = 1
    amb_name = 'window'
    PT_master = 6
    PT_sensor = 25e-3
    no_bounces = 4
    Rb_master = 1e3
    Rb_sensor = 1e3
    IWU = 1.3e-3
    tWU = 20e-3
    IRO = 1.3e-3
    tRO = 40e-3
    IRX = 1.3e-3
    bits_master = 200
    bits_sensor = 200
    ID_sensor = 10e-3
    Tcycle = 60
    ISL = 400e-9
    QmAh = 220
    BER_target = 1e-3
    Imax_m = 100e-3
    Imax_s = 100e-3
    Imin_m = 0e-3
    Imin_s = 0e-3
    
    
    def __init__(self):
        self.driver_pol_l = np.poly1d( self.driver_pol )
        self.r_master = np.array([[self.room_L * 0.25,
                                   self.room_W * 0.25,
                                   self.room_H ],
                                  [self.room_L * 0.5,
                                   self.room_W * 0.5,
                                   self.room_H ],
                                  [self.room_L * 0.75,
                                   self.room_W * 0.75,
                                   self.room_H ]])
        
        self.r_sensor = np.array([[self.room_L * 0.25,
                                   self.room_W * 0.25,
                                   0.0 ],
                                  [self.room_L * 0.75,
                                   self.room_W * 0.75,
                                   0.0 ]])

        self.ST_m = WHITE_LED_SPECTRUM( self.l )
        self.ST_s = TSFF5210_SPECTRUM( self.l )
        self.SR_m = VLC_DROP_FILTER( self.l )
        self.SR_s = INFRARED_DROP_FILTER( self.l )
        self.ST_a = SUN_SPECTRUM( self.l )
        self.ptd_a = np.trapz( self.ST_a, self. l)        
        self.R_m = RESPONSIVITY( self.l )
        self.R_s = RESPONSIVITY( self.l )
        self.ST_m = normalize_dist( self.l, self.ST_m)
        self.ST_s = normalize_dist( self.l, self.ST_s)
        self.ST_a = normalize_dist( self.l, self.ST_a)
        self.md_pol = self.driver_pol
        self.md_poli = self.driver_pol_inv
        self.sd_pol = self.driver_pol
        self.sd_poli = self.driver_pol_inv
        
        
        