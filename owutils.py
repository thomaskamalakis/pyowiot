#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 14:12:45 2021

@author: thkam
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from defaults import constants
from scipy.interpolate import griddata
from scipy.special import erfc, erfcinv

timer = {}

def bandwidth(t, x, lvl = 3):
# Calculate the full width bandwidth 
    xdB = 20 * np.log10(np.abs(x) / np.max(np.abs(x) ) )
    imax = np.argmax(xdB)
    xmax = xdB[ imax ]
    i = imax - 1
    
    while i >= 0:
        if xdB[i] > xmax - lvl:
            i -= 1
        else:
            break
        
    Df_left = np.interp(xmax - lvl, xdB[ i : i + 1], t[ i : i + 1])
    
    i = imax + 1        
    while i < xdB.size:
        if xdB[i] > xmax - lvl:
            i += 1
        else:
            break
    Df_right = np.interp(xmax - lvl, xdB[i-1 : i ], t[i-1 : i ])
    
    return Df_right - Df_left

    
# create frequency axis
def frequency_axis(t):
    N = t.size
    Dt = t[1] - t[0]
    n = np.arange(-N / 2.0, N / 2.0, 1)
    Df = 1.0 / ( N * Dt)
    return n * Df

# Calculate spectrum of x using FFT
def spectrum(t, x):
    Dt = t[1] - t[0]
    return Dt*np.fft.fftshift(np.fft.fft(np.fft.fftshift(x)))

def Qfunction(x):
    return 0.5 * erfc( x/np.sqrt(2) )

def Qinv(y):
    return np.sqrt(2) * erfcinv( 2 * y )

def contains_None(l):
    p = [x for x in l if x is None]
    return len(p) > 0

def timed(func):
    def wrapper(*args, **kwargs):            
        tstart = time.time()
        res = func(*args, **kwargs)
        tend = time.time()
        telapsed = tend - tstart
        print('Execution lasted %f s' %telapsed)
        fun_name = func.__name__
        timer[fun_name] = telapsed        
        return res
    return wrapper  

def check_plot(close_all, figure_no):
    
    if close_all:
        plt.close('all')
    
    if figure_no is None:
        figure_no = plt.figure()
    else:
        plt.figure( figure_no )
        
    return figure_no

def dot_2D(a , b):
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]

def rel_2D(a, b):
    c = np.copy(a)
    c[:, 0] -= b[0]
    c[:, 1] -= b[1]
    c[:, 2] -= b[2]
    return c

def dot_2D1D(a, b):
    return a[:, 0] * b[0] + a[:, 1] * b[1] + a[:, 2] * b[2]

def mul_2Dsc(a, s):
    b = np.copy(a)
    b[:, 0] = a[:, 0] * s
    b[:, 1] = a[:, 1] * s
    b[:, 2] = a[:, 2] * s
    return b

def lambertian_xyz(m, N):
    u = np.random.rand(N)
    v = np.random.rand(N)
    z = u ** ( 1 / (m+1) )
    r0 = np.sqrt(1 - z ** 2.0)
    x = r0 * np.cos(2 * np.pi * v)
    y = r0 * np.sin(2 * np.pi * v)
    return x, y, z

def random_directions(n_max, N, m):
    """
    Generate rays according to the Lambertian pattern
    """
    x, y, z = lambertian_xyz(m, N)    
    
    """
    Rotate so that ez is aligned to n_max
    """
    Mrot = rotation_matrix( constants.ez, n_max )
    n = np.zeros( [N, 3] )
    n[:, 0] = Mrot[0, 0] * x + Mrot[0, 1] * y + Mrot[0, 2] * z
    n[:, 1] = Mrot[1, 0] * x + Mrot[1, 1] * y + Mrot[1, 2] * z
    n[:, 2] = Mrot[2, 0] * x + Mrot[2, 1] * y + Mrot[2, 2] * z
    
    return n

def random_directions_mul(n_max, N, m):
    """
    Generate random directions according to the directions provided in the N x 3 n_max
    """
    
    xx, yy, zz = lambertian_xyz(m, N)    
    
    Mrot = rotation_matrices( constants.ez, n_max)
         
    nrot, _, _ = Mrot.shape
    
    n = np.zeros([N, 3])
    j = 0
    for i in range(N):
        v = np.transpose( np.array([xx[i], yy[i], zz[i]]) )        
      

        u = np.matmul(np.squeeze(Mrot[j]), v)
        n[i] = np.transpose( u )
        j += 1
        if j >= nrot:
            j = 0
    
    return n


def cart_to_spher(x, y, z):
    r = np.sqrt( x**2.0 + y**2.0 + z**2.0 )
    inc = np.arccos(z / r)
    az = np.arctan2(y, x)
    return r, inc, az

def spher_to_cart(r, inc, az):
    return r * np.cos(inc) * np.sin(az), r * np.sin(inc) * np.sin(az), r * np.cos(az) 

def spher_to_cart_ar(r, theta, phi):
    return np.array([r * np.cos(phi) * np.sin(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(theta)]) 

def skew_symmetric(v):
    """
    skew symmetric matrix obtained from vector v
    """    
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def rotation_matrix(a,b):
    """
    rotation matrix so that vector a coincides with vector b    
    """
    a = a / np.linalg.norm( a )
    b = b / np.linalg.norm( b )
    
    v = np.cross(a,b)
    c = np.inner(a,b)
    
    V = skew_symmetric(v)
    
    if c==-1.0:
        M = - np.eye(3)
    else:
        M = np.eye(3) + V + np.linalg.matrix_power(V,2) / (1+c)
    
    return M


def expand_to(a, N):
    """
    Expand vector a to N x 3 nd.array
    """
    if a.ndim == 1:
        return np.tile( a, (N, 1) )
    else:
        return a
    
def expand_matched(a, b):
    """
    Expand vectors a and b to be N x 3 nd.array
    """
    if len(a.shape) == 1:
        na = 1
    else:
        na = a.shape[0]
    
    a = np.tile(a, (na, 1) )
    
    if len(b.shape) == 1:
        nb = 1
    else:
        nb = b.shape[0]

    b = np.tile(b, (nb, 1) )
    
    if (na > nb):
        b = np.tile(b, (na, 1) )
    elif (na < nb):
        a = np.tile(a, (nb, 1) )
    
    return a, b

def rotation_matrices(a, b):
    """
    Multiple rotation matrices assuming a and b are N x 3 vectors
    """
    aa, bb = expand_matched(a, b)
    N = aa.shape[0]
    
    M = np.zeros([N, 3, 3])
    
    for n in range( N ):
        M[n, :, :] = rotation_matrix(aa[n], bb[n])
    
    return M
        

def interp_on_grid(r, values, x1, x2, 
                      p1 = constants.ex, 
                      p2 = constants.ey):
    
    if r.ndim == 1:
        r = r.reshape([1,3])
    sn1 = r[: , 0] * p1[0] + r[: , 1] * p1[1] + r[: , 2] * p1[2]
    sn2 = r[: , 0] * p2[0] + r[: , 1] * p2[1] + r[: , 2] * p2[2]
    xx1, xx2 = np.meshgrid(x1, x2)
    ri = np.array([sn1, sn2]).transpose()    
    return xx1, xx2, griddata(ri, values, (xx1, xx2))
    
def plot_on_grid(r, values, x1, x2, 
                    p1 = constants.ex, 
                    p2 = constants.ey):
    
    xx1, xx2, v = interp_on_grid(r, values, x1, x2, p1 = p1, p2 = p2)
    plt.pcolor(x1, x2, v,  shading = 'auto')
    
def half_angle_to_order(phi):
    return -np.log(2) / np.log ( np.cos(phi) )

def overlap(a, b, c, d):
    
    if a > c:
        a_tmp = a
        b_tmp = b
        
        a = c
        b = d
        c = a_tmp
        d = b_tmp
    
    if b < c:
        return None
    
    end = min(b, d)
    
    return [c, end]
        
def repeat_row(a, n):
    a = np.squeeze(a)
    return np.repeat( a.reshape([a.size, 1]), n, axis = 1)

def repeat_column(a, n):
    a = np.squeeze(a)
    return np.repeat( a.reshape([1, a.size]), n, axis = 0)
    
def normalize_to_unity(v):
    if v.ndim == 2:
        vn = np.sqrt( v[:,0] ** 2.0 + v[:,1] ** 2.0 + v[:,2] ** 2.0)
        v2 = np.zeros(v.shape)
        v2[:,0] = v[:,0] / vn
        v2[:,1] = v[:,1] / vn
        v2[:,2] = v[:,2] / vn
    else:
        v2 = v / np.linalg.norm(v)
    return v2    

def aligned_to(r_rec, r_tra):

    n = np.zeros( r_rec.shape )
    if r_rec.ndim == 2:
        n[:, 0] = r_rec[:, 0] - r_tra[0]
        n[:, 1] = r_rec[:, 1] - r_tra[1]
        n[:, 2] = r_rec[:, 2] - r_tra[2]
        return normalize_to_unity(n)    
    else:
        return (r_rec - r_tra) / np.linalg.norm(r_rec - r_tra)
    
def distances(rS, rR):
    
    rows_S, columns_S = rS.shape
    rows_R, columns_R = rR.shape

    rSx = rS[:, 0]
    rSy = rS[:, 1]
    rSz = rS[:, 2]
    
    rRx = rR[:, 0]
    rRy = rR[:, 1]
    rRz = rR[:, 2]
    
    rSx2 = repeat_row(rSx, rows_R)
    rSy2 = repeat_row(rSy, rows_R)
    rSz2 = repeat_row(rSz, rows_R)
        
    rRx2 = repeat_column(rRx, rows_S)
    rRy2 = repeat_column(rRy, rows_S)
    rRz2 = repeat_column(rRz, rows_S)
    
    return np.sqrt( 
          (rSx2 - rRx2) ** 2.0 + (rSy2 - rRy2) ** 2.0 
        + (rSz2 - rRz2) ** 2.0 )
    
    
def lambertian_gains(rS, nS, rR, nR, mS, AR, FOV, calc_delays = False):
    """
    Calculate the channel DC gain
    """
    if rS.ndim == 1:
        rS = rS.reshape([1,3])
    if rR.ndim == 1:
        rR = rR.reshape([1,3])
    if nS.ndim == 1:
        nS = nS.reshape([1,3])
    if nR.ndim == 1:
        nR = nR.reshape([1,3])

    nR = normalize_to_unity(nR)
    nS = normalize_to_unity(nS)
    
    # Make sure unity vectors are normalized    
    rows_S, columns_S = rS.shape
    rows_R, columns_R = rR.shape
       
    rSx = rS[:, 0]
    rSy = rS[:, 1]
    rSz = rS[:, 2]
    
    rRx = rR[:, 0]
    rRy = rR[:, 1]
    rRz = rR[:, 2]
    
    nSx = nS[:, 0]
    nSy = nS[:, 1]
    nSz = nS[:, 2]
    
    nRx = nR[:, 0]
    nRy = nR[:, 1]
    nRz = nR[:, 2]
    
    rSx2 = repeat_row(rSx, rows_R).reshape(-1)
    rSy2 = repeat_row(rSy, rows_R).reshape(-1)
    rSz2 = repeat_row(rSz, rows_R).reshape(-1)

    nSx2 = repeat_row(nSx, rows_R).reshape(-1)
    nSy2 = repeat_row(nSy, rows_R).reshape(-1)
    nSz2 = repeat_row(nSz, rows_R).reshape(-1)

    mS2 = repeat_row(mS, rows_R).reshape(-1)
    
    AR2 = repeat_column(AR, rows_S).reshape(-1)
    FOV2 = repeat_column(FOV, rows_S).reshape(-1)
    
    rRx2 = repeat_column(rRx, rows_S).reshape(-1)
    rRy2 = repeat_column(rRy, rows_S).reshape(-1)
    rRz2 = repeat_column(rRz, rows_S).reshape(-1)
    
    
    nRx2 = repeat_column(nRx, rows_S).reshape(-1)
    nRy2 = repeat_column(nRy, rows_S).reshape(-1)
    nRz2 = repeat_column(nRz, rows_S).reshape(-1)
        
    RR = np.sqrt( 
              (rSx2 - rRx2) ** 2.0 + (rSy2 - rRy2) ** 2.0 
            + (rSz2 - rRz2) ** 2.0 )
    

    i = np.where( RR != 0.0 )
    cos_theta = np.zeros( RR.shape )
    cos_phi = np.zeros( RR.shape )
    h = np.zeros( RR.shape )
    
    cos_theta[i] = ( nRx2[i] * (rSx2[i] - rRx2[i]) + nRy2[i] * (rSy2[i] - rRy2[i]) + nRz2[i] * (rSz2[i] - rRz2[i]) ) / RR[i]
    cos_phi[i] = ( nSx2[i] * (rRx2[i] - rSx2[i]) + nSy2[i] * (rRy2[i] - rSy2[i]) + nSz2[i] * (rRz2[i] - rSz2[i]) ) / RR[i] 
    
    rect_theta = cos_theta >= np.cos(FOV2)

    h[i] = (mS2[i] + 1) / (2 * np.pi * RR[i] ** 2.0 ) * (cos_phi[i] ** mS2[i]) * cos_theta[i] * AR2[i] * rect_theta[i]
    
    h = h.reshape([rows_S, rows_R])
    
    i = np.where(h < 0)
    h[i] = 0
    
    if calc_delays:
        d = RR / constants.c0
        d = d.reshape([rows_S, rows_R])
        return h, d
    else:
        return h
    

def vectorize_if_scalar(x, n):
    if np.isscalar(x):
        return x * np.ones(n)
    else:
        return x
    
def array_if_single_vector(x, n):
    
    if x.ndim == 1:        
        xx = np.zeros([n, x.size])
        xx[:] = x
    else:
        xx = x
    return xx
#    return np.squeeze(xx)

def closest_to(r_n, r_o):
    d = distances(r_n, r_o)
    i = np.argmin(d, axis = 1)
    return i

def interp_closest(r_n, r_o, v_o):
    return v_o[ closest_to(r_n, r_o) ]

def project(r, n):
    n = n / np.linalg.norm(n)
    return r[:, 0] * n[0] + r[:, 1] * n[1] + r[:, 2] * n[2]

def project_plane(r, n1, n2):
    v1 = project(r, n1).reshape(-1, 1)   
    v2 = project(r, n2).reshape(-1, 1)
 
    return np.concatenate( (v1, v2), axis = 1)    
    
    