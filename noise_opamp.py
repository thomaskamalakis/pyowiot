#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 14:40:52 2021

@author: thkam
"""

SV_s = 15e-9
SI_s = 400e-15
Zf = 1e7
KB = 1.38e-23
B = 25e3
T = 300
SR = 4 * KB * T * Zf  
P_noise = SV_s ** 2.0 * B / Zf ** 2.0 + \
          SI_s ** 2.0 * B + \
          SR * B / Zf ** 2.0
          