import numpy as np
import matplotlib.pyplot as plt
from libow3 import plot_on_grid, spectra, constants, sensor_net, grid_of_points, aligned_to

HOME_DIR = '/home/thkam/Documents/ow_iot/jocn_paper/figures/'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 14,
    "lines.linewidth" : 2,
})

# Wavelength range
l = np.linspace(200e-9, 1300e-9, 1000)

# Master VLC node at the ceiling
r_m = np.array([2.5, 2.5, 3])
n_m = -constants.ez

SpecT_m = spectra.white_led(l)
SpecR_m = spectra.visible_drop_filter(l)
R_m = spectra.pin_resp(l)
A_m = 1e-4
FOV_m = np.pi / 2.0
m_m = 1
PT_m = 6

# Sensor IR node at the floor
r_s = grid_of_points( r0 = constants.O, 
                      dr1 = 5 * constants.ex, 
                      dr2 = 5 * constants.ey, 
                      N1 = 20, 
                      N2 = 20 )

n_s = aligned_to(r_m, r_s)

SpecT_s = spectra.ir_led(l)
SpecR_s = spectra.ir_drop_filter(l)
R_s = spectra.pin_resp(l)
A_s = 1e-4
FOV_s = np.pi / 2.0
m_s = 45
PT_s = 25e-3

# TIA for the master node
tia_master = {
        'RF' : 1e6,
        'CF' : 1e-9,
        'Vn' : 15e-9,
        'In' : 400e-15,
        'fncI' : 1e3,
        'fncV' : 1e3,
        'temperature' : 300 }

# TIA for the sensor nodes
tia_nodes = {
        'RF' : 1e6,
        'CF' : 1e-9,
        'Vn' : 15e-9,
        'In' : 400e-15,
        'fncI' : 1e3,
        'fncV' : 1e3,
        'temperature' : 300 }


master = {
        'rS' : r_m,
        'rR' : r_m,
        'FOV' : FOV_m,
        'A' : A_m,
        'm' : m_m,
        'nS' : n_m,
        'nR' : n_m,
        'SpecT' : SpecT_m,
        'SpecR' : SpecR_m,
        'R' : R_m,
        'PT' : PT_m,
        'TIA' : tia_master,
        'Rbt' : 1e3,
        'sp_eff' : 0.4}

sensors = {
        'rS' : r_s,
        'rR' : r_s,
        'FOV' : FOV_s,
        'A' : A_s,
        'm' : m_s,
        'nS' : n_s,
        'nR' : n_s,
        'SpecT' : SpecT_s,
        'SpecR' : SpecR_s,
        'R' : R_s,
        'PT' : PT_s,
        'TIA' : tia_nodes,
        'Rbt' : 1e3,
        'sp_eff' : 0.4}

data_rates_u = 1e3 * np.ones([1, 400])
data_rates_d = 2e3 * np.ones([1, 400])

# Window
pd_peak = 2e9
rw_d = grid_of_points(dr1 = np.array([1, 0, 0]), 
                      dr2 = np.array([0, 0, 1]),
                      rm = np.array([2.5, 5, 1.5]),
                      N1 = 10,
                      N2 = 10,
                      return_dict = True)


rw = rw_d['r']
Aw = rw_d['A']
Specw = spectra.sun(l) * pd_peak * Aw

window = {        
        'rS' : rw,
        'A' : Aw,
        'm' : 1,
        'nS' : -constants.ey,
        'SpecT' : Specw}

amb_surfs = [window]

sn = sensor_net(master = master,
                sensors = sensors,
                l = l,
                data_rates_u = data_rates_u,
                data_rates_d = data_rates_d,
                amb_surfs = amb_surfs)

sn.calc_downlink()
sn.calc_uplink()
sn.calc_ambient_light()

x1 = np.linspace(0, 5, 100)
x2 = np.linspace(0, 5, 100)

plt.close('all')
plt.figure(1)
plot_on_grid(sn.sensors['rR'], sn.Pambin_s, x1, x2, 
             p1 = constants.ex, p2 = constants.ey)
plt.title('$P_\mathrm{in}$ [mW]')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.colorbar()
plt.axis('equal')
plt.savefig(HOME_DIR + 'Pambin.png')

plt.figure(2)
plot_on_grid(sn.sensors['rR'], sn.Pamb_s * 1e3, x1, x2, 
             p1 = constants.ex, p2 = constants.ey)

plt.colorbar()
plt.title('$P_\mathrm{amb}$ [mW]')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.axis('equal')
plt.savefig(HOME_DIR + 'Pamb.png')

plt.figure(3)
plot_on_grid(sn.sensors['rR'], sn.Iamb_s * 1e3, x1, x2, 
             p1 = constants.ex, p2 = constants.ey)

plt.colorbar()
plt.title('$I_\mathrm{amb}$ [mA]')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.axis('equal')
plt.savefig(HOME_DIR + 'Iamb.png')

imax = np.max(sn.Iamb_s)
Sshot = 2 * constants.qe * imax