import numpy as np
import matplotlib.pyplot as plt
from libow import (plot_on_grid, spectra, constants, sensor_net, grid_of_points,  
                  aligned_to, sensor_consumption, interp_on_grid, plane_surface,
                  nodes, TIA, driver)
from scipy.special import erfc, erfcinv

HOME_DIR = '/home/thkam/Documents/ow_iot/jocn_paper/figures/'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 14,
    "lines.linewidth" : 2,
})

def plot_arrow(ax, x, y, narrow, dxarrow, dyarrow,fc):
    x_arrow = x[narrow]
    y_arrow = y[narrow]
    plt.annotate( '' , 
                  xy = (x_arrow + dxarrow, y_arrow + dyarrow), 
                  xytext = (x_arrow, y_arrow),                  
                  arrowprops = dict(arrowstyle = '->',
                                    color = fc,
                                    connectionstyle = "angle3,angleA=-90,angleB=180"),
                )

def Qfunction(x):
    return 0.5 * erfc( x/np.sqrt(2) )

def Qinv(y):
    return np.sqrt(2) * erfcinv( 2 * y )
    
BER_0 = 1e-3 
gamma_0 = Qinv(BER_0)
SNR_0 = gamma_0 ** 2.0
SNR_0_dB = 10 * np.log10(SNR_0)

# Room dimensions
L = 5
W = 5
H = 3

# TIA for the master node
tia_master = TIA(RF = 1e6,
                 CF = 1e-9,
                 Vn = 15e-9,
                 In = 400e-15,
                 fncI = 1e3,
                 fncV = 1e3,
                 temperature = 300)

# TIA for the sensor nodes
tia_nodes = TIA(RF = 1e6,
                CF = 1e-9,
                Vn = 15e-9,
                In = 400e-15,
                fncI = 1e3,
                fncV = 1e3,
                temperature = 300)

# sensor node driver
sensor_driver = driver(imax = 100e-3,
                       imin = 0e-3,
                       pol = np.array([ 1.35376064e-01,  1.86846949e-01, -1.01789073e-04]),
                       polinv = np.array([-1.74039667e+01, 5.32917840e+00, 5.61867428e-04]) )

Tcycle = 1 # cycle duration in seconds
# sensor node cycle:


# Uplink and downlink data rates
Rbu = 1e3
Rbd = 1e3

l = np.linspace(200e-9, 1300e-9, 1000)

# master node specification
master = nodes(r = np.array([L/2, W/2, H]),
               FOV = np.pi/2.0,
               A = 1e-4,
               m = 1,
               n = -constants.ez,
               SpecT = spectra.white_led(l),
               SpecR = spectra.visible_drop_filter(l),
               R = spectra.pin_resp(l),
               PT = 6,
               TIA = tia_master,
               sp_eff = 0.4)

# Number of sensor nodes
N1 = 20
N2 = 20

# Positions of the sensor nodes
r_sg = grid_of_points( r0 = constants.O, 
                       dr1 = L * constants.ex, 
                       dr2 = W * constants.ey, 
                       N1 = N1, 
                       N2 = N2 )
r_s = r_sg.r

# Orientations of the sensor nodes
#n_s = aligned_to(master.rS, r_s)
n_s = constants.ez
sensors = nodes(r = r_s,
                FOV = np.pi/2, 
                A = 1e-4, 
                m = 1,
                n = n_s,
                SpecT = spectra.ir_led(l),
                SpecR = spectra.ir_drop_filter(l),
                R = spectra.pin_resp(l),
                PT = 25e-3,
                TIA = tia_nodes, 
                sp_eff = 0.4)

# Uplink and downlink data rates
data_rates_u = Rbu * np.ones([1, N1 * N2])
data_rates_d = Rbd * np.ones([1, N1 * N2])

# Window
pd_peak = 2e9
window = plane_surface(dr1 = np.array([1, 0, 0]), 
                       dr2 = np.array([0, 0, 1]),
                       rm = np.array([2.5, 5, 1.5]),
                       N1 = 10,
                       N2 = 10,
                       m = 1,
                       l = l,
                       n = -constants.ey,
                       pd_peak = pd_peak)


amb_surfs = [window]

sn = sensor_net(master = master,
                sensors = sensors,
                l = l,
                sensor_driver = sensor_driver,
                data_rates_u = data_rates_u,
                data_rates_d = data_rates_d,
                amb_surfs = amb_surfs)

sn.calc_downlink()
sn.calc_uplink()
sn.calc_ambient_light()
sn.calc_snr()

nx = 40
ny = 40

x = np.linspace(0, L, nx)
y = np.linspace(0, W, ny)

# Uplink required power

snu = np.sqrt(sn.p_u[0]) # noise power is the same for all sensor nodes in the uplink

PTu = 2 * gamma_0 * snu / sn.uplink.h.reshape(-1) / sn.uplink.celn 

plt.close('all')
plt.figure(1)
plot_on_grid(sn.sensors.rR, PTu.reshape(-1)/1e-3, x, y, 
             p1 = constants.ex,
             p2 = constants.ey)

plt.title('$P_\mathrm{T0}$ [mW]')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.colorbar()
plt.axis('equal')
#plt.savefig(HOME_DIR + 'PT0_1.png')

ID = sn.calc_sensor_I(PTu)
plt.figure(3)
plot_on_grid(sn.sensors.rR, ID.reshape(-1)/1e-3, x, y, 
             p1 = constants.ex,
             p2 = constants.ey)

plt.title('$I_\mathrm{D0}$ [mA]')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.colorbar()
plt.axis('equal')
#plt.savefig(HOME_DIR + 'ID0_1.png')

xx1, xx2, ID2D = interp_on_grid(sn.sensors.rR,
                                ID.reshape(-1), 
                                x, y, 
                                p1 = constants.ex,
                                p2 = constants.ey)
IDd = np.diagonal(ID2D)
fig, ax = plt.subplots()
p1, = ax.plot(x,IDd/1e-3, '-bo', label = '$I_\mathrm{D}$')
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$I_\mathrm{D}$ [mA]')
plot_arrow(ax, x, IDd/1e-3, 10, -0.7, 0, 'b')

c = sensor_consumption(IWU = 1.3e-3,
                       tWU = 20e-3,
                       IRO = 1.3e-3,
                       tRO = 40e-3,
                       IRX = 1.3e-3,
                       Ldatau = 200,
                       Ldatad = 200,
                       Rbu = 1e3,
                       Rbd = 1e3,
                       ID = 10e-3,
                       Tcycle = 10,
                       ISL = 400e-9,
                       QmAh = 220)

tBL10 = c.battery_life_ITX(IDd, Tcycle = 10)
tBL60 = c.battery_life_ITX(IDd, Tcycle = 60)

ax2 = ax.twinx()

p2, = ax2.plot(x, tBL10, '-rs', label = '$t_\mathrm{BL}$ for $t_\mathrm{CY}=10\mathrm{s}$')
p3, = ax2.plot(x, tBL60, '-go', label = '$t_\mathrm{BL}$ for $t_\mathrm{CY}=60\mathrm{s}$')
plot_arrow(ax2, x, tBL10, 30, 0.7, 100, 'r')
plot_arrow(ax2, x, tBL60, 30, 0.7, 50, 'g')

ax2.set_ylabel('$t_\mathrm{BL}$ [days]')
lgd = plt.legend(handles = [p1, p2, p3], bbox_to_anchor=(1.1, 1), loc='upper left')
#plt.savefig('/home/thkam/Documents/ow_iot/jocn_paper/figures/tBL1.png',
#            bbox_extra_artists=[lgd], bbox_inches='tight')
