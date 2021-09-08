# pyowiot
This is an open-source Python3 library that can be used to design optical wireless systems for internet-of-things (IoT) applications

## Funding support
We wrote this code under COST action NewFocus CA19111.

## What is all about?
We can use this library to simulate a typical optical wireless system used for IoT. We assume a master node (MN) placed somewhere in a room (preferably the ceiling). The master node uses visible light communications (VLC) to send acknowledgment messages to sensor nodes (SNs) also placed inside the room. There are two main components of the model: the physical layer (PHY) and the power consumption model (PCM). Typically we expect the SN transmitted to be in the infrared (IR) range, e.g. an IR light emitting diode (LED). On the other hand, the MN would use a visible LED which could provide both illumination in the room and communication with the SNs.

## PHY model
The PHY model deals with all sorts of details of the communications system including:
  - transmitter model: radiation pattern (Lorentzian or extended Lorentzian), light/current characteristic.
  - channel gain: line-of-sight components from the MN to the SN and vice-versa.
  - receiver model: spectral matching, responsivity, optical rejection filter, ambient light noise, amplifier noise.

## Requirements
It requires `scipy`, `numpy` and `matplotlib`. Install them using `pip3`

## How to use it
`libow.py` is the library file, `sysdesign_wide.py` is an example of how to use the library. Basically you need to define the system parameters one step at the time for each subsystem of the VLC/IR link. At the very least, you need to specify the following things in a bottom up approach. Here are some important points from `sysdesign_wide.py` you need to pay attention to.

### Master node transimpendance amplifier (MN-TIA)
```
tia_master = TIA(RF = 1e6,
                 CF = 1e-9,
                 Vn = 15e-9,
                 In = 400e-15,
                 fncI = 1e3,
                 fncV = 1e3,
                 temperature = 300)
```
This defines a transimpendance amplifier (TIA) for the master node with a feedback resistor equal to 10<sup>6</sup>Ω, a feedback capacitance 1nF, at temperature equal to 300K, assuming 15nV and 400fA noise root mean square (RMS) amplitudes for the operational amplifier of the TIA and 1kHz corner frequencies for these noises. See a full explanation of noise characteristics of the opamp [here](https://www.ti.com/lit/an/slva043b/slva043b.pdf).

### Sensor node transimpendance amplifier (SN-TIA)
```
tia_nodes = TIA(RF = 1e6,
                CF = 1e-9,
                Vn = 15e-9,
                In = 400e-15,
                fncI = 1e3,
                fncV = 1e3,
                temperature = 300)
```
This is pretty much the same except it is for the SNs.

### LED and driver circuit
```
sensor_driver = driver(imax = 100e-3,
                       imin = 0e-3,
                       pol = np.array([ 1.35376064e-01,  1.86846949e-01, -1.01789073e-04]),
                       polinv = np.array([-1.74039667e+01, 5.32917840e+00, 5.61867428e-04]) )
```
This defines the light/current and current/light characteristic using `numpy` polyomials, `pol` and `polinv` respectively, while `imin` and `imax` define the range where these polynomials are valid.   

### Master node (MN)
```
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
```
This defines a master node positioned at `r`, with a field-of-view `FOV`, area equal to `A` (measured in m<sup>2</sup>), with lambertian order `m`, orientation normal vector `n`. `SpecT`, `SpecR` and `R` are the transmission spectra, the optical receiver filter characteristic and the responsivity. The class `spectra` defines some common spectrum models that can be used, see the code documentation for more info. `PT` is the transmission power, `TIA` is the TIA amplifier description (see above) and `sp_eff` is the spectral efficiency.

### Sensor nodes (SNs)
You can define multiple sensor nodes (SNs) located at different positions in the room. One quick way to do this is to place all SNs on a plane (say parallel to the floor). To do this, we use the `grid_of_points` class.
```
r_sg = grid_of_points( r0 = constants.O, 
                       dr1 = L * constants.ex, 
                       dr2 = W * constants.ey, 
                       N1 = N1, 
                       N2 = N2 )
r_s = r_sg.r
```
This defines a set of points located on a grid on a plane surface parallel to the floor. `r0` is the plane surface origin and `dr1` and `dr2` define the orientation and size of the plane surface. The continuous surface contains all points `r0 + I * dr1 + J * dr2` where `I` and `J` range from 0 to 1. The actual grid points are taken inside the surface `r0 + (i + 0.5) * dr1 / N1 + (j + 0.5) * dr2 / N2` where `i` and `j` range from 0 to `N1-1` and `N2-1` respectively. We next use the `nodes` class to define the sensor nodes as well.
```
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
```





 

