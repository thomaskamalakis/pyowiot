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
`libow.py` is the library file, `sysdesign_wide.py` is an example of how to use the library. Basically you need to define the system parameters one step at the time for each subsystem of the VLC/IR link. At the very least, you need to specify the following things in a bottom up approach.

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
This defines a master node positioned at `r`, with a field-of-view `FOV`, area equal to `A` (measured in m<sup>2</sup>), with lambertian order `m`, orientation normal vector `n`.


 

