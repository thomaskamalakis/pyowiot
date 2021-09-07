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

`tia_master = TIA(RF = 1e6,
                 CF = 1e-9,
                 Vn = 15e-9,
                 In = 400e-15,
                 fncI = 1e3,
                 fncV = 1e3,
                 temperature = 300)`


