# Pyowiot
`Pyowiot` is an open-source Python3 library that can be used to design optical wireless systems for internet-of-things (IoT) applications. It contains many components describing the optical wireless channel and transceiver model as well as an optimization engine based on a genetic algorithm.

## Funding support
We wrote this code under COST action NewFocus CA19111. Official acknowledgment is as follows:

> This software was based upon work from COST Action
NEWFOCUS CA19111, supported by COST (European Cooperation in Science and Technology)

## What is it all about?
We can use this library to simulate a typical optical wireless system used for IoT. We assume a master node (MN) placed somewhere in a room (preferably the ceiling). The master node uses visible light communications (VLC) to send acknowledgment messages to sensor nodes (SNs) also placed inside the room. There are two main components of the model: the physical layer (PHY) and the power consumption model (PCM). Typically we expect the SN transmitted to be in the infrared (IR) range, e.g. an IR light emitting diode (LED). On the other hand, the MN would use a visible LED which could provide both illumination in the room and communication with the SNs.

## PHY model
The PHY model deals with all sorts of details of the communications system including:
  - transmitter model: radiation pattern (Lorentzian or extended Lorentzian), light/current characteristic.
  - channel gain: 
  	- line-of-sight (LOS) components from the MN to the SN and vice-versa.
  	- diffuse propagation: we use a fast method for estimating the diffuse power impinging on the MN
  - receiver model: spectral matching, responsivity, optical rejection filter, ambient light noise, amplifier noise.

## Can you give me more info on the models?
The is a paper describing the LOS model and transceiver model:
> T. Kamalakis, Z. Ghassemlooy, S. Zvanovec, and L.N. Alves, “Analysis and simulation of a hybrid visible-light/infrared optical wireless network for IoT applications,” Journal of Optical Communications and Networking 14, 69-78 (2022). DoI: 10.1364/JOCN.442787 

You can download the paper [here](https://www.newfocus-cost.eu/wp-content/uploads/2022/01/442787-1.pdf).

We have also submitted a second paper describing the genetic algorithm and the diffuse channel model which will be made available if it gets accepted. An earlier version of the genetic algorithm used in this work can be found [here](https://galaxy.hua.gr/~thkam/Publications/Journals/optics_communications_2021.pdf).

## Requirements
It requires `scipy`, `numpy` and `matplotlib`. Install them using `pip3`

## Documentation
Definitely not complete yet. Will be adding further documentation in due time.

## What is contained in this repo?
There are several quick-start examples in the repo and others will be added soon. 

- `opt_room_pool_A.py` is an example regarding the optimization of the SN along the diagonal of the floor of a 5m x 5m x 3.5m room with the MN located at the center of the ceiling.
- `opt_room_pool_C.py` is the same as above but the room is larger, 10m x 10m x 4m 

You can use the `opt_floor_plot_A.py` and `opt_floor_plot_C.py` to plot the results of the simulations. Additional files include:
- `libow8.py` is the library file containing the transceiver, LOS and diffuse channel model.
- `mixed_ga.py` contains the implementation of the genetic algorithm.
- `owutils.py` contains some utility functions (but no classes!)
- `defaults.py` defines some constants and default values for the various parameters of the model. 
- `designs.py` contains some example room configurations.

Basically you need to define the system parameters one step at the time for each subsystem of the VLC/IR link. At the very least, you need to specify the following things in a bottom up approach. Here are some important points from `sysdesign_wide.py` you need to pay attention to.

## How do I use your code?
At this stage, it is preferable to start from one quick-start example (e.g.`opt_room_pool_A.py`) and work your way through. Taking `opt_room_pool_A.py` as a starting point:

- provide a file where your results will be stored. The `FILENAME` variable is used for this purpose.

### GA setup
define a variable map type for the genetic algorithm. In this example we run the optimizations over three real-valued variables: elevation, azimuth and data rate. So the map is 
```
map_type = 'R' * 3`
```

implying three real-valued variables. Except `'R'` you can specify `'I'` implying integer-valued variables.

Next you need to provide the ranges of these variables using the `mins` and `maxs` variables.
```
mins = np.array([0, 0, 1e3])
```
signifies that the first two variables (elevation and azimuth) will have minimum value equal to zero and the third (data rate) will have a minimum value of `1e3` (i.e., 1 kb/s ). In a similar fashion 
`maxs` determines the upper bounds for these parameters.

### Sensor node positions
You then define the positions of the sensor nodes to be considered. 
```
Nx = 30
L = designs[KEY]['room_L']
W = designs[KEY]['room_W']    
x = L * np.arange(1, Nx + 1) / (Nx+1)
y = W * np.arange(1, Nx + 1) / (Nx+1)
```
creates a set of 30 x and y pairs corresponding to the diagonal of the ceiling.

### Initial subsurface gains
Note that the `h_ww` is a global variable that is used in diffuse channel calculations. It represents the gains between all elementary subsurfaces of the room that are used in the diffuse channel model. You only need to calculate this once, since they are independent on the parameters of the SN and MN. We initially set 
```
h_ww = None
```
so that the gains are calculated the first time they are needed (and used afterwards in subsequent calculation)

### Optimization objective function

We next define a auxiliary function `sensor_ar` that is used to estimate the battery life of an SN under a specific sensor arrangement. The battery life is the fitness function, i.e. the optimization objective to be used in the optimizations. The function accepts several input variables:

```
sensor_ar(theta_t, phi_t, Rb, angle, FOM, KEY = None, designs = designs) 
```
where `theta_t` is the elevation angle, `phi_t` is the azimuth (both measured in radians), `Rb` is the data rate (b/s), `angle` is the beamwidth of the SN beam (this one is measured in degrees). `FOM` specifies under which conditions the battery life will be calculated:
- `FOM = 'tb_los'` implies only LOS component will be accounted for,
- `FOM = 'tb_diff'` implies only diffuse component will be accounted for,
- `FOM = 'tb_tot'` implies both LOS and diffuse components will be accounted for.

The variable `designs` is a dictionary of the form:
```
designs = {
  design_key :  {
    'room_L' : room length,
    'room_W' : room width,
    'room_H' : room height,
    'refl_north' : north wall reflectivity,
    'refl_south' : south wall reflectivity,
    'refl_east' : east wall reflectivity,
    'refl_west' : west wall reflectivity,
    'refl_ceiling' : ceiling reflectivity,
    'refl_floor' : floor reflectivity,
    'm_sensor' : sensor transmitter Lambertian order,
    'r_sensor' : sensor position,
    'm_master' : master node transmitter Lambertian order,
    'r_master' : master node position,
    'FOV_master' : master node field-of-view (FOV) [rad],
    'FOV_sensor' : sensor node field-of-view (FOV) [rad],
    'amb_L1' : ambient light source size (horizontal),
    'amb_L2' : ambient light source size (vertical),
    'nR_sensor' : sensor receiver orientation,
    'nS_sensor' : sensor transmitter orientation,
    'nR_master' : master node receiver orientation,
    'nS_master' : master node transmitter orientation,
    'no_bounces' : number of light bounces considered in the simulation,
    'Rb_master' : master node data rate,
    'Rb_sensor' : sensor node data rate,
    'PT_sensor' : sensor node transmit power,
    'PT_master' : master node transmit power,
    'A_master' : effective area of master node receiver,
    'A_sensor' : effective area of sensor node receiver
    },...
```

Example:
```
designs = {
  'A' :  {
    'room_L' : 5,
    'room_W' : 5,
    'room_H' : 3,
    'refl_north' : 0.8,
    'refl_south' : 0.8,
    'refl_east' : 0.8,
    'refl_west' : 0.8,
    'refl_ceiling' : 0.8,
    'refl_floor' : 0.3,
    'm_sensor' : 1,
    'r_sensor' : np.array([2.5, 2.5, 0]),
    'm_master' : 1,
    'r_master' : np.array([2.5, 2.5, 3]),
    'FOV_master' : np.pi / 2.0,
    'FOV_sensor' : np.pi / 2.0,
    'amb_L1' : 1.0,
    'amb_L2' : 1.0,
    'nR_sensor' : constants.ez,
    'nS_sensor' : constants.ez,
    'nR_master' : -constants.ez,
    'nS_master' : -constants.ez,
    'no_bounces' : 4,
    'Rb_master' : 10e3,
    'Rb_sensor' : 10e3,  
    'PT_sensor' : 25e-3,
    'PT_master' : 6,
    'A_master' : 1e-4,
    'A_sensor' : 1e-4
    }
```

In the quick-start example `opt_room_pool_A.py`, the values of `Rb_sensor`, `nS_sensor`, `rS_sensor` and `m_sensor` are overwritten by the arguments of the `sensor_ar` function.

We also define the `ga_optimization` function which carries out the GA optimization. Note that this function is passed to the `Pool` instance at the end of the file.

## The defaults class
If some parameters are omitted, an effort will be made to get their values from the class `defaults` that is defined in `defaults.py`. Here is some information regarding classes contained in `libow8.py` and the parameters found in `defaults`

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
### Ambient light sources
We can define surfaces that emit ambient light (windows, etc). The `plane_surface` class is the way to do this.```
```
window = plane_surface(dr1 = np.array([1, 0, 0]), 
                       dr2 = np.array([0, 0, 1]),
                       rm = np.array([2.5, 5, 1.5]),
                       N1 = 10,
                       N2 = 10,
                       m = 1,
                       l = l,
                       n = -constants.ey,
                       pd_peak = pd_peak)
```
The parameters `N1`, `N2`, `dr1` and `dr2` are the same as in `grid_of_points`. We now specify the middle of the plane surface `rm` and also provide the wavelength range  `l`, the orientation of the surface  `n` and the  peak spectral irradiance `pd_peak`

### The sensor_net class
At the top level, we use the `sensor_net` class to define our topology.
```
amb_surfs = [window]
sn = sensor_net(master = master,
                sensors = sensors,
                l = l,
                sensor_driver = sensor_driver,
                data_rates_u = data_rates_u,
                data_rates_d = data_rates_d,
                amb_surfs = amb_surfs)
```
We can then use the following code to calculate the PHY performance:
```
sn.calc_downlink()        # Downlink channel gain
sn.calc_uplink()          # Uplink channel gain
sn.calc_ambient_light()   # Ambient light sources
sn.calc_snr()             # signal-to-noise calculation assuming on/off keying (OOK)
```

### Energy usage
The `sensor_consumption` class can be used to estimate the energy usage of an SN node. Example:
```
c = sensor_consumption(IWU = 1.3e-3,      # Wake up current of the micro controller unit (MCU) of the node [Amps]
                       tWU = 20e-3,       # Duration of the wake up phase [seconds]                       
                       IRO = 1.3e-3,      # Current drawn during read-out (RO) phase [Amps]
                       tRO = 40e-3,       # Duration of the read-out phase [seconds]
                       IRX = 1.3e-3,      # Current drawn during listening phase [Amps]
                       Ldatau = 200,      # Length of message to be transmitted at the MN [bits]
                       Ldatad = 200,      # Length of message to be received by the MN [bits]
                       Rbu = 1e3,         # Uplink data rate [b/s]
                       Rbd = 1e3,         # Downlink data rate [b/s]
                       ID = 10e-3,        # LED driving current [Amps]
                       Tcycle = 10,       # Total cycle duration [seconds]
                       ISL = 400e-9,      # Sleep mode current [Amps]
                       QmAh = 220)        # battery capacity [mAh]
 ```

