import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

ORIGIN = np.array([0, 0, 0])
EX = np.array([1, 0, 0])
EY = np.array([0, 1, 0])
EZ = np.array([0, 0, 1])
C0 = 3e8
QE = 1.602e-19
I_MAX = 100e-3
P_MAX = 24.5e-3
KB = 1.38e-23
HP = 6.62e-34
BK = 2.8977729e-3                 # Wien's constant
SPEC_POINTS = 200                 # Default number of points used in spectral calculations

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
    
pol_TSFF5210 = np.poly1d([-3, 1, 0])

def interp_on_grid(r, values, x1, x2, p1 = EX, p2 = EY):
    if r.ndim == 1:
        r = r.reshape([1,3])
    sn1 = r[: , 0] * p1[0] + r[: , 1] * p1[1] + r[: , 2] * p1[2]
    sn2 = r[: , 0] * p2[0] + r[: , 1] * p2[1] + r[: , 2] * p2[2]
    xx1, xx2 = np.meshgrid(x1, x2)
    ri = np.array([sn1, sn2]).transpose()    
    return xx1, xx2, griddata(ri, values, (xx1, xx2))

def plot_on_grid(r, values, x1, x2, p1 = EX, p2 = EY):
    xx1, xx2, v = interp_on_grid(r, values, x1, x2, p1 = p1, p2 = p2)
    plt.pcolor(xx1, xx2, v,  shading = 'auto')
    
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
        
def TSFF5210_IP(I):
    if np.isscalar(I):
        is_scalar = True
        I = np.array([I])
    else:
        is_scalar = False        
    
    p = np.zeros(I.shape)
    i1 = np.where(I >= I_MAX)
    p[i1] = P_MAX
    i2 = np.where(I < I_MAX)
    scale = P_MAX / pol_TSFF5210( I_MAX )
    p[i2] = scale * pol_TSFF5210( I[i2] )
    
    if is_scalar:
        p = p[0]
    return p

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
    P = 2.0**HP**2 *C0 ** 2.0 / l**5/(np.exp( HP * C0 / l / KB / T) - 1.0)
    return P 

# wavelength maximum for blackbody radiation
def blackbodymax(T):
    lmax = BK / T
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

class spectra:
    
    white_led = WHITE_LED_SPECTRUM
    ir_led = TSFF5210_SPECTRUM
    pin_resp = RESPONSIVITY
    ir_drop_filter = INFRARED_DROP_FILTER
    visible_drop_filter = VLC_DROP_FILTER
    sun = SUN_SPECTRUM
    
    
def grid_of_points(dr1 = None, dr2 = None, r0 = None, N1 = None, N2 = None, return_dict = False, rm = None):
    ddr1 = dr1 / N1 
    ddr2 = dr2 / N2
    abs_ddr1 = np.linalg.norm(ddr1)
    abs_ddr2 = np.linalg.norm(ddr2)
    
    n1 = np.arange(N1) + 0.5
    n2 = np.arange(N2) + 0.5
    
    nn2, nn1 = np.meshgrid(n2, n1)

    # grid data (can be reliably used only for orthogonal surfaces)
    x1 = nn1 * abs_ddr1 
    x2 = nn2 * abs_ddr2 
  
    nn1 = nn1.reshape(-1)
    nn2 = nn2.reshape(-1)
    
    r = np.zeros([nn1.size, 3])
    
    if r0 is None:
       r0 = rm - dr1 / 2 - dr2 / 2

    # scattered grid
    r[:, 0] = r0[0] + nn1 * ddr1[0] + nn2 * ddr2[0]
    r[:, 1] = r0[1] + nn1 * ddr1[1] + nn2 * ddr2[1]
    r[:, 2] = r0[2] + nn1 * ddr1[2] + nn2 * ddr2[2]
    
    A = np.linalg.norm( np.cross( ddr1, ddr2) )                
    if return_dict:
        return {'r' : r,
                'x1' : x1,
                'x2' : x2,
                'ddr1' : ddr1,
                'ddr2' : ddr2,
                'A' : A}
    else:
        return r

def repeat_row(a, n):
    a = np.squeeze(a)
    return np.repeat( a.reshape([a.size, 1]), n, axis = 1)

def repeat_column(a, n):
    a = np.squeeze(a)
    return np.repeat( a.reshape([1, a.size]), n, axis = 0)


def normalize_to_unity(v):
    vn = np.sqrt( v[:,0] ** 2.0 + v[:,1] ** 2.0 + v[:,2] ** 2.0)
    v2 = np.zeros(v.shape)
    v2[:,0] = v[:,0] / vn
    v2[:,1] = v[:,1] / vn
    v2[:,2] = v[:,2] / vn
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
    
    
def lambertian_gains(rS, nS, rR, nR, mS, AR, FOV):
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
    
    AR2 = repeat_column(AR, rows_S).reshape(-1)
    FOV2 = repeat_column(FOV, rows_S).reshape(-1)

    mS2 = repeat_row(mS, rows_R).reshape(-1)
    
    rRx2 = repeat_column(rRx, rows_S).reshape(-1)
    rRy2 = repeat_column(rRy, rows_S).reshape(-1)
    rRz2 = repeat_column(rRz, rows_S).reshape(-1)
    
    nSx2 = repeat_row(nSx, rows_R).reshape(-1)
    nSy2 = repeat_column(nSy, rows_R).reshape(-1)
    nSz2 = repeat_column(nSz, rows_R).reshape(-1)
    
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
    
    return h.reshape([rows_S, rows_R])
    

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
    
    return np.squeeze(xx)

class txrx_pairings:
    
    def __init__(self, *args, **argv):
        
        rS = argv.pop('rS')
        nS = argv.pop('nS')
        m = argv.pop('m')
        A = argv.pop('A')
        rR = argv.pop('rR')
        nR = argv.pop('nR')
        FOV = argv.pop('FOV')
        PT = argv.pop('PT', None)
        SpecT = argv.pop('SpecT')
        SpecR = argv.pop('SpecR')
        R = argv.pop('R')
        l = argv.pop('l')
                
        if rS.ndim == 1:
            self.nel_tx = 1
            
        else:
            self.nel_tx, _ = rS.shape

        if rR.ndim == 1:
            self.nel_rx = 1

        else:
            self.nel_rx, _ = rR.shape
            
        self.rS = array_if_single_vector(rS, self.nel_tx)
        self.nS = array_if_single_vector(nS, self.nel_tx)
        self.m = vectorize_if_scalar(m, self.nel_tx)
        
        if PT is None:
            PT = np.trapz(SpecT, l)
            
        SpecT = SpecT / np.trapz(SpecT, l)            
            
        self.PT = vectorize_if_scalar(PT, self.nel_tx)
        
        self.rR = array_if_single_vector(rR, self.nel_rx)
        self.nR = array_if_single_vector(nR, self.nel_rx)
        self.A = vectorize_if_scalar(A, self.nel_rx)
        self.FOV = vectorize_if_scalar(FOV, self.nel_rx)
        
        self.SpecT = SpecT / np.trapz(SpecT, l) # Normalized transmission spectra
        self.SpecR = SpecR 
        self.R = R
        self.l = l        
        
    def calc_h(self):
        self.h = lambertian_gains(self.rS, self.nS, self.rR, self.nR, self.m, self.A, self.FOV)
        
    def calc_Pin(self):
        self.Pin = self.h * repeat_row(self.PT, self.nel_rx)
        
    def calc_Pin_tot(self):
        self.Pin_tot = np.sum( self.Pin, axis = 0)
    
    def as_2D_rn(self):
        if self.rS.ndim == 1:
            rS = np.array([self.rS])
        else:
            rS = self.rS

        if self.rR.ndim == 1:
            rR = np.array([self.rR])
        else:
            rR = self.rR

        if self.nS.ndim == 1:
            nS = np.array([self.nS])
        else:
            nS = self.nS

        if self.nR.ndim == 1:
            nR = np.array([self.nR])
        else:
            nR = self.nR
            
        return rS, nS, rR, nR
    
    def show_pos(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')           
        
        rS, nS, rR, nR = self.as_2D_rn()
            
        ax.plot(rS[:,0], rS[:,1], rS[:,2], 'ro')
        ax.quiver(rS[:,0], rS[:,1], rS[:,2] ,
                  nS[:,0], nS[:,1], nS[:,2])
        
        ax.plot(rR[:,0], rR[:,1], rR[:,2], 'bo')
        ax.quiver(rR[:,0], rR[:,1], rR[:,2] ,
                  nR[:,0], nR[:,1], nR[:,2])
                
        plt.show()
        
    def calc_c(self):
        self.coptn = np.trapz( self.SpecT * self.SpecR, self.l )    
        self.celn = np.trapz( self.R * self.SpecT * self.SpecR, self.l )
        self.copt = self.coptn * self.PT
        self.cel = self.celn * self.PT
    
    def calc_PR(self):
        self.PR = self.Pin * self.coptn

    def calc_PR_tot(self):
        self.PR_tot = self.Pin_tot * self.coptn

    def calc_iR(self):
        self.iR = self.Pin * self.celn

    def calc_iR_tot(self):
        self.iR_tot = self.Pin_tot * self.celn        
        
    def calc_power_budget(self):
        self.calc_h()  
        self.calc_c()
        self.calc_Pin()
        self.calc_Pin_tot() 
        self.calc_PR()
        self.calc_PR_tot()
        self.calc_iR()
        self.calc_iR_tot()
        

    def interp_on_grid(self, values, x1, x2, n1, n2):
        xx1, xx2, v = interp_on_grid(self.rR, values, x1, x2, p1 = n1, p2 = n2)
        return xx1, xx2, v
    
class TIA:
    
    def __init__(self, **argv):
        
        self.RF = argv.pop('RF')
        self.CF = argv.pop('CF')
        self.Vn = argv.pop('Vn')
        self.In = argv.pop('In')
        self.fncV = argv.pop('fncV')
        self.fncI = argv.pop('fncI')
        self.temperature = argv.pop('temperature')
        
    def adjust_CF(self, B):
        self.CF = 1 / (2 * np.pi * B * self.RF)
        
    def ZF(self, f):
        return self.RF / (1 + 1j * 2 * np.pi * f * self.CF * self.RF) 

    def RF_psd(self, f):
        return 4 * KB * self.temperature / self.RF * np.ones(f.shape)
    
    def SV_psd(self, f):
        return (self.Vn ** 2.0 + self.Vn ** 2.0 * self.fncV / f) / np.abs(self.ZF(f)) ** 2.0
            
    def SI_psd(self, f):
        return self.In ** 2.0 + self.In ** 2.0 * self.fncI / f
    
    def psd(self, f):
        return self.RF_psd(f) + self.SV_psd(f) + self.SI_psd(f)
    
    def calc_noise_power(self, B):
        
        self.adjust_CF(B)
        f = np.linspace(0.1, B, 1000)        
        self.noise_power = np.trapz(self.psd(f), f) 
        return self.noise_power 

    def plot_psd(self, f = None):          
        if f is None:
            f = np.linspace(10, 2 * self.B, num = 1000)

        plt.loglog(f, self.psd(f))
        plt.xlabel('f [Hz]')
        plt.ylabel('PSD')
        
class sensor_consumption:

    def __init__(self, IWU = 1.3e-3,
                       tWU = 20e-3,
                       IRO = 1.3e-3,
                       tRO = 40e-3,
                       IRX = 1.3e-3,
                       Ldatau = 200,
                       Ldatad = 200,
                       Rbu = 1e3,
                       Rbd = 1e3,
                       ITX = 10e-3,
                       Tcycle = 1,
                       ISL = 400e-9,
                       QmAh = 620):
        
        self.IWU = IWU
        self.tWU = tWU
        self.IRO = IRO
        self.tRO = tRO
        self.Ldatau = Ldatau
        self.Ldatad = Ldatad
        self.Rbu = Rbu
        self.Rbd = Rbd
        self.tTX = Ldatau / Rbu
        self.IRX = IRX
        self.ITX = ITX
        self.tRX = Ldatad / Rbd
        self.Tcycle = Tcycle
        self.tSL = Tcycle - tWU - tRO - self.tTX - self.tRX
        self.ISL = ISL
        self.QmAh = QmAh
        self.Q = self.QmAh * 3600 * 1e-3
        
    def calc_cycle_consumption(self):
        self.con_c = self.tWU * self.IWU + self.tRO * self.IRO + self.tTX * self.ITX + self.tRX * self.IRX + self.tSL * self.ISL
        return self.con_c
    
    def battery_life(self):
        self.calc_cycle_consumption()
        self.Ncycles = self.Q / self.con_c
        self.tbattery = self.Ncycles * self.Tcycle
        return self.tbattery
    
    def battery_life_ITX(self, ITX, Tcycle = None):      
        
        if Tcycle is not None:
            self.Tcycle = Tcycle
            
        tbatt = np.zeros( ITX.shape )
        for i, ITXv in enumerate(ITX):
            self.ITX = ITXv
            tbatt[i] = self.battery_life() / 3600 / 24
        
        return tbatt
   
class sensor_net:
    
    def __init__(self, *args, **kwargs):
        """ 
        The sensor network class
        """
        
        master = kwargs.pop('master')
        self.master = master

        sensors = kwargs.pop('sensors')
        self.sensors = sensors
        
        amb_surfs = kwargs.pop('amb_surfs', [])
        self.amb_surfs = amb_surfs
        
        sensor_driver = kwargs.pop('sensor_driver', None)
        if sensor_driver is not None:
            p = sensor_driver['pol']
            imax = sensor_driver['imax']
            imin = sensor_driver['imin']
            
            sensor_driver['Pmax'] = np.polyval(p, imax)            
            sensor_driver['Pmin'] = np.polyval(p, imin)
            
        self.sensor_driver = sensor_driver
        
        l = kwargs.pop('l')
        self.l = l
        
        self.amp_master = TIA( **master['TIA'] )
        self.amp_sensor = TIA( **sensors['TIA'] )
        
        self.data_rates_u = kwargs.pop('data_rates_u')
        self.data_rates_d = kwargs.pop('data_rates_d')        
        
        if self.master['rS'].ndim == 1:
            self.no_master = 1
        else:
            self.no_master, _ = self.master['rS'].shape
       
        if self.sensors['rS'].ndim == 1:
            self.no_sensors = 1
        else:
            self.no_sensors, _ = self.sensors['rS'].shape
        
    def calc_sensor_I(self, PT):
        
       pinv = self.sensor_driver['polinv']
       return np.polyval(pinv, PT)
            
    def calc_downlink(self):
        """
        Calculate power budget for downlink (master -> sensors)
        """
        self.downlink = txrx_pairings(
                rS = self.master['rS'],
                nS = self.master['nS'],
                rR = self.sensors['rR'],                
                nR = self.sensors['nR'],
                m = self.master['m'],
                A = self.sensors['A'],
                FOV = self.sensors['FOV'],
                PT = self.master.get('PT', None),
                SpecT = self.master['SpecT'],
                SpecR = self.sensors['SpecR'],
                R = self.sensors['R'],
                l = self.l
                )
        
        self.downlink.calc_power_budget()

    def calc_uplink(self):
        """
        Calculate power budget for uplink (master <- sensors)
        """
        self.uplink = txrx_pairings(
                rS = self.sensors['rS'],
                nS = self.sensors['nS'],
                rR = self.master['rR'],                
                nR = self.master['nR'],
                m = self.sensors['m'],
                A = self.master['A'],
                FOV = self.master['FOV'],
                PT = self.sensors.get('PT', None),
                SpecT = self.sensors['SpecT'],
                SpecR = self.master['SpecR'],
                R = self.master['R'],
                l = self.l
                )
        
        self.uplink.calc_power_budget()
    
    
    def calc_ambient_light(self):
        
        self.Pambin_s = np.zeros(self.no_sensors)
        self.Pambin_m = np.zeros(self.no_master)

        self.Pamb_s = np.zeros(self.no_sensors)
        self.Pamb_m = np.zeros(self.no_master)

        self.Iamb_s = np.zeros(self.no_sensors)
        self.Iamb_m = np.zeros(self.no_master)
        
        for source in self.amb_surfs:
             
            pss = txrx_pairings(
                rS = source['rS'],
                nS = source['nS'],
                rR = self.sensors['rR'],                
                nR = self.sensors['nR'],
                m = source['m'],
                A = self.sensors['A'],
                FOV = self.sensors['FOV'],
                SpecT = source['SpecT'],
                SpecR = self.sensors['SpecR'],
                R = self.sensors['R'],
                l = self.l
                )
            
            psm = txrx_pairings(
                rS = source['rS'],
                nS = source['nS'],
                rR = self.master['rR'],                
                nR = self.master['nR'],
                m = source['m'],
                A = self.master['A'],
                FOV = self.master['FOV'],
                SpecT = source['SpecT'],
                SpecR = self.master['SpecR'],
                R = self.master['R'],
                l = self.l
                )
            
            pss.calc_power_budget() 
            psm.calc_power_budget() 

            self.Pambin_s += pss.Pin_tot
            self.Pambin_m += psm.Pin_tot

            self.Pamb_s += pss.PR_tot
            self.Pamb_m += psm.PR_tot
            
            self.Iamb_s += pss.iR_tot
            self.Iamb_m += psm.iR_tot
            
            self.copt_s = pss.coptn
            self.copt_m = psm.coptn
            self.cel_s = pss.celn
            self.cel_m = psm.celn

            
                                      
    def calc_snr(self):
        """
        Calculate signal-to-noise ratios for the uplink and the downlink
        """
        
        self.B_u = self.data_rates_u / self.sensors['sp_eff']
        self.B_d = self.data_rates_d / self.master['sp_eff']
        
        self.p_el_u = np.zeros([self.no_master, self.no_sensors])
        self.p_el_d = np.zeros([self.no_master, self.no_sensors])
        self.p_sh_u = np.zeros([self.no_master, self.no_sensors])
        self.p_sh_d = np.zeros([self.no_master, self.no_sensors])
        self.p_u = np.zeros([self.no_master, self.no_sensors])
        self.p_d = np.zeros([self.no_master, self.no_sensors])
        
        self.snr_el_d = np.zeros([self.no_master, self.no_sensors])
        self.snr_sh_d = np.zeros([self.no_master, self.no_sensors])
        self.snr_d = np.zeros([self.no_master, self.no_sensors])

        self.snr_el_u = np.zeros([self.no_master, self.no_sensors])
        self.snr_sh_u = np.zeros([self.no_master, self.no_sensors])
        self.snr_u = np.zeros([self.no_master, self.no_sensors])
        
        for i in range(self.no_master):
            for j in range(self.no_sensors):
                
                # Remember downlink means i ---> j
                #          uplink   means j ---> i

                self.p_el_u[i, j] = self.amp_master.calc_noise_power( self.B_u[i,j] )
                self.p_el_d[i, j] = self.amp_sensor.calc_noise_power( self.B_d[i,j] )
                
                self.p_sh_u[i, j] = 2 * QE * self.B_u[i,j] * self.Iamb_m[i]
                self.p_sh_d[i, j] = 2 * QE * self.B_d[i,j] * self.Iamb_s[j]
                
                self.p_u[i, j] = self.p_el_u[i, j] + self.p_sh_u[i, j]
                self.p_d[i, j] = self.p_el_d[i, j] + self.p_sh_d[i, j]
                
                self.snr_el_u[i , j] = 0.5 * self.uplink.PR[j, i] / np.sqrt( self.p_el_u[i, j] )
                self.snr_el_d[i , j] = 0.5 * self.downlink.PR[i, j] / np.sqrt( self.p_el_d[i, j] )
                self.snr_sh_u[i , j] = 0.5 * self.uplink.PR[j, i] / np.sqrt( self.p_sh_u[i, j] )
                self.snr_sh_d[i , j] = 0.5 * self.downlink.PR[i, j] / np.sqrt( self.p_el_d[i, j] )
                self.snr_u[i , j] = 0.5 * self.uplink.PR[j, i] / np.sqrt( self.p_u[i, j] )
                self.snr_d[i , j] = 0.5 * self.downlink.PR[i, j] / np.sqrt( self.p_d[i, j] )
                    
        
#plt.close('all')
#
## Wavelength range
#l = np.linspace(200e-9, 1300e-9, 1000)
#
## Master VLC node at the ceiling
#r_m = np.array([2.5, 2.5, 3])
#n_m = np.array([0, 0, -1])
#SpecT_m = WHITE_LED_SPECTRUM(l)
#SpecR_m = VLC_DROP_FILTER(l)
#R_m = RESPONSIVITY(l)
#A_m = 1e-4
#FOV_m = np.pi / 2.0
#m_m = 1
#PT_m = 6
#
## Sensor IR node at the floor
#r_s = grid_of_points(r0 = ORIGIN, dr1 = 5 * EX, dr2 = 5 * EY, N1 = 20, N2 = 20 )
#n_s = np.array([0, 0, 1])
#SpecT_s = TSFF5210_SPECTRUM(l)
#SpecR_s = INFRARED_DROP_FILTER(l)
#R_s = RESPONSIVITY(l)
#A_s = 1e-4
#FOV_s = np.pi / 2.0
#m_s = 45
#PT_s = 25e-3
#
#tia_master = {
#        'RF' : 1e6,
#        'CF' : 1e-9,
#        'Vn' : 15e-9,
#        'In' : 400e-15,
#        'fncI' : 1e3,
#        'fncV' : 1e3,
#        'temperature' : 300 }
#
#tia_nodes = {
#        'RF' : 1e6,
#        'CF' : 1e-9,
#        'Vn' : 15e-9,
#        'In' : 400e-15,
#        'fncI' : 1e3,
#        'fncV' : 1e3,
#        'temperature' : 300 }
#
#master = {
#        'rS' : r_m,
#        'rR' : r_m,
#        'FOV' : FOV_m,
#        'A' : A_m,
#        'm' : m_m,
#        'nS' : n_m,
#        'nR' : n_m,
#        'SpecT' : SpecT_m,
#        'SpecR' : SpecR_m,
#        'R' : R_m,
#        'PT' : PT_m,
#        'TIA' : tia_master,
#        'Rbt' : 1e3,
#        'sp_eff' : 0.4}
#
#sensors = {
#        'rS' : r_s,
#        'rR' : r_s,
#        'FOV' : FOV_s,
#        'A' : A_s,
#        'm' : m_s,
#        'nS' : n_s,
#        'nR' : n_s,
#        'SpecT' : SpecT_s,
#        'SpecR' : SpecR_s,
#        'R' : R_s,
#        'PT' : PT_s,
#        'TIA' : tia_nodes,
#        'Rbt' : 1e3,
#        'sp_eff' : 0.4}
#
#data_rates_u = 1e3 * np.ones([1, 400])
#data_rates_d = 2e3 * np.ones([1, 400])
#
## Window
#pd_peak = 2e9
#rw_d = grid_of_points(dr1 = np.array([1, 0, 0]), 
#                      dr2 = np.array([0, 0, 1]),
#                      rm = np.array([2.5, 5, 1.5]),
#                      N1 = 10,
#                      N2 = 10,
#                      return_dict = True)
#
#rw = rw_d['r']
#Aw = rw_d['A']
#Specw = SUN_SPECTRUM(l) * pd_peak * Aw
#
#window = {        
#        'rS' : rw,
#        'A' : Aw,
#        'm' : 1,
#        'nS' : -EY,
#        'SpecT' : Specw}
#
#amb_surfs = [window]
#
#sn = sensor_net(master = master,
#                sensors = sensors,
#                l = l,
#                data_rates_u = data_rates_u,
#                data_rates_d = data_rates_d,
#                amb_surfs = amb_surfs)
#
#sn.calc_downlink()
#sn.calc_uplink()
#sn.calc_ambient_light()
#
#x1 = np.linspace(0, 5, 100)
#x2 = np.linspace(0, 5, 100)
#plt.figure(1)
#plot_on_grid(sn.sensors['rR'], sn.Pambin_s, x1, x2, p1 = EX, p2 = EY)
#plt.colorbar()
#
#plt.figure(2)
#plot_on_grid(sn.sensors['rR'], sn.Pamb_s, x1, x2, p1 = EX, p2 = EY)
#plt.colorbar()
#
#sn.calc_snr()
# Sensor node positions at the floor


#rS_dict = grid_of_points(dr1 = np.array([1, 0, 0]), 
#                         dr2 = np.array([0, 0, 1]),
#                         rm = np.array([2.5, 5, 1.5]),
#                         N1 = 20,
#                         N2 = 20,
#                         return_dict = True)
#    
#rR_dict = grid_of_points(dr1 = np.array([5, 0, 0]), 
#                         dr2 = np.array([0, 5, 0]),
#                         r0 = np.array([0, 0, 0]),
#                         N1 = 20,
#                         N2 = 20,
#                         return_dict = True)
#
#rS = rS_dict['r']
#rR = rR_dict['r']
#AS = rS_dict['A']
#pd_peak = 2e9
#nS = np.array([0, -1, 0])
#nR = np.array([0, 0, 1])
#A = 1e-4
#FOV = np.pi/2.0
#PT = 1
#m = 1
#pd_peak = 2e9
#
#l = np.linspace(200e-9, 1300e-9, 1000)
##SpecT = WHITE_LED_SPECTRUM(l)
#SpecR = INFRARED_DROP_FILTER(l)
#R = RESPONSIVITY(l)
#SpecT = SUN_SPECTRUM(l) * pd_peak * AS
#
#p = txrx_pairings(rS = rS, 
#                  nS = nS, 
#                  m = m, 
#                  A = A, 
#                  FOV = FOV, 
#                  rR = rR, 
#                  nR = nR,
#                  SpecT = SpecT,
#                  SpecR = SpecR,
#                  R = R,
#                  l = l)
##p.show_pos()
#p.calc_power_budget()
#x1 = np.linspace(0, 5, 100)
#x2 = np.linspace(0, 5, 100)
#
#xx1, xx2, v = p.interp_on_grid(p.Pin_tot, x1, x2, n1 = EX, n2 = EY)
#plt.figure()
#plt.pcolor(xx1, xx2, v, shading = 'auto')
#plt.colorbar()
#
#xx1, xx2, v = p.interp_on_grid(p.iR_tot, x1, x2, n1 = EX, n2 = EY)
#plt.figure()
#plt.pcolor(xx1, xx2, v, shading = 'auto')
#plt.colorbar()
