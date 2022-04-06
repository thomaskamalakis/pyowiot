import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.integrate import cumtrapz
import time

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

def random_directions(n_max, N, m):
    """
    Generate rays according to the Lambertian pattern
    """

    u = np.random.rand(N)
    v = np.random.rand(N)
    z = u ** ( 1 / (m+1) )
    r0 = np.sqrt(1 - z ** 2.0)
    x = r0 * np.cos(2 * np.pi * v)
    y = r0 * np.sin(2 * np.pi * v)
    
    """
    Rotate so that ez is aligned to n_max
    """
    Mrot = rotation_matrix( constants.ez, n_max )
    n = np.zeros( [N, 3] )
    n[:, 0] = Mrot[0, 0] * x + Mrot[0, 1] * y + Mrot[0, 2] * z
    n[:, 1] = Mrot[1, 0] * x + Mrot[1, 1] * y + Mrot[1, 2] * z
    n[:, 2] = Mrot[2, 0] * x + Mrot[2, 1] * y + Mrot[2, 2] * z
    
    return n
    
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

def cart_to_spher(x, y, z):
    r = np.sqrt( x**2.0 + y**2.0 + z**2.0 )
    inc = np.arccos(z / r)
    az = np.arctan2(y, x)
    return r, inc, az

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
    
class grid_of_points:
    
    def __init__(self, dr1 = None, 
                       dr2 = None, 
                       r0 = None, 
                       N1 = None, 
                       N2 = None,
                       rm = None):
        
            self.ddr1 = dr1 / N1 
            self.ddr2 = dr2 / N2
            self.dr1 = dr1
            self.dr2 = dr2
            self.N1 = N1
            self.N2 = N2
            self.abs_dr1 = np.linalg.norm(self.dr1)
            self.abs_dr2 = np.linalg.norm(self.dr2)
            self.abs_ddr1 = np.linalg.norm(self.ddr1)
            self.abs_ddr2 = np.linalg.norm(self.ddr2)
            self.r0 = r0
            self.rm = rm
            
            self.n1 = np.arange(N1) + 0.5
            self.n2 = np.arange(N2) + 0.5
            
            self.nn2, self.nn1 = np.meshgrid(self.n2, self.n1)
        
            # grid data (can be reliably used only for orthogonal surfaces)
            self.x1 = self.nn1 * self.abs_ddr1 
            self.x2 = self.nn2 * self.abs_ddr2 
          
            self.nn1 = self.nn1.reshape(-1)
            self.nn2 = self.nn2.reshape(-1)
            
            self.r = np.zeros([self.nn1.size, 3])
            
            if r0 is None:
               self.r0 = self.rm - self.dr1 / 2 - self.dr2 / 2
        
            # scattered grid
            self.r[:, 0] = self.r0[0] + self.nn1 * self.ddr1[0] + self.nn2 * self.ddr2[0]
            self.r[:, 1] = self.r0[1] + self.nn1 * self.ddr1[1] + self.nn2 * self.ddr2[1]
            self.r[:, 2] = self.r0[2] + self.nn1 * self.ddr1[2] + self.nn2 * self.ddr2[2]
            
            self.A = np.linalg.norm( np.cross( self.ddr1, self.ddr2) )
    
class plane_surface(grid_of_points):
    """
    A plane surface class
    """
    def __init__(self,
                 n = constants.ez,
                 m = 1,
                 l = 1e-6,
                 pd_peak = 0,
                 spectra = spectra.sun,
                 name = None,
                 refl = 1.0,
                 **kwargs):
    
        super().__init__(**kwargs)
        
        self.n = n
        self.SpecT = spectra(l) * pd_peak * self.A
        self.m = m
        self.edges = np.array([
             self.r0,
             self.r0 + self.dr1,
             self.r0 + self.dr1 + self.dr2,
             self.r0 + self.dr2,
             ])
        self.name = name
        self.refl = refl
        
    def contains( self, p ):
        """
        Check if a point p is inside the surface
        """
        p0 = p - self.r0
        
        if np.dot(p0, self.n) == 0:            
            pr_dr1 = np.dot(p0, self.dr1 ) / self.abs_dr1
            pr_dr2 = np.dot(p0, self.dr2 ) / self.abs_dr2
            return 0 <= pr_dr1 <= self.abs_dr1 and 0 <= pr_dr2 <= self.abs_dr2
        else:
            return False
    
    def show(self, ax = None):
        if ax is None:
            fig = plt.figure()            
            ax = fig.add_subplot(111, projection = '3d')                   
        
        x = np.concatenate( (self.edges[:, 0], [self.edges[0, 0] ]) )
        y = np.concatenate( (self.edges[:, 1], [self.edges[0, 1] ]) )
        z = np.concatenate( (self.edges[:, 2], [self.edges[0, 2] ]) )
        
        ax.plot(x, y, z, '-ro')
        
        return ax
        

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
    
    if calc_delays:
        d = RR / C0
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
        SpecT = argv.pop('SpecT', None)
        SpecR = argv.pop('SpecR', None)
        R = argv.pop('R', None)
        l = argv.pop('l', None)
                
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
            if SpecT is not None:
                PT = np.trapz(SpecT, l)
            
            
        self.PT = vectorize_if_scalar(PT, self.nel_tx)
        
        self.rR = array_if_single_vector(rR, self.nel_rx)
        self.nR = array_if_single_vector(nR, self.nel_rx)
        self.A = vectorize_if_scalar(A, self.nel_rx)
        self.FOV = vectorize_if_scalar(FOV, self.nel_rx)
        
        if SpecT is not None:
            SpecT = SpecT / np.trapz(SpecT, l)                    
            self.SpecT = SpecT / np.trapz(SpecT, l) # Normalized transmission spectra
        
        self.SpecR = SpecR 
        self.R = R
        self.l = l        
        
    def calc_h(self):
        self.h, self.d = lambertian_gains(self.rS, self.nS, self.rR, 
                                          self.nR, self.m, self.A, self.FOV,
                                          calc_delays = True)
        
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

class driver:
    
    def __init__(self, 
                 imax = 100e-3,
                 imin = 0e-3,
                 pol = np.array([ 1.35376064e-01,  1.86846949e-01, -1.01789073e-04]),
                 polinv = np.array([-1.74039667e+01, 5.32917840e+00, 5.61867428e-04])
                 ):
        
        self.imax = imax
        self.imin = imin
        self.pol = pol
        self.polinv = polinv
        
        self.Pmax = np.polyval(pol, imax)
        self.Pmin = np.polyval(pol, imin)
        
    def calc_I(self, P):
        return np.polyval(self.polinv, P)
    
    def calc_P(self, I):        
        return np.polyval(self.pol, I)
        
    
class nodes:

    def __init__(self,
                 r = None,
                 FOV = np.pi/2.0,
                 A = None,
                 m = 1,
                 n = -constants.ez,
                 SpecT = None,
                 SpecR = None,
                 R = None,
                 PT = None,
                 TIA = None,
                 sp_eff = 0.4):
        
        self.rS = r
        self.rR = r
        self.FOV = FOV
        self.A = A
        self.m = m
        self.nS = n
        self.nR = n
        self.SpecT = SpecT
        self.SpecR = SpecR
        self.R = R
        self.PT = PT
        self.TIA = TIA
        self.sp_eff = sp_eff

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
                       ID = 10e-3,
                       Tcycle = 1,
                       ISL = 400e-9,
                       QmAh = 620):
        
        self.ID = ID
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
        self.ITX = self.ID * 0.5 + self.IWU
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
    
    def battery_life_tTX(self, tTX, Tcycle = None):      
        
        if Tcycle is not None:
            self.Tcycle = Tcycle
            
        tbatt = np.zeros( tTX.shape )
        for i, tTXv in enumerate(tTX):
            self.tTX = tTXv
            tbatt[i] = self.battery_life() / 3600 / 24
        
        return tbatt
    
    def battery_life_ID(self, ID, Tcycle = None):      
        
        if Tcycle is not None:
            self.Tcycle = Tcycle
            
        tbatt = np.zeros( ID.shape )
        for i, IDv in enumerate(ID):
            self.ITX = 0.5 * IDv + self.IWU
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
        self.sensor_driver = sensor_driver
        
        l = kwargs.pop('l')
        self.l = l
        
        self.amp_master = master.TIA 
        self.amp_sensor = sensors.TIA 
        
        self.data_rates_u = kwargs.pop('data_rates_u')
        self.data_rates_d = kwargs.pop('data_rates_d')        
        
        if self.master.rS.ndim == 1:
            self.no_master = 1
        else:
            self.no_master, _ = self.master.rS.shape
       
        if self.sensors.rS.ndim == 1:
            self.no_sensors = 1
        else:
            self.no_sensors, _ = self.sensors.rS.shape
        
    def calc_sensor_I(self, PT):        
       return self.sensor_driver.calc_I(PT)
            
    def calc_downlink(self):
        """
        Calculate power budget for downlink (master -> sensors)
        """
        self.downlink = txrx_pairings(
                rS = self.master.rS,
                nS = self.master.nS,
                rR = self.sensors.rR,                
                nR = self.sensors.nR,
                m = self.master.m,
                A = self.sensors.A,
                FOV = self.sensors.FOV,
                PT = self.master.PT,
                SpecT = self.master.SpecT,
                SpecR = self.sensors.SpecR,
                R = self.sensors.R,
                l = self.l
                )
        
        self.downlink.calc_power_budget()

    def calc_uplink(self):
        """
        Calculate power budget for downlink (master -> sensors)
        """
        self.uplink = txrx_pairings(
                rS = self.sensors.rS,
                nS = self.sensors.nS,
                rR = self.master.rR,                
                nR = self.master.nR,
                m = self.sensors.m,
                A = self.master.A,
                FOV = self.master.FOV,
                PT = self.sensors.PT,
                SpecT = self.sensors.SpecT,
                SpecR = self.master.SpecR,
                R = self.master.R,
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
                rS = source.r,
                nS = source.n,
                rR = self.sensors.rR,                
                nR = self.sensors.nR,
                m = source.m,
                A = self.sensors.A,
                FOV = self.sensors.FOV,
                SpecT = source.SpecT,
                SpecR = self.sensors.SpecR,
                R = self.sensors.R,
                l = self.l
                )
            
            psm = txrx_pairings(
                rS = source.r,
                nS = source.n,
                rR = self.master.rR,                
                nR = self.master.nR,
                m = source.m,
                A = self.master.A,
                FOV = self.master.FOV,
                SpecT = source.SpecT,
                SpecR = self.master.SpecR,
                R = self.master.R,
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
        
        self.B_u = self.data_rates_u / self.sensors.sp_eff
        self.B_d = self.data_rates_d / self.master.sp_eff
        
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
       

class ray:
    """
    The ray class used in multipath simulations
    """    
    def __init__(self,
                 r = constants.O,
                 t = 0.0,
                 n = -constants.ez,
                 P = 1.0,
                 m = 1, 
                 Mrot = None):
        
        self.r = r
        self.t = t
        self.P = P

        u = np.random.rand(1)
        v = np.random.rand(1)
        z = u ** ( 1 / (m+1) )
        r0 = np.sqrt(1 - z ** 2.0)
        x = r0 * np.cos(2 * np.pi * v)
        y = r0 * np.sin(2 * np.pi * v)
        
        if Mrot is None:
            Mrot = rotation_matrix( constants.ez, n )
            
        self.n = np.matmul(
                     Mrot,
                     np.array([[x, y, z]])                
                )
        
                
    def intersection_plane(self, ps):
        """
        Intersection of a ray with a plane surface
        """  
        c1 = np.dot(ps.r0 - self.r, ps.n)
        c2 = np.dot( ps.n, self.n )
          
        if c2 == 0 and c1 == 0: 
            return np.inf
        elif c2 == 0:
            return None
        else:
            r_int = self.r + self.n * c1/c2
            orientation = r_int - self.r
            if np.dot(orientation, self.n) > 0:
                return r_int
            else:
                return None

class rays:
    """
    The class describing a group of rays
    """
    
    def init_trajectories(self):
        self.trajectories = []
        for i in range(self.N):
            self.trajectories.append([ {'pos' : np.copy(self.r[i]),
                                        'surface' : None,
                                        'orientation' : np.copy(self.n[i]),
                                        'time' : 0.0,
                                        'power' : 1.0,
                                        'max_n' : self.n_max,
                                        'm' : self.m,
                                        'bounce_no' : 0} ])
            
    def __init__(self,
                 r = np.array([constants.O]),
                 n_max = -constants.ez,
                 m = 1,
                 N = None):
        
        if r.ndim == 1:
            r = repeat_column(r, N)
            
        self.r = r
        self.n_max = n_max
        self.m = m
        self.N = N
        
        self.n = random_directions(n_max, N, m)
        self.currently_on = np.ones(N) * np.nan
        self.init_trajectories()
        
    def show(self, ax = None, focus_axis = False):
        """
        Draw rays
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')           

        r = self.r
        n = self.n  
        ax.plot(r[:,0], r[:,1], r[:,2], 'bo')
        ax.quiver(r[:,0], r[:,1], r[:,2] ,
                  n[:,0], n[:,1], n[:,2])
        
        if focus_axis:
            xmin = np.min(r[:,0]) - 1
            xmax = np.max(r[:,0]) + 1
            ymin = np.min(r[:,1]) - 1
            ymax = np.max(r[:,1]) + 1
            zmin = np.min(r[:,2]) - 1
            zmax = np.max(r[:,2]) + 1
            
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_zlim([zmin, zmax])
                
    def orientations(self):
        n = self.n
        return cart_to_spher( n[: , 0], n[: , 1], n[:, 2])
    
    def angle_dist(self, Nbins = 100):
        r, inc, az = self.orientations()
        o_inc, e_inc = np.histogram(inc, bins = Nbins, range = (0, np.pi) )
        o_az, e_az = np.histogram(az, bins = Nbins, range = (-np.pi, np.pi) )
        
        e_inc = 0.5 * (e_inc[1:] + e_inc[:-1] )
        e_az = 0.5 * (e_az[1:] + e_az[:-1] )
        
        o_inc = cumtrapz(o_inc, e_inc, initial = 0)
        o_az = cumtrapz(o_az, e_az, initial = 0)        
        
        return o_inc / np.max(o_inc), e_inc, o_az / np.max(o_az), e_az
    
    def intersection_with_plane(self, ps, check_boundaries = True):
        """
        Intersection points of the rays with a plane surface
        """   
        
        c1 = -dot_2D1D( rel_2D(self.r, ps.r0), ps.n )        
        c2 = dot_2D1D( self.n, ps.n )         
        
        
        i12 = np.where( (c1 == 0) & (c2 == 0) )
        i1 = np.where( (c1 == 0) & (c2 != 0) )
        i = np.where( (c1 != 0) & (c2 != 0) )
        i = i[0]
        r_int = np.zeros( self.r.shape )
        
        r_int[i12, :] = np.inf
        r_int[i1, :] = np.nan
        
        r_int[i] = self.r[i] + mul_2Dsc( self.n[i], c1[i] / c2[i])
        
        dot_r = dot_2D(r_int - self.r, self.n)
        
        ineg = np.where( dot_r <= 0 )
        r_int[ineg] = np.nan
        
        
        if check_boundaries:            
            ipos = np.where( dot_r > 0 )            
            r_can = r_int[ipos]
            
            dot_dr1 = dot_2D1D( rel_2D( r_can, ps.r0), normalize_to_unity(ps.dr1) )
            dot_dr2 = dot_2D1D( rel_2D( r_can, ps.r0), normalize_to_unity(ps.dr2) )

            ind_outside = np.where( (dot_dr1 < 0) | (dot_dr1 > ps.abs_dr1) |
                                    (dot_dr2 < 0) | (dot_dr2 > ps.abs_dr2) )

            r_can[ind_outside] = np.nan
            r_int[ipos] = r_can  
        
        return r_int
    
    
    def bounce(self, pss, bounce_no = None):
        already_bounced = []
        
        """
        Find bouncing points
        """
  
        for i, ps in enumerate(pss):
            
            r_int = self.intersection_with_plane(ps)            
            i_int = np.where( ~np.isnan(r_int).all(axis = 1) )[0].tolist()
            
            in_c = []
            
            for indx in i_int:
                """
                A ray that still has to reach a surface 
                is to be moved if it intersects with a surface if
                different to the one it is currently on
                """
                if (not indx in already_bounced) and (self.currently_on[ indx ] != i):
                    already_bounced.append(indx)
                    in_c.append(indx)
                    
            # generate new directions for the rays already bounced
            n = random_directions( ps.n, len(in_c), 1 )
            
            # change position and orientation of these rays
            
            self.r[ in_c ] = r_int[ in_c ]
            self.n[ in_c ] = n
            self.currently_on[ in_c ] = i
            for c, in_cv in enumerate(in_c):
               tc = self.trajectories[in_cv][-1]['time']

               rstart = self.trajectories[in_cv][-1]['pos']
               d = np.linalg.norm(rstart - self.r[in_cv])
               power = self.trajectories[in_cv][-1]['power']
               self.trajectories[in_cv].append({'pos' : np.copy(self.r[in_cv]),
                                                'surface' : i,
                                                'orientation' : np.copy(self.n[in_cv]),
                                                'max_n' : np.copy( ps.n ),
                                                'power' : ps.refl * power,
                                                'time' : tc + d / C0,
                                                'm' : 1,
                                                'bounce_no' : bounce_no})
    
    def show_trajectories(self, i, ax = None):
        """
        Show the trajectories of the rays
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')           
        
        if np.isscalar(i):
            i = np.array([i])
        else:
            i = np.array(i)
        
        rstart = None

        for iv in i:
            for el in self.trajectories[iv]:
                rend = el['pos']
                n = el['orientation']
                if rstart is not None:
                    x = np.array([rstart[0], rend[0]])
                    y = np.array([rstart[1], rend[1]])
                    z = np.array([rstart[2], rend[2]])
                    ax.plot(x,y,z,'--go')
                    
                else:
                    x = np.array([rend[0]])
                    y = np.array([rend[1]])
                    z = np.array([rend[2]])
                    ax.plot(x,y,z,'go')

                ax.quiver(x[-1], y[-1], z[-1] ,
                          n[0], n[1], n[2])                
                rstart = [x[-1], y[-1], z[-1] ]
    
    def concat_traj(self, attr = 'pos'):

        for i, el in enumerate(self.trajectories):                 
             c = np.array([x[attr] for x in el])
             if i == 0:
                 r = np.copy(c)
             else:
                 r = np.concatenate( (r, c ) )
        return r
          
    def as_transmitters(self):
        """
        Get secondary transmitters from ray trajectories
        """
        return {'r' : self.concat_traj('pos'),
                'n' : self.concat_traj('max_n'),
                't' : self.concat_traj('time'),
                'PT' : self.concat_traj('power'),
                'm' : self.concat_traj('m'),
                'bounce_no' : self.concat_traj('bounce_no')
                }
    
    def get_secondary_transmitters(self, filter_by_bounce = None):
        
        ray_dict = self.as_transmitters()
        rrS = ray_dict['r']
        nnS = ray_dict['n']
        mm = ray_dict['m']
        tt = ray_dict['t']
        PPT = ray_dict['PT']
        bb_no = ray_dict['bounce_no']
        
        if filter_by_bounce is not None:
            indx1 = np.where( ray_dict['bounce_no'] == filter_by_bounce )
        else:
            indx1 = np.arange(1, mm.size, dtype = int)
        
        return rrS[indx1], nnS[indx1], mm[indx1], tt[indx1], PPT[indx1], bb_no[indx1] 
        
    def los(self, rR = None, nR = None, 
                  FOV = None, A = None, 
                  block_size = None,
                  filter_by_bounce = None):
        
        rrS, nnS, mm, tt, PPT, bb_no = self.get_secondary_transmitters(filter_by_bounce = filter_by_bounce)
        
        Nt = mm.size
        
        if rR.ndim == 1:
            Nr = 1
        else:
            Nr, _ = rR.shape
        
        bb_no = bb_no.reshape(bb_no.size, 1)
        if block_size is None:
            block_size = Nt
            
        s = 0
        e = np.min([block_size, Nt])
        
        PR = np.zeros([Nt, Nr])
        td = np.zeros([Nt, Nr])
        bounces = np.zeros([Nt, Nr])
                   
        while s < Nt:
            if s + block_size >= Nt:
                e = Nt
            else:
                e = s + block_size
            
            indxs = slice(s, e, 1)
            rS = rrS[indxs]
            nS = nnS[indxs]
            m = mm[indxs]
            PT = PPT[indxs]
            t = tt[indxs]  
            P = txrx_pairings(
                rS = rS,
                nS = nS,
                PT = PT,
                m = m,
                A = A,
                FOV = FOV,
                rR = rR,
                nR = nR)
            
            P.calc_h()
            P.calc_Pin()
            
            t = repeat_row(t, Nr)
            t_tot = t.reshape(P.d.shape) + P.d
            
            PR[indxs] = P.Pin
            td[indxs] = t_tot

            bounces[indxs,:] = bb_no[indxs]
            
            s = e 
            
        return PR, td, bounces
            
class multipath_sim:
    
    def __init__(self, nrays = 100,
                       surfaces = None,
                       bounces = 3,
                       FOV = np.pi/2.0,
                       nS = -constants.ez,
                       rS = None,
                       nR = constants.ez,
                       rR = None,
                       m = 1,
                       A = 1e-4,
                       t = None, 
                       ray_block_size = None):
        
        self.nrays = nrays
        self.surfaces = surfaces
        self.bounces = bounces
        self.FOV = FOV
        self.nR = nR
        self.nS = nS
        self.rR = rR        
        self.rS = rS
        self.A = A
        self.m = m
        self.t = t
        self.ray_block_size = ray_block_size
               
        if self.rR.ndim == 1:
            self.Nr = 1
        else:
            self.Nr, _ = self.rR.shape
      
        self.Pin_h = np.zeros([self.Nr, self.t.size - 1])
        self.Pin_hb = np.zeros([self.Nr, self.t.size - 1, self.bounces])
        self.Pin_tot_b = np.zeros([self.Nr, self.bounces])
        self.rays_g = 0             # Generated rays so far
        
    def ray_trace(self):
        """
        Generate rays and update trajectories
        """
        self.rays = rays(r = self.rS, n_max = self.nS, m = self.m, N = self.nrays)
 
        for i in range(1, self.bounces+1):
            print('Starting bounce %d' %i)
            self.rays.bounce(self.surfaces, bounce_no = i)        
                    
             
    def los_evals(self):
        self.Pin_s, self.tin_s, self.bin_s = self.rays.los(rR = self.rR, nR = self.nR,
                                                           A = self.A, FOV = self.FOV,
                                                           block_size = self.ray_block_size)
        
    def hist_Pin(self):
        """
        Update the histogram of the receivers incident power
        """
        
        for i in range(self.Nr):
            h, b = np.histogram(self.tin_s[: , i], bins = self.t, weights = self.Pin_s[: , i])
            self.Pin_h[i, :] += np.copy(h)
            
    def hist_Pin_b(self, bounce_no):
        """
        Current histogram of the receivers incident power for bounce bounce_no
        """
        
        Pin_hb = np.zeros([self.Nr, self.t.size - 1])
        for i in range(self.Nr):
            q = np.where(self.bin_s[:, i] == bounce_no)[0]   
            h, b = np.histogram(self.tin_s[q , i], bins = self.t, weights = self.Pin_s[q , i])
            Pin_hb[i, :] = np.copy(h)
        
        return Pin_hb
        
    def update_hists(self):
        """
        Update histograms for all bounces
        """
        for i in range( self.bounces ):
            Pin_hb = self.hist_Pin_b( i )
            self.Pin_hb[:, :, i] += np.copy(Pin_hb)
            self.Pin_tot_b[:, i] += np.sum( Pin_hb, axis = 1 )
        
        self.Pin_h = np.sum(self.Pin_hb, axis = 2)
        self.Pin = np.sum(self.Pin_tot_b, axis = 1)
    
    def get_jr(self, closest_to):
        return np.argmin( np.linalg.norm(self.rR - closest_to, axis = 1) )
   
    def show_h(self, jr = None, closest_to = None):
        plt.figure()
        if jr is None:
            jr = self.get_jr(closest_to)
            
        for i in range(self.bounces):
            plt.plot( self.t[0:-1]/1e-9, self.Pin_hbn[jr, :, i], label = 'Bounce :%s' %i )
        plt.plot(self.t[0:-1]/1e-9, self.Pin_hn[jr, :], '--', label = 'Total')    
        plt.legend()        
        
        plt.xlabel('$t$ [ns]')
        plt.ylabel('$h(t)$')
        rS = self.rR[jr]
        plt.title('Receiver at %s' %np.array2string(rS, separator = ', ' ) )
        return jr

    def calc_norm(self):
        self.Pin_hn = self.Pin_h / self.rays_g
        self.Pin_n = self.Pin / self.rays_g
        self.Pin_hbn = self.Pin_hb / self.rays_g
    
    def iterate(self):
        print('Generating rays and trajectories')
        t0 = time.time()
        self.ray_trace()
        t1 = time.time()
        print('Execution lasted: %6.4f s' %(t1-t0))
        
        print('Beginning LOS evals')
        t2 = time.time()
        self.los_evals()
        t3 = time.time()
        print('Execution lasted: %6.4f s' %(t3-t2))
        
        print('Updating histograms')
        t4 = time.time()
        self.update_hists()
        t5 = time.time()
        print('Execution lasted: %6.4f s' %(t5-t4))            
       
        self.rays_g += self.nrays
        self.calc_norm()
        
    def dict_hist(self):
        return {'Pin_hn' : self.Pin_hn,
                'Pin_n' : self.Pin_n,
                'Pin_hbn' : self.Pin_hb}
        
        
        

        
        
        
        

            
            
        
    
        
        
    
                
        
                
                
                    
                
                
                
        
            
            

        
                    
            
                    
            
            
            
            
            
        
        
        
    

            
        
    
        
        
        
        
        
    
            
        
            
    
        
    
        
