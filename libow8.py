import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import owutils as ut
import multiprocessing as mp
from defaults import constants, defaults
import fnmatch
import matplotlib.tri as tri

D = defaults()
   
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
            self.un_n1 = self.dr1 / self.abs_dr1
            self.un_n2 = self.dr2 / self.abs_dr2

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
            else:
               self.rm = self.r0 + self.dr1 / 2 + self.dr2 / 2
        
            # scattered grid
            self.r[:, 0] = self.r0[0] + self.nn1 * self.ddr1[0] + self.nn2 * self.ddr2[0]
            self.r[:, 1] = self.r0[1] + self.nn1 * self.ddr1[1] + self.nn2 * self.ddr2[1]
            self.r[:, 2] = self.r0[2] + self.nn1 * self.ddr1[2] + self.nn2 * self.ddr2[2]
            
            self.A = np.linalg.norm( np.cross( self.ddr1, self.ddr2) )
            self.nel, _ = self.r.shape
            
         
class plane_surface(grid_of_points):
    """
    A plane surface class
    """
    def __init__(self,
                 n = constants.ez,
                 m = 1,
                 l = 1e-6,
                 name = None,
                 refl = 1.0,
                 PT = 0,
                 **kwargs):
    
        super().__init__(**kwargs)
        
        self.n = n
        self.m = m
        self.PT = 0
        self.edges = np.array([
             self.r0,
             self.r0 + self.dr1,
             self.r0 + self.dr1 + self.dr2,
             self.r0 + self.dr2,
             ])
        self.name = name
        self.refl = refl
        self.x1_u = np.unique(self.x1)        
        self.x2_u = np.unique(self.x2)

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
    
    def interp_on_surf(self, r, v):
        """
        Interpolate a distribution over the surface
        """
            
        v_s = ut.interp_closest( self.r, r, v )
        r_s = ut.project_plane( self.r, self.un_n1, self.un_n2 )
        triang = tri.Triangulation(r_s[:, 0], r_s[:, 1])
        interpolator = tri.LinearTriInterpolator(triang, v_s)
        x1i, x2i = np.meshgrid(self.x1_u, self.x2_u)
        return interpolator(x1i, x2i)

    def plot_on_surf(self, r, v, 
                           show_title = True, 
                           show_colorbar = True,
                           figure_no = None):
        """
        Make a 2D plot of the distribution over the surface
        """
        if figure_no is not None:
            plt.figure( figure_no )
        
        v = self.interp_on_surf(r, v)
        
        plt.pcolor( self.x1_u, self.x2_u, v, shading = 'auto')
        if show_title:
            plt.title( self.name + ' at %s' %self.rm )
            
        if show_colorbar:
            plt.colorbar()

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
                
        if rS.ndim == 1:
            self.nel_tx = 1
            
        else:
            self.nel_tx, _ = rS.shape

        if rR.ndim == 1:
            self.nel_rx = 1

        else:
            self.nel_rx, _ = rR.shape
            
            
        self.rS = ut.array_if_single_vector(rS, self.nel_tx)
        self.nS = ut.array_if_single_vector(nS, self.nel_tx)
        self.m = ut.vectorize_if_scalar(m, self.nel_tx)
        self.PT = ut.vectorize_if_scalar(PT, self.nel_tx)
        
        self.rR = ut.array_if_single_vector(rR, self.nel_rx)
        self.nR = ut.array_if_single_vector(nR, self.nel_rx)
        self.A = ut.vectorize_if_scalar(A, self.nel_rx)
        self.FOV = ut.vectorize_if_scalar(FOV, self.nel_rx)
        
        
    def calc_h(self):
        self.h, self.d = ut.lambertian_gains(self.rS, self.nS, self.rR, 
                                          self.nR, self.m, self.A, self.FOV,
                                          calc_delays = True)
        return self.h
        
    def calc_Pin(self):
        self.Pin = self.h * ut.repeat_row(self.PT, self.nel_rx)
        
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
        
    def interp_on_grid(self, values, x1, x2, n1, n2):
        xx1, xx2, v = ut.interp_on_grid(self.rR, values, x1, x2, p1 = n1, p2 = n2)
        return xx1, xx2, v

class txrx_elements:
    """
    A number of transceiver elements
    """
    def __init__(self, *args, **kwargs):
                 
       if len(args) == 0:
            
            r = kwargs.pop('r')
            if 'n' in kwargs:
                n = kwargs.pop('n')
                nS = np.copy(n)
                nR = np.copy(n)
                
            else:
                nS = kwargs.pop('nS')
                nR = kwargs.pop('nR')
                
            m = kwargs.pop('m', 1)
            refl = kwargs.pop('refl', 1.0)
            A = kwargs.pop('A',None)
            PT = kwargs.pop('PT', 0)
            
       
       else:
           list_of_surfs = args[0]
           
           for i, surf in enumerate(list_of_surfs):
               nel = surf.nel                   
               r0 = ut.array_if_single_vector( surf.r, nel )
               n0 = ut.array_if_single_vector( surf.n, nel )
               m0 = ut.vectorize_if_scalar( surf.m, nel )
               A0 = ut.vectorize_if_scalar( surf.A, nel )
               refl0 = ut.vectorize_if_scalar( surf.refl, nel )
               
               if hasattr(surf, 'PT'):
                   PT0 = ut.vectorize_if_scalar( surf.PT, nel )
               else:
                   PT0 = ut.vectorize_if_scalar( 0.0, nel )
                   
               if (i == 0):
                 r = np.copy( r0 )
                 n = np.copy( n0 )
                 m = np.copy( m0 )
                 A = np.copy( A0 )
                 refl = np.copy( refl0 )
                 PT = np.copy( PT0 )
                            
               else:
                 r = np.concatenate( (r,r0), axis = 0 )
                 n = np.concatenate( (n,n0), axis = 0 )
                 m = np.concatenate( (m,m0), axis = 0 )
                 A = np.concatenate( (A,A0), axis = 0 )
                 refl = np.concatenate( (refl,refl0), axis = 0 )
                 PT = np.concatenate( (PT,PT0), axis = 0 )
                 
           nS = np.copy(n)
           nR = np.copy(n)
            
       if r.ndim == 1:
            self.nel = 1
            
       else:
            self.nel, _ = r.shape

       FOV = kwargs.pop('FOV', np.pi/2.0)
 
       self.rS = ut.array_if_single_vector(r, self.nel)
       self.rR = self.rS
       self.nS = ut.array_if_single_vector(nS, self.nel)
       self.nR = ut.array_if_single_vector(nR, self.nel)
       self.m = ut.vectorize_if_scalar(m, self.nel)
       self.refl = ut.vectorize_if_scalar(refl, self.nel)              
       self.A = ut.vectorize_if_scalar(A, self.nel)       
       self.FOV = ut.vectorize_if_scalar(FOV, self.nel)       
       self.PT = ut.vectorize_if_scalar(PT, self.nel)
       self.Pin_tot = np.zeros([1,self.nel]) 
       
    def set_PT(self, PT):
        self.PT = PT
        
    def set_PT_from_Pin(self, Pin):
        self.PT = self.refl * Pin
    
    def txrx_pair_from(self, source):
        txrx = txrx_pairings(
                rS = source.rS,
                nS = source.nS,
                m = source.m,
                PT = source.PT,
                rR = self.rR,
                nR = self.nR,
                A = self.A,
                FOV = self.FOV
                )

        txrx.calc_h()
        return txrx
    
    def power_from(self, source, txrx = None):
        if txrx is None:
            txrx = self.txrx_pair_from( source )
            
        txrx.calc_Pin()
        txrx.calc_Pin_tot()
        self.Pin = txrx.Pin
        self.Pin_tot = txrx.Pin_tot
        self.PT = self.refl * self.Pin_tot
        
    def power_exchange(self):
        self.power_from(self, txrx = self.slf_txrx_pairs )
        
    def show(self, ax = None):
     
        if ax is None:
            fig = plt.figure()            
            ax = fig.add_subplot(111, projection = '3d')                   
        
        x = self.rR[:, 0]
        y = self.rR[:, 1]
        z = self.rR[:, 2]
        nx = self.nR[:, 0]
        ny = self.nR[:, 1]
        nz = self.nR[:, 2]
        
        ax.plot(x, y, z, 'ro')
        ax.quiver(x, y, z,
                  nx, ny, nz)

        return ax
                
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
        return 4 * constants.kB * self.temperature / self.RF * np.ones(f.shape)
    
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
        I = np.polyval(self.polinv, P)
        i = np.where(I >= self.imax)
        I[i] = np.inf
        j = np.where(P >= self.Pmax)
        I[j] = np.inf        
        return I
    
    def calc_P(self, I):        
        return np.polyval(self.pol, I)
        
    
class cycle:

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
                       QmAh = 620,
                       ID_max = 100.0e-3,
                       ID_min = 0.0):
        
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
        self.ID_max = ID_max
        self.ID_min = ID_min
        
    def calc_cycle_consumption(self):
        if (self.ID > self.ID_max) or (self.ID < self.ID_min):
           con = np.inf
        else:
           con = self.tWU * self.IWU + self.tRO * self.IRO + self.tTX * self.ITX + self.tRX * self.IRX + self.tSL * self.ISL
        self.con_c = con
        
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
        if ID <= self.ID_max and ID >= self.ID_min:            
            self.ITX = 0.5 * ID + self.IWU
            tbatt = self.battery_life() / 3600 / 24
        else:
            tbatt = 0.0
            
        return tbatt
    
    def no_dev(self):
        return self.Tcycle / (self.Tcycle - self.tSL)
       
class group_of_rays:
    """
    Group of rays class
    """
    def __init__(self, nrays = 100,
                       nS = -constants.ez,
                       rS = None,
                       m = 1,
                       surfs = None):
        
        self.i = np.arange(nrays, dtype = int)
        self.nrays = nrays
               
        self.nt = int ( np.floor(rS.size) / 3.0 )
        self.rS = ut.expand_to(rS, self.nt)
        self.nS = ut.expand_to(nS, self.nt)
        self.it = np.mod(self.i, self.nt, dtype = int)
        
        self.r0 = self.rS[ self.it ]
        self.n0 = self.nS[ self.it ]
        self.m = m
        
        self.trajectories = []
        
        self.r = np.copy(self.r0)
        
        self.n = ut.random_directions_mul(self.n0, self.nrays, self.m)
        
        self.t = np.zeros( nrays )
        self.P = np.ones( nrays )
        
        self.b = 0
        self.surfs = surfs
        self.now_on = -np.ones( nrays )
        self.append_to_traj()
        self.nrays_t = int( np.floor( self.nrays / self.nt ) )
        
    def append_to_traj(self):
    
        self.trajectories.append({'r' : np.copy(self.r),
                                  'n' : np.copy(self.n),
                                  't' : np.copy(self.t),
                                  'bounce' : self.b,
                                  'P' : np.copy(self.P),
                                  'now_on' : np.copy(self.now_on),
                                  'n0' : np.copy(self.n0)})
            
    def plane_inter(self, ps):
        """
        Intersection points of the rays with a plane         
        """   
        c1 = -ut.dot_2D1D( ut.rel_2D(self.r, ps.r0), ps.n )        
        c2 = ut.dot_2D1D( self.n, ps.n )         
        
        
        i12 = np.where( (c1 == 0) & (c2 == 0) )
        i1 = np.where( (c1 == 0) & (c2 != 0) )
        i = np.where( (c1 != 0) & (c2 != 0) )
        i = i[0]
        r_int = np.zeros( self.r.shape )
        
        r_int[i12, :] = np.inf
        r_int[i1, :] = np.nan
        
        r_int[i] = self.r[i] + ut.mul_2Dsc( self.n[i], c1[i] / c2[i])
        
        dot_r = ut.dot_2D(r_int - self.r, self.n)
        
        ineg = np.where( dot_r <= 0 )
        r_int[ineg] = np.nan
        
        return r_int, dot_r
        
        
    def ps_inter(self, ps):
        """
        Intersection points of the rays with a plane surface
        """   
        
        r_int, dot_r = self.plane_inter(ps)
        
        ipos = np.where( dot_r > 0 )            
        r_can = r_int[ipos]
        
        dot_dr1 = ut.dot_2D1D( ut.rel_2D( r_can, ps.r0), ut.normalize_to_unity(ps.dr1) )
        dot_dr2 = ut.dot_2D1D( ut.rel_2D( r_can, ps.r0), ut.normalize_to_unity(ps.dr2) )

        ind_outside = np.where( (dot_dr1 < 0) | (dot_dr1 > ps.abs_dr1) |
                                (dot_dr2 < 0) | (dot_dr2 > ps.abs_dr2) )

        r_can[ind_outside] = np.nan
        r_int[ipos] = r_can  
    
        return r_int
    
    def bounce(self):
        already_bounced = []
        
        """
        Find bouncing points
        """
  
        for i, ps in enumerate(self.surfs):
            
            r_int = self.ps_inter(ps)            
            i_int = np.where( ~np.isnan(r_int).all(axis = 1) )[0].tolist()
            
            in_c = []
            
            for indx in i_int:
                """
                A ray that still has to reach a surface 
                is to be moved if it intersects with a surface if
                different to the one it is currently on
                """
                if (not indx in already_bounced) and (self.now_on[ indx ] != i):
                    already_bounced.append(indx)
                    in_c.append(indx)
                    
            # generate new directions for the rays already bounced
            n = ut.random_directions( ps.n, len(in_c), ps.m )
            
            # change position and orientation of these rays
            
            d = np.linalg.norm(self.r[in_c] - r_int[ in_c ], axis = 1)
            
            self.r[ in_c ] = r_int[ in_c ]
            self.n[ in_c ] = n
            self.now_on[ in_c ] = i
            self.P[ in_c ] = ps.refl * self.P[ in_c ]
            self.t[ in_c ] += d / constants.c0
            self.n0[ in_c ] = ps.n
        self.b += 1
        self.append_to_traj()
        
    def show_trajectories(self, i, ax = None):
        """
        Show the trajectories of the rays
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')           
        
        rstart = None

        for el in self.trajectories:
            rend = el['r'][i]
            n = el['n'][i]
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
            
        return ax
    
    def calc_los(self, rR, nR, AR, FOV):
        """
        Calculate line-of-sight components from the rays 
        to the receivers described by rR, nR, AR and FOV
        """
        P = txrx_pairings(
            rS = self.r,
            nS = self.n0,
            PT = self.P,
            m = self.m,
            A = AR,
            FOV = FOV,
            rR = rR,
            nR = nR)
        P.calc_h()
        P.calc_Pin() 
       
        nr = int( np.floor( rR.size / 3 ) )        
        Pin_raw = np.transpose(P.Pin)
        ft_raw = np.tile( np.transpose(self.t), (nr, 1) ) + np.transpose(P.d)  

        Pin = np.zeros( [self.nt, nr, self.nrays_t] )
        ft = np.zeros( [self.nt, nr, self.nrays_t] )
        
        for i in range(self.nt):
            Pin[i, :, :] = Pin_raw[:, slice(i, self.nrays, self.nt) ]
            ft[i, :, :] = ft_raw[:, slice(i, self.nrays, self.nt) ]
        
        return Pin, ft, Pin_raw, ft_raw

class diffuse_sim:

    def __init__(self, surfaces = None,
                       bounces = 3,
                       rS = None,
                       nS = -constants.ez,
                       rR = None,
                       nR = +constants.ez,
                       m = 1,
                       AR = 1e-4,
                       FOV = np.pi / 2.0,
                       PT = 1):
        
        if rS.ndim == 1:
            self.nt = 1
        else:
            self.nt, _ = rS.shape
        
        if rR.ndim == 1:
            self.nr = 1
        else:
            self.nr, _ = rR.shape
  
        self.surfaces = surfaces
        self.rS = ut.array_if_single_vector(rS, self.nt)
        self.nS = ut.array_if_single_vector(nS, self.nt)
        self.rR = ut.array_if_single_vector(rR, self.nr)
        self.nR = ut.array_if_single_vector(nR, self.nr)
        self.AR = ut.vectorize_if_scalar(AR, self.nr)
        self.m = ut.vectorize_if_scalar(m, self.nt)
        self.bounces = bounces
        self.PT = ut.vectorize_if_scalar(PT, self.nt)
        self.FOV = ut.vectorize_if_scalar(FOV, self.nr)
        self.Pin = np.zeros([self.nt, self.nr, bounces + 1])
    
        # Initialize simulation elements
        self.surf_elements = txrx_elements( self.surfaces )        

        self.tx_elements = txrx_elements(
                        r = rS,
                        n = nS,
                        m = m,
                        PT = PT)

        self.rx_elements = txrx_elements(
                        r = self.rR,
                        n = self.nR,
                        FOV = self.FOV,                          
                        A = self.AR)
        
    def calc_Pin_tr(self, i):

        print('Starting calculations for transmitter: ', i)                
        self.rx_elements.power_from(self.tx_elements)
        self.Pin[i, :, 0] = self.rx_elements.Pin[i]
        self.surf_elements.power_from( self.tx_elements )
            
        for b in range(self.bounces):
            self.rx_elements.power_from( self.surf_elements )
            self.surf_elements.power_exchange()
            self.Pin[ i, :, b + 1 ] = np.copy(self.rx_elements.Pin_tot)
        
    def calc_Pin(self):        
        
        for i, rS in enumerate(self.rS):
            print('Starting iterations for %d / %d' %(i, self.nt) )
            self.calc_Pin_tr(i)          
            
    def calc_Pin_mp(self, no_procs = None):
        """
        Calculate diffuse power using multiprocessing
        """
        if no_procs is None:
            no_procs = mp.cpu_count()
        
        pool = mp.Pool( no_procs )
        pool.map( self.calc_Pin_tr, np.arange(0, self.nt, dtype = int) )            
        pool.close()
            
            
    def plot_on_surf(self, surf, save_prefix = None):
        for i in range( self.bounces + 1 ):
            plt.figure()
            
            ut.plot_on_grid(surf.r, self.Pin[:, i], 
                         surf.x1, 
                         surf.x2, 
                         p1 = surf.un_n1,
                         p2 = surf.un_n2)
            plt.title('Bounce no = ' + str(i) )            
            plt.colorbar()
            if save_prefix is not None:
                plt.savefig( save_prefix + str(i) + '.png')
    
    def to_dict(self):
        return {
          'rR' : self.rR,
          'nR' : self.nR,
          'Pin' : self.Pin,
          'AR' : self.AR,
          'm' : self.m,
          'nS' : self.nS,
          'rS' : self.rS,
          'FOV' : self.FOV,
          'PT' : self.PT
          }      
        
    def save_to(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump( self.to_dict(), f)
                
class multipath_sim:
    """
    Multipath simulation class
    """
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
                       t = None):
        
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
        
        self.Nt = int( np.floor( rS.size / 3 ) )
        self.Nr = int( np.floor( rR.size / 3 ) )
        
        self.rS = ut.expand_to(rS, self.Nt)
        self.nS = ut.expand_to(nS, self.Nt)        
        self.rR = ut.expand_to(rR, self.Nr)
        self.nR = ut.expand_to(nR, self.Nr)
        
        self.h = np.zeros([self.Nt, self.Nr, self.t.size - 1])                  # received power with respect to t
        self.hb = np.zeros([self.Nt, self.Nr, self.bounces + 1, self.t.size - 1])   # received power per bounce with respect to t
        self.P = np.zeros([self.Nt, self.Nr])                                   # total power
        self.Pb = np.zeros([self.Nt, self.Nr, self.bounces + 1])                    # total power per bounce
        self.rays_g = 0                                                         # Generated rays so far
        self.timings = {}
        
    def time(func):
        def wrapper(self):            
            self.tstart = time.time()
            res = func(self)
            self.tend = time.time()
            self.telapsed = self.tend - self.tstart
            print('Execution lasted %f s' %self.telapsed)
            fun_name = func.__name__
            
            if fun_name not in self.timings:
                self.timings[fun_name] = 0.0
            self.timings[fun_name] += self.telapsed
            return res
        return wrapper           
    
    
    def init_rays(self):
        """
        Initialize the rays to be used in the current execution
        """
        self.rays = group_of_rays(nrays = self.nrays,
                                  surfs = self.surfaces,
                                  rS = self.rS,
                                  nS = self.nS,
                                  m = self.m)
        self.rays_g += self.nrays
        
    @time
    def single_bounce(self):
        self.rays.bounce()
    
    @time    
    def calc_los(self):
        self.Pin, self.ft, _, _ = self.rays.calc_los( self.rR, self.nR, self.A, self.FOV)

    @time
    def update_hists(self):
        for i in range(self.Nt):
            for j in range(self.Nr):
                h, _ = np.histogram(self.ft[i, j, :], bins = self.t, weights = self.Pin[i, j, :])
                self.hb[i, j, self.rays.b, :] += h
                self.h[i, j, :] += h
                
                P = np.sum(h)
                self.Pb[ i, j, self.rays.b ] += P
                self.P[ i, j ] += P                                
        
    def get_jr(self, closest_to):
        return np.argmin( np.linalg.norm(self.rR - closest_to, axis = 1) )
   
    def calc_norm(self):
        self.hn = self.h / (self.rays_g / self.Nt)
        self.hbn = self.hb / (self.rays_g / self.Nt)
        self.Pn = self.P / (self.rays_g / self.Nt)
        self.Pbn = self.Pb / (self.rays_g / self.Nt)
                
    def simulate(self):
        print('Generating rays and trajectories')
        self.init_rays()
        print('Performing LOS calculations')
        self.calc_los()
        print('Updating histograms')
        self.update_hists()        
        
        for i in range(self.bounces):
            print('Starting bounce %d / %d' %(i, self.bounces) )
            self.single_bounce()
            print('Performing LOS calculations ' )
            self.calc_los()
            print('Updating histograms')
            self.update_hists()
        
        self.calc_norm()
    
    def show_h(self, it, ir, bounce = None, fig = None):
        """
        Show the impulse response obtained for the transceiver set (it, ir)
        """
        if fig is None:
            fig = plt.figure()        
        
        plt.figure(fig)
        plt.xlabel('Time [ns]')
        plt.ylabel('Impulse response')
        title = 'rS = %s\n rR = %s' %(np.array2string(self.rS[it], separator = ','),                                                      
                                      np.array2string(self.rR[ir], separator = ',') )
        plt.title(title)
        t = self.t
        
        if bounce is None:
            plt.plot(t[:-1]/1e-9, self.hn[it, ir, :])
            return fig
        
        elif bounce == 'all':
            ib = range( self.bounces + 1 )
        
        else:
            ib = bounce
            
        for ic in ib:
            plt.plot(t[:-1]/ 1e-9, self.hbn[it, ir, ic, :], label = 'bounce: %s' %ic )
            
        plt.legend()
        return fig
            
    def to_dict(self):
        return {'h' : self.hn,
                'hbn' : self.hbn,
                'Pn' : self.Pn,
                'Pbn' : self.Pbn,
                'rS' : self.rS,
                'nS' : self.nS,
                'rR' : self.rR,
                'nR' : self.nR,
                'Nt' : self.Nt,
                'Nr' : self.Nr,
                't' : self.t,
                'rays_g' : self.rays_g,
                'timings' : self.timings,
                'FOV' : self.FOV,
                'A' : self.A,
                'bounces' : self.bounces,
                'm' : self.m}
        
    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump( vars(self), f)

class sensor_net:

    REQUIRED_PARAMS = ['r_master', 'nS_master', 'nR_master', 
                   'r_sensor', 'nS_sensor', 'nR_sensor',
                   'A_master', 'A_sensor', 'm_master', 'm_sensor',
                   'RF_master', 'RF_sensor', 'CF_master', 'CF_sensor',
                   'sp_eff_master', 'sp_eff_sensor', 'FOV_master', 'FOV_sensor',
                   'pd_peak', 'ST_m', 'SR_m', 'ST_s', 'SR_s',
                   'ST_a', 'FOV_master', 'FOV_sensor', 'l', 'R_s', 'R_m',
                   'refl_floor', 'refl_ceiling', 'refl_east', 'refl_west',
                   'refl_south', 'refl_north', 'room_N', 'room_L', 'room_W', 'room_H',
                   'amb_pos', 'amb_H', 'amb_L1', 'amb_L2', 'ptd_a', 'amb_name',
                   'PT_master', 'PT_sensor', 'no_bounces',
                   'Vn_master', 'Vn_sensor', 'In_master', 'In_sensor',
                   'fncI_master', 'fncI_sensor', 'fncV_master', 'fncV_sensor', 'temperature',
                   'Rb_master', 'Rb_sensor',
                   'IWU', 'tWU', 'IRO', 'tRO', 'IRX', 'bits_sensor',  'bits_master', 
                   'ID_sensor', 'Tcycle', 'ISL', 'QmAh',
                   'BER_target', 'md_pol', 'md_poli', 'sd_pol', 'sd_poli',
                   'Imax_s', 'Imin_s', 'Imax_m', 'Imin_m' ]

    MUST_VECTORIZE = {
        'PT_master' : 'no_masters',
        'PT_sensor' : 'no_sensors',
        'Rb_master' : 'no_masters',
        'Rb_sensor' : 'no_sensors',
        'IWU' : 'no_sensors',
        'tWU' : 'no_sensors',
        'IRO' : 'no_sensors',
        'tRO' : 'no_sensors',
        'IRX' : 'no_sensors',
        'bits_sensor' : 'no_sensors',
        'bits_master' : 'no_masters',
        'ID_sensor' : 'no_sensors',
        'Tcycle' : 'no_sensors',
        'ISL' : 'no_sensors',
        'QmAh' : 'no_sensors'
        }    

    def set_params(self, kwargs):
        """
        Setup parameters. If the required parameters are not specified, use
        default values.
        """
        
        for key in self.REQUIRED_PARAMS:
            if key in kwargs:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, getattr(D,key) )
        
        self.set_no_els()
        
        if 'walls' in kwargs:
            self.walls = kwargs['walls']
        else:
            self.set_default_walls()
            
        if 'amb_surfs' in kwargs:
            self.amb_surfs = kwargs['amb_surfs']
        else:
            self.set_default_amb()
            
        if 'PT_sensor' in kwargs:
            self.PT_sensor = kwargs['PT_sensor']
        else:
            self.PT_sensor = 0
            
        for key, no_attr in self.MUST_VECTORIZE.items():
            if np.isscalar( getattr(self, key) ):
                n = getattr( self, no_attr )
                setattr( self, key, 
                         ut.vectorize_if_scalar( getattr( self, key ), n ) 
                       )
        
        self.set_TIAs()
        self.B_master = self.Rb_master / self.sp_eff_master
        self.B_sensor = self.Rb_sensor / self.sp_eff_sensor
        self.set_drivers()
        
    def __init__(self, *args, **kwargs):
        """ 
        The sensor network class
        """
        self.set_params( kwargs )
        self.calc_spec_match()
        self.set_elements()
        
    def set_no_els(self):
        if self.r_master.ndim == 1:
            self.no_masters = 1
        else:
            self.no_masters, _ = self.r_master.shape
       
        if self.r_sensor.ndim == 1:
            self.no_sensors = 1
        else:
            self.no_sensors, _ = self.r_sensor.shape

    def set_default_walls(self):  
        L = self.room_L
        W = self.room_W
        H = self.room_H
        N = self.room_N
        r_f = self.refl_floor
        r_c = self.refl_ceiling
        r_w = self.refl_west
        r_s = self.refl_south
        r_n = self.refl_north
        r_e = self.refl_east
        
        self.walls = [ 
           plane_surface( r0 = constants.O, 
                          n = constants.ez,
                          dr1 = L * constants.ex, 
                          dr2 = W * constants.ey, 
                          N1 = N, 
                          N2 = N,
                          refl = r_f,
                          name = 'floor' ),
           plane_surface( r0 = np.array([0, 0, H]), 
                          n = -constants.ez,
                          dr1 = L * constants.ex, 
                          dr2 = W * constants.ey, 
                          N1 = N, 
                          N2 = N,
                          refl = r_c,
                          name = 'ceiling' ),
           plane_surface( r0 = np.array([0, 0, 0]), 
                          n = constants.ex,
                          dr1 = W * constants.ey, 
                          dr2 = H * constants.ez, 
                          N1 = N, 
                          N2 = N,
                          refl = r_w,
                          name = 'west wall'),
           plane_surface( r0 = np.array([L, 0, 0]), 
                          n = -constants.ex,
                          dr1 = W * constants.ey, 
                          dr2 = H * constants.ez, 
                          N1 = N, 
                          N2 = N,
                          refl = r_e,
                          name = 'east wall'),
           plane_surface( r0 = np.array([0, W, 0]), 
                          n = -constants.ey,
                          dr1 = L * constants.ex, 
                          dr2 = H * constants.ez, 
                          N1 = N, 
                          N2 = N,
                          refl = r_s,
                          name = 'south wall'),
           plane_surface( r0 = np.array([0, 0, 0]), 
                          n = constants.ey,
                          dr1 = L * constants.ex, 
                          dr2 = H * constants.ez, 
                          N1 = N, 
                          N2 = N,
                          refl = r_n,
                          name = 'north wall')
           ]
   
    def get_wall_by_name(self, pattern):
        l = [ w for w in self.walls if fnmatch.fnmatch(w.name, pattern) ]
        if len(l) == 1:
            return l[0]
        else:
            return l
        
    def set_default_amb(self):
        
        s_list = [ x for x in self.walls if x.name == self.amb_pos ]
        s = s_list[0]
        dr1 = self.amb_L1 * s.un_n1
        dr2 = self.amb_L2 * s.un_n2
        window = plane_surface(
                        dr1 = dr1, 
                        dr2 = dr2,
                        rm = s.rm,
                        N1 = self.room_N,
                        N2 = self.room_N,
                        name = self.amb_name,
                        n = s.n,
                        m = 1)    
        window.PT = self.pd_peak * window.A * self.ptd_a
        self.amb_surfs = [ window ]
        
    def calc_spec_match(self):
        """
        Calculate spectral matchings
        """        
        self.cel_sm = np.trapz( self.ST_m * self.SR_s * self.R_s, self.l )
        self.cel_ms = np.trapz( self.ST_s * self.SR_m * self.R_m, self.l )
        self.cel_sa = np.trapz( self.ST_a * self.SR_s * self.R_s, self.l )
        self.cel_ma = np.trapz( self.ST_a * self.SR_m * self.R_m, self.l )
    
    def plot_spectra(self):
        """
        Plot all spectra involved
        """
        plt.figure()
        l = self.l
        plt.plot(l, self.ST_m / np.max( self.ST_m ), 'b', label = '$S_\mathrm{T}$ - master')
        plt.plot(l, self.SR_m / np.max( self.SR_m ), 'r', label = '$S_\mathrm{R}$ - master')
        plt.plot(l, self.R_m / np.max( self.R_m ), 'k', label = '$\mathcal{R}$ - master')
        plt.plot(l, self.ST_s / np.max( self.ST_s ), 'b--', label = '$S_\mathrm{T}$ - sensor')
        plt.plot(l, self.SR_s / np.max( self.SR_s ), 'r--', label = '$S_\mathrm{R}$ - sensor')
        plt.plot(l, self.R_s / np.max( self.R_s ), 'k--', label = '$\mathcal{R}$ - sensor')
        plt.plot(l, self.ST_a / np.max( self.ST_a ), 'g--', label = '$S_\mathrm{T}$ - ambient')
        plt.legend()
          
    def set_elements(self):
        """
        Calculate the simulation elements
        """
        self.m_els = txrx_elements(
                r = self.r_master,
                nS = self.nS_master,
                nR = self.nR_master,
                A = self.A_master,
                m = self.m_master,
                FOV = self.FOV_master,
                PT = self.PT_master
                )
        
        self.s_els = txrx_elements(
                r = self.r_sensor,
                nS = self.nS_sensor,
                nR = self.nR_sensor,
                A = self.A_sensor,
                m = self.m_sensor,
                FOV = self.FOV_sensor,
                PT = self.PT_sensor
                )
        
        self.w_els = txrx_elements(
                self.walls
                )               
        
        self.a_els = txrx_elements(
                self.amb_surfs
                )
        
        self.n_w = self.w_els.nel
        self.n_a = self.a_els.nel
        
    def set_TIAs(self):
        self.tia_m = TIA(
            RF = self.RF_master,
            CF = self.CF_master,
            Vn = self.Vn_master,
            In = self.In_master,
            fncV = self.fncV_master,
            fncI = self.fncI_master,
            temperature = self.temperature
        )
        
        self.tia_s = TIA(
            RF = self.RF_sensor,
            CF = self.CF_sensor,
            Vn = self.Vn_sensor,
            In = self.In_sensor,
            fncV = self.fncV_sensor,
            fncI = self.fncI_sensor,
            temperature = self.temperature
        )
    
    def set_drivers(self):
        self.driver_m = driver(
            imax = self.Imax_m,
            imin = self.Imin_m,
            pol = self.md_pol,
            polinv = self.md_poli
        )
        
        self.driver_s = driver(
            imax = self.Imax_s,
            imin = self.Imin_s,
            pol = self.sd_pol,
            polinv = self.sd_poli
        )
    def calc_tbattery(self, ID = None):
        if ID is None:
            ID = np.squeeze(self.ID_rq_s_tot)
            
        self.cycles = []
        self.tb_los = np.zeros( [self.no_masters, self.no_sensors] )
        self.tb_diff = np.zeros( [self.no_masters, self.no_sensors] )
        self.tb_tot = np.zeros( [self.no_masters, self.no_sensors] )
        
        for i in range(self.no_masters):
            for j in range(self.no_sensors):
            
                c = cycle(IWU = self.IWU[j],
                          tWU = self.tWU[j],
                          IRO = self.IRO[j],
                          tRO = self.tRO[j],
                          IRX = self.IRX[j],
                          Ldatau = self.bits_master[i],
                          Ldatad = self.bits_sensor[j],
                          Rbu = self.Rb_sensor[j],
                          Rbd = self.Rb_master[i],
                          Tcycle = self.Tcycle[j],
                          ISL = self.ISL[j],
                          QmAh = self.QmAh[j],
                          ID_max = self.Imax_s,
                          ID_min = self.Imin_s
                          )
                
                self.tb_los[i, j] = c.battery_life_ID(ID = self.ID_rq_s_los[i, j] )
                self.tb_diff[i, j] = c.battery_life_ID(ID = self.ID_rq_s_diff[i, j] )
                self.tb_tot[i, j] = c.battery_life_ID(ID = self.ID_rq_s_tot[i, j] )
                
    def calch_n_to_n(self):
        """
        Calculate the LOS channel gains
        """        
        self.h_ms = self.m_els.txrx_pair_from( self.s_els ).h 
        self.h_sm = self.s_els.txrx_pair_from( self.m_els ).h 
        
    def calch_w_to_w(self):
        """
        Calculate the channel gains for the wall elements        
        """        
        self.h_ww = self.w_els.txrx_pair_from( self.w_els ).h 
        
    def calch_a_to_n(self):
        """
        Calculate the LOS channel gains for the ambient source elements        
        """        
        self.h_ma = self.m_els.txrx_pair_from( self.a_els ).h 
        self.h_sa = self.s_els.txrx_pair_from( self.a_els ).h         
        
    def calch_w_to_n(self):
        """
        Calculate the LOS channel gains from the wall to the nodes        
        """        
        self.h_mw = self.m_els.txrx_pair_from( self.w_els ).h 
        self.h_sw = self.s_els.txrx_pair_from( self.w_els ).h         
    
    def calch_n_to_w(self):        
        """
        Calculate the LOS channel gains from the nodes to the walls        
        """                
        self.h_wm = self.w_els.txrx_pair_from( self.m_els ).h 
        self.h_ws = self.w_els.txrx_pair_from( self.s_els ).h        
        
    def calch(self, h_ww = None):
        """
        Calculate all channel gains
        """
        self.calch_n_to_n()
        if h_ww is None:        
            self.calch_w_to_w()
        else:
            self.h_ww = h_ww
            
        self.calch_a_to_n()
        self.calch_w_to_n()
        self.calch_n_to_w()
        
    def calc_n_to_n(self):
        """
        Calculate incident power from the master to the sensor nodes and vice-versa       
        """
        self.Pin_sm = self.h_sm * self.PT_master.reshape([self.no_masters, 1]) 
        self.Pin_ms = self.h_ms * self.PT_sensor.reshape([self.no_sensors, 1])
        
    def calc_n_to_w(self):
        """
        Calculate incident power to the walls from the master and sensor nodes
        """
        self.Pin_wm = self.h_wm * self.PT_master.reshape([self.no_masters, 1])
        self.Pin_ws = self.h_ws * self.PT_sensor.reshape([self.no_sensors, 1])        
    
    def calc_a_to_n(self):
        """
        Calculate total incident power to the nodes from ambient light sources
        """
        PT_a = self.a_els.PT.reshape([self.n_a, 1])
        self.Pin_ma = np.sum(self.h_ma * PT_a, axis = 0)
        self.Pin_sa = np.sum(self.h_sa * PT_a, axis = 0)
    
    def calc_w_to_w(self):
        """
        Calculate power exchange between the wall elements
        """
        PT_w = self.w_els.PT.reshape([self.n_w, 1])
        self.Pin_ww = np.sum(self.h_ww * PT_w, axis = 0)
    
    def calc_w_to_m(self):
        """
        Calculate the LOS channel gains from the wall to the master nodes        
        """        
        PT_w = self.w_els.PT.reshape([self.n_w, 1])        
        self.Pin_mw = np.sum(self.h_mw * PT_w, axis = 0)

    def calc_w_to_s(self):
        """
        Calculate the LOS channel gains from the wall to the sensor nodes        
        """        
        PT_w = self.w_els.PT.reshape([self.n_w, 1])        
        self.Pin_sw = np.sum(self.h_sw * PT_w, axis = 0)           
    
    def calc_noise(self):
        self.n_m = np.zeros( [self.no_masters, self.no_sensors] )
        self.n_s = np.zeros( [self.no_masters, self.no_sensors] )
        
        for i in range( self.no_masters ):
            for j in range(self.no_sensors):
                self.n_m[i , j] = self.tia_m.calc_noise_power( self.B_sensor[j] )
                self.n_m[i , j] += 2 * constants.qe * self.B_sensor[j] * self.i_ma[i]
                self.n_s[i , j] = self.tia_s.calc_noise_power( self.B_master[i] )
                self.n_s[i , j] += 2 * constants.qe * self.B_master[i] * self.i_sa[j]
        
        self.snr_m = np.zeros( [self.no_masters, self.no_sensors] )
        self.snr_s = np.zeros( [self.no_masters, self.no_sensors] )
        
 
        self.g_m_los = self.i_ms_los / 2 / np.sqrt(self.n_m)
        self.g_s_los = self.i_sm_los / 2 / np.sqrt(self.n_s)
        self.g_m_tot = self.i_ms_tot / 2 / np.sqrt(self.n_m)
        self.g_s_tot = self.i_sm_tot / 2 / np.sqrt(self.n_s)
        self.g_m_diff = self.i_ms_diff / 2 / np.sqrt(self.n_m)
        self.g_s_diff = self.i_sm_diff / 2 / np.sqrt(self.n_s)
        
        self.snr_m_los = self.g_m_los ** 2.0
        self.snr_s_los = self.g_s_los ** 2.0
        self.snr_m_tot = self.g_m_tot ** 2.0
        self.snr_s_tot = self.g_s_tot ** 2.0
        self.snr_m_diff = self.g_m_diff ** 2.0
        self.snr_s_diff = self.g_s_diff ** 2.0
        
        self.snr_m_los_dB = 10 * np.log10(self.snr_m_los)
        self.snr_s_los_dB = 10 * np.log10(self.snr_s_los)
        self.snr_m_tot_dB = 10 * np.log10(self.snr_m_tot)
        self.snr_s_tot_dB = 10 * np.log10(self.snr_s_tot)
        self.snr_m_diff_dB = 10 * np.log10(self.snr_m_diff)
        self.snr_s_diff_dB = 10 * np.log10(self.snr_s_diff)
        
    def light_sim(self, output = False):
        """
        Simulate the uplink channel
        """
        self.Pin_ms_diff = np.zeros([self.no_masters, self.no_sensors, self.no_bounces])
        self.Pin_sm_diff = np.zeros([self.no_masters, self.no_sensors, self.no_bounces])
        if output:
            print('Beginning sensor net simulation.')
            print('Calculating node-to-node LOS power')
        self.calc_n_to_n()
        self.calc_a_to_n()
        if output:        
            print('Calculating power incident to walls from sensor and master elements')
        self.calc_n_to_w()        
        for j in range(self.no_sensors):
            if output:        
                print('Estimating power from sensor node %d/%d' %(j + 1, self.no_sensors ) )
            self.w_els.set_PT_from_Pin( self.Pin_ws[ j , : ] )
            for i in range(self.no_bounces):
                self.calc_w_to_m()
                self.Pin_ms_diff[:, j, i] = self.Pin_mw
                if output:
                    print('Starting diffuse power simulation for bounce %d/%d' %(i + 1, self.no_bounces ) )
                self.calc_w_to_w()
                self.w_els.set_PT_from_Pin( self.Pin_ww )
        
        for j in range(self.no_masters):
            if output:
                print('Estimating power from master node %d/%d' %(j + 1, self.no_masters ) )
            self.w_els.set_PT_from_Pin( self.Pin_wm[ j , : ] )
            for i in range(self.no_bounces):
                self.calc_w_to_s()
                self.Pin_sm_diff[j, :, i] = self.Pin_sw
                if output:
                    print('Starting diffuse power simulation for bounce %d/%d' %(i + 1, self.no_bounces ) )
                self.calc_w_to_w()
                self.w_els.set_PT_from_Pin( self.Pin_ww )
        
        self.Pin_ms_diff_tot = np.sum(self.Pin_ms_diff, axis = 2)
        self.Pin_sm_diff_tot = np.sum(self.Pin_sm_diff, axis = 2)
        self.Pin_ms_tot = self.Pin_ms_diff_tot + np.transpose(self.Pin_ms)
        self.Pin_sm_tot = self.Pin_sm_diff_tot + self.Pin_sm
        
        # Calculate signal and ambient light currents
        self.i_ms_los = self.cel_ms * np.transpose(self.Pin_ms)
        self.i_sm_los = self.cel_sm * self.Pin_sm        
        self.i_ma = self.cel_ma * self.Pin_ma
        self.i_sa = self.cel_sa * self.Pin_sa
        self.i_ms_diff = self.cel_ms * self.Pin_ms_diff_tot
        self.i_sm_diff = self.cel_sm * self.Pin_sm_diff_tot
        self.i_ms_tot = self.cel_ms * self.Pin_ms_tot
        self.i_sm_tot = self.cel_sm * self.Pin_sm_tot
        
        self.hi_ms_los = self.i_ms_los / self.PT_sensor
        self.hi_sm_los = self.i_sm_los / self.PT_master

        self.hi_ms_diff = self.i_ms_diff / self.PT_sensor
        self.hi_sm_diff = self.i_sm_diff / self.PT_master
        
        self.hi_ms_tot = self.i_ms_tot / self.PT_sensor
        self.hi_sm_tot = self.i_sm_tot / self.PT_master
        
    def calc_rq(self):
        self.g0 = ut.Qinv( self.BER_target )
        self.PT_rq_s_los = 2 * self.g0 * np.sqrt( self.n_m ) / self.hi_ms_los
        self.PT_rq_m_los = 2 * self.g0 * np.sqrt( self.n_s ) / self.hi_sm_los
        self.PT_rq_s_diff = 2 * self.g0 * np.sqrt( self.n_m ) / self.hi_ms_diff
        self.PT_rq_m_diff = 2 * self.g0 * np.sqrt( self.n_s ) / self.hi_sm_diff
        self.PT_rq_s_tot = 2 * self.g0 * np.sqrt( self.n_m ) / self.hi_ms_tot
        self.PT_rq_m_tot = 2 * self.g0 * np.sqrt( self.n_s ) / self.hi_sm_tot
        
        self.ID_rq_s_los = self.driver_s.calc_I( self.PT_rq_s_los )
        self.ID_rq_m_los = self.driver_s.calc_I( self.PT_rq_m_los )
        self.ID_rq_s_diff = self.driver_s.calc_I( self.PT_rq_s_diff ) 
        self.ID_rq_m_diff = self.driver_s.calc_I( self.PT_rq_m_diff )
        self.ID_rq_s_tot = self.driver_s.calc_I( self.PT_rq_s_tot )
        self.ID_rq_m_tot = self.driver_s.calc_I( self.PT_rq_m_los )
        
    def plot_on( self, 
                 attr = 'Pin_mw', 
                 surf = 'east wall',
                 show_title = True, 
                 show_colorbar = True,
                 figure_no = None,
                 index = None):
        
        """
        Visualize distribution on surface provided by either a wall name or a surface object
        """
        if type(surf) == str:
            surf = self.get_wall_by_name( surf )        
        
        if type(attr) == str:
            attr = getattr(self, attr)
            
        if index is not None:
            attr = attr[ : ,index ]
        
        surf.plot_on_surf(self.w_els.rS,
                          attr,
                          show_title = show_title,
                          show_colorbar = show_colorbar,
                          figure_no = figure_no)       
    
    def save(self, filename):
        with open( filename, 'wb' ) as f:
            pickle.dump( vars(self), f )
