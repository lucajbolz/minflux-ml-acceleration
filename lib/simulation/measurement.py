"""
Module that hosts the measurement capabilities/modes of an experiment.
Its main purpose is to create a tuple of points/parameters at which photons are retrieved.
"""

import numpy as np
import jax.numpy as jnp
import lib.utilities as ut
from lib.constants import *

class Measurement:
    """
    Class for the Software components: Mainly PatternGenerator and Estimator
    """
    def __init__(self, param_dict={}):
        # PatternGenerator
        # Estimator
        default_params = {
            # measurement/protocol config
            'meas_pos' : None # parameter to set custom measurement positions
            ,'block_size' : BLOCK_SIZE # parameter for phase scans: no of consecutive scans in one axis/ number of scans at each iteration stage
            ,'phi0_pos' : None # parameter to set custom TEM angles
            ,'meas_mode' : MEASUREMENT_MODE
            ,'repetitions':REPETITIONS
            ,'iteration_depth' : ITERATION_DEPTH
            ,'shrink_factor' : SHRINK_FACTOR # shrink characteristic length of pattern
            ,'light_factor' : LIGHT_FACTOR # alter brightness of intensity profile with each iteration
            ,'num_positions' : PIXEL_NUMBER # number of measurement points in one scan pattern
            , 'L': SCAN_RANGE # sth like scan_range, Minflux L
            ,'phi0': 0. #orientation of standing wave: angle to x-axis (second wave: phi0+pi)
            ,'phase_offset' : [0.,0.]
            ,'x0' : np.zeros((2))
        }
        self.config = ut.Configuration(default_params,param_dict)
        pass

    def get_pattern(self, instrument):
        """
        Retrieve a new pattern (for current iteration stage)
        """
        self.config.set_params(instrument.config.__dict__)
        generator = PatternGenerator(self.config).get_generator()
        pattern = generator.get_pattern() # returns new pattern object
        return pattern


class PatternGenerator:

    def __init__(self, config):
        if config.meas_mode == 'phase':
            self.pattern_generator = PhasePatternGenerator(config)
        elif config.meas_mode=='minflux':
            self.pattern_generator = MinfluxPatternGenerator(config)
        else:
            raise Exception('please specify config.meas_mode!')
        
    def get_generator(self):    
        return self.pattern_generator

class PhasePatternGenerator:
    """
    Stay at a single spatial position and perform a phase scan.
    No input record is needed to get a pattern...
    """

    def __init__(self, config) -> None:
        self.config = config
        self.lambda0 = np.zeros((config.dimensions))
        self.lambda0.fill(config.Lambda0)
        self.c0 = np.ones((config.dimensions))
        #self.c0 = np.array([1.,.1])
        #self.c0.fill(config.alpha * config.dwell_time)
        self.phi0 = config.phi0
        self.x0 = config.x0
        self.phase_offset = config.phase_offset

    def get_pattern(self):
        pattern = Pattern() # empty pattern object
        L_vec = np.repeat(self.config.L,self.config.dimensions,axis=0) # create quadratic grid with N**2 measurement points
        num_vec = np.repeat(self.config.num_positions,self.config.dimensions,axis=0)
        x0 = self.x0
        p = PointLinearPhasePattern(self.config.Lambda0/2, self.config.dimensions, L_vec, num_vec, self.config.block_size)
        
        pattern.x0 = p.get_positions(self.x0)
        #pattern.lam = np.repeat([self.lambda0],len(pattern.x0),axis=0)
        pattern.c0 = p.get_c0s(self.c0)
        pattern.phi = 2*np.pi/(self.config.Lambda0/2) * p.get_phases(self.phase_offset)
        pattern.k = p.get_kvec(self.phi0)
        
        
        #pattern.x0 = p.get_positions()
        #pattern.lam = np.repeat([self.lambda0],len(pattern.x0),axis=0)
        #pattern.c0 = p.get_c0s()
        #pattern.phi = 2*np.pi/(self.config.Lambda0/2) * p.get_phases()
        #pattern.phi0, pattern.theta = p.get_angles(self.phi0) # orientation and incident angles
        return pattern

class PointLinearPhasePattern:
    """
    Pattern that allows a linear phase scan at a certain fixed
    spatial coordinate.
    Phase scan will always be centered around 0 and range from -L/2 to +L/2.
    One gets block_size lines in two perpendicular directions, specified via phi0 
    as offset angle to x-axis.
    """

    def __init__(self, lamb, dim, L, N, block_size) -> None:
        """
        :param lamb: wavelength of standing wave
        :param dim: number of dimensions
        :param x0: point at which to perform that phase scan (fixed!)
        :param L: length of phase scan (nm)
        :param N: Number of steps in each dimension
        """
        self.lamb = lamb
        self.dim = dim
        # x0, c0, L, N should have dim entries
        self.L = L
        self.N = N
        self.block_size = block_size
        return None

    def get_positions(self, x0):
        """
        Center position of PSF (fixed at one spatial position for phase scan)
        :return: x0 with shape K,n,M,d
        """
        g = []
        for i in range(self.dim):
            N = int(self.N[i])
            xi = self.block_size * np.repeat([x0],N,axis=0).tolist()
            #xi = np.repeat(xi,,axis=0)
            if i==0:
                g=xi
            else:
                g = np.concatenate((g,xi),axis=0)
        points = g
        # umsortieren so that product(N) einzelpositionen von dim-dimensional vektoren
        return points[:,jnp.newaxis,jnp.newaxis,:]

    def get_phases(self,phase_offset):
        """
        make the linear phase scan for each dimension.
        Alter only phase of one vec for phase scan and set the other to zero+max to measure with minimum
        :return: phases in *nm*, shape (K, n, M)
        """
        g = []
        for i in range(self.dim):
            N = int(self.N[i])
            phi_0 = phase_offset[i]
            min, max, N =  phi_0 - self.L[i]/2, phi_0 + self.L[i]/2, int(self.N[i])
            phi_i = self.block_size * ((max-min)*np.arange(N)/N + min).tolist()
            phi_i = np.asarray([phi_i])
            phi_i = np.concatenate([phi_i,0*phi_i+self.lamb/2],axis=0)
            if i==0:
                g=phi_i
            else:
                g = np.concatenate((g,phi_i),axis=1)
        phases = g.T
        return phases[:,:,jnp.newaxis]

    def get_c0s(self,c0):
        """
        amplitude of electric field of each incident k-vector
        :return: c0, shape = (K, n, M)
        """
        #TODO: enable two different c0s, i.e. different quality of minimum in each axis.
        c0s = np.empty((self.block_size * np.sum(self.N),)+(c0.size,))
        c0s[:]=c0
        #for i in range(0, self.block_size * np.sum(self.N)):
        #    c0 = c0.reshape((-1,2))#*rng.random()
        #    c0s = np.concatenate((c0s, c0),axis=0)
        """for i in range(self.dim):
            N = int(self.N[i])
            initial = self.block_size * np.zeros((N)).tolist() # repeat list block_size times
            ci = self.block_size * np.repeat(self.c0[i],N).tolist()
            for idx in range(0,i): # add zeros for other directions
                ci = initial + ci
            for idx in range(i+1,self.dim):
                ci = ci + initial
            ci = np.asarray([ci])#.reshape((2,5)).tolist()
            #ci = np.repeat(ci,self.block_size,axis=0)
            if i==0:
                g=ci
            else:
                g = np.concatenate((g,ci),axis=0)
        c0s = g.T"""
        # umsortieren so that sum(self.N) positions von dim-dimensional vektoren
        return c0s[:,:,jnp.newaxis]

    def get_kvec(self,phi0):
        """
        Retrieve k vectors along pattern as array. theta is opening angle, phi0 is orientation angle (to x-axis).
        Two angles for the two incoming waves-> they have an offset of pi (head-on collision along orientation axis).
        theta is always chosen to be 45 degree, i.e. normal beams
        :param phi0: offset orientation to x-axis
        """
        theta_angles, phi0_angles = np.empty((0,2)), np.empty((0,2))#self.dim
        phi_list, theta_list = [],[]
        for idx, n in enumerate(self.N):
            phi = np.repeat(np.array([[phi0 + idx*np.pi/2,phi0+(idx+2)*np.pi/2]]),n*self.block_size,axis=0)#*rng.random()
            theta = np.pi/2 * np.repeat(np.array([[1.,1.]]),n*self.block_size,axis=0)
            #phi_list.append(phi)
            #theta_list.append(theta)
            phi0_angles = np.concatenate((phi0_angles,phi),axis=0)
            theta_angles = np.concatenate((theta_angles,theta),axis=0)
        phi0, theta = phi0_angles, theta_angles
        
        #phi0 = np.concatenate(phi_list,axis=0) #shape=(K,d)
        #theta = np.concatenate(theta_list,axis=0)

        arr1 = jnp.sin(theta) * jnp.cos(phi0)
        arr2 = jnp.sin(theta) * jnp.sin(phi0)

        #arr = jnp.sin(theta) * jnp.sin(np.array([phi0+np.pi/2,phi0])) # cos, sin shape=(n,K,d)
        #arr = jnp.transpose(arr,axes=(1,0,2)) # shape=(K,n,d)
        #k_arr = arr[:,:,jnp.newaxis,:]
        #arr3 = jnp.sin(theta) * jnp.sin(phi0-np.pi/2)
        #arr = jnp.array([arr1, arr2])
        arr1, arr2 = arr1[:,:,jnp.newaxis,jnp.newaxis], arr2[:,:,jnp.newaxis,jnp.newaxis] # shape (K,n,#M,#d)
        k_arr = jnp.concatenate([arr1, arr2],axis=-1) # shape (K,n,#M,d)
        return k_arr

    def get_angles(self, phi0):
        """
        Retrieve PSF-angles along pattern as array. 2 theta is opening angle, phi0 is orientation angle.
        Two angles for the two incoming waves-> they have an offset of pi.
        theta is always chosen to be 45 degree, i.e. normal beams
        :param phi0: offset orientation to x-axis
        """
        theta_angles, phi0_angles = np.empty((0,2)), np.empty((0,2))#self.dim
        rng = np.random.default_rng()
        for idx, n in enumerate(self.N):
            phi = np.repeat(np.array([[phi0 + idx*np.pi/2,phi0+(idx+2)*np.pi/2]]),n*self.block_size,axis=0)#*rng.random()
            theta = np.pi/2 * np.repeat(np.array([[1.,1.]]),n*self.block_size,axis=0)
            phi0_angles = np.concatenate((phi0_angles,phi),axis=0)
            theta_angles = np.concatenate((theta_angles,theta),axis=0)
        return phi0_angles, theta_angles
    
class MinfluxPatternGenerator:
    """
    Perform a Minflux sequence at specified position.
    Requires Record to estimate new position of pattern (phase offset)!
    """

    def __init__(self, config) -> None:
        self.config = config
        self.lambda0 = np.zeros((config.dimensions))
        self.lambda0.fill(config.Lambda0)
        self.c0 = np.ones((config.dimensions))
        #self.c0 = np.array([1.,.1])
        #self.c0.fill(config.alpha * config.dwell_time)
        self.phi0 = config.phi0
        self.x0 = config.x0
        self.phase_offset = config.phase_offset

    def get_pattern(self):
        pattern = Pattern() # empty pattern object
        L_vec = np.repeat(self.config.L,self.config.dimensions,axis=0) # create quadratic grid with N**2 measurement points
        num_vec = np.repeat(self.config.num_positions,self.config.dimensions,axis=0)
        x0 = self.x0
        p = MinfluxPattern(self.config.Lambda0/2, self.config.dimensions, L_vec, num_vec, self.config.block_size)
        
        pattern.x0 = p.get_positions(self.x0)
        pattern.c0 = p.get_c0s(self.c0)
        pattern.phi = 2*np.pi/(self.config.Lambda0/2) * p.get_phases(self.phase_offset)
        pattern.k = p.get_kvec(self.phi0)
        return pattern

class MinfluxPattern(PointLinearPhasePattern):
    """Class that overloads get_phases of PointLinearPhase pattern to include the endpoint
    of an interval centered around phase_offset.
    """

    def __init__(self,lamb, dim, L, N, block_size):
        PointLinearPhasePattern.__init__(self, lamb, dim, L, N, block_size)
        pass

    def get_phases(self,phase_offset):
        """
        make the linear phase scan for each dimension.
        Alter only phase of one vec for phase scan and set the other to zero+max to measure with minimum
        :return: phases in *nm*, shape (K, n, M)
        """
        g = []
        for i in range(self.dim):
            N = int(self.N[i])
            phi_0 = phase_offset[i]
            min, max, N =  phi_0 - self.L[i]/2, phi_0 + self.L[i]/2, int(self.N[i])
            phi_i = self.block_size * np.linspace(min,max,N,endpoint=True).tolist()
            phi_i = np.asarray([phi_i])
            phi_i = np.concatenate([phi_i,0*phi_i+self.lamb/2],axis=0)
            if i==0:
                g=phi_i
            else:
                g = np.concatenate((g,phi_i),axis=1)
        phases = g.T
        return phases[:,:,jnp.newaxis]

class Pattern:
    """
    Class that represents Pattern objects, i.e. information about the
    measurement sites and instrument's configuration at those sites.
    :param config: configuration object from which to initialize the 
                    pattern object.
    :return: None
    """
    def __init__(self) -> None:
        """
        every entry has shape K,n,M,d
        with K #measurements, n #incident wave vectors, M #molecules, d #dimensions.
        x0, lam and c0 have the shape (dim, K)
        """
        #self.lam = np.empty((0,0))
        self.x0 = np.empty((0,0,0,0))
        self.c0 = np.empty((0,0,0))
        self.phi = np.empty((0,0,0)) # phases of the pattern
        #self.phi0 = np.empty((0,0)) # 2 orientation angle for 2 incident beams...
        #self.theta = np.empty((0,0)) # opening 2 angle for 2 incident beams...
        self.k = np.empty((0,0,0,0))

    def append(self, obj):
        """
        Method to append new entries to the measurement record.
        :param obj: Object that contains keys/vals of MeasurementRecord.
        :return: pass
        """
        param_dict = obj.__dict__
        eigen_dict = self.__dict__
        for key in param_dict:
            if key in eigen_dict.keys() and key!='config':
                if eigen_dict[key].size==0:
                    val = param_dict[key]
                else:
                    val = np.concatenate((eigen_dict[key],param_dict[key]))
                #self[key] = self[key] + param_dict[key]
                setattr(self, key, val)
        pass