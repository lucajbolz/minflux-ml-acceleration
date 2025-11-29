"""
Module that provides some base-classes to set-up an experiment.
Main components are the instrument and the sample.
"""
import numpy as np
import copy as copy
import os
import jax as jax
import jax.numpy as jnp

import lib.utilities as ut
from lib.constants import *
from lib.simulation.psf_models import standingWave, Background, Detection

class Setup:
    """
    Class for the Hardware components: Instrument and Sample
    """
    def __init__(self, instrument=None, sample=None):
        if instrument is None:
            self.instrument = Instrument()
        else:
            self.instrument = instrument
        if sample is None:
            self.sample = SampleGenerator().generate_sample()
        else:
            self.sample=sample
        pass

    def get_trace(self,pattern,mode='real'):
        """
        Trigger a response of the setup to a pattern object.
        :return: Trace object with photon counts.
        """
        if mode=='real':
            sig_photons, bgr_photons = self.sample.get_photons(pattern, self.signal_model(), self.bgr_model(), self.instrument.config.dwell_time, mean=False, cumulative=True)
        elif mode=='mean':
            sig_photons, bgr_photons = self.sample.get_photons(pattern, self.signal_model(), self.bgr_model(), self.instrument.config.dwell_time, mean=True, cumulative=True)
        elif mode=='strawman':
            sig_photons, bgr_photons = np.zeros((2,pattern.x0.shape[0]))
        return Trace().construct_trace(sig_photons, bgr_photons)

    def get_model(self):
        """
        Numerical expression for the cumulative poissonian mean lambda
        (all molecules' emission combined, *including* background)
        :return: function f(X,PSFp,**kwargs)
        """
        sig = self.signal_model()
        bgr = self.bgr_model()
        def model(X, pattern, **kwargs):
            signal = np.sum(sig(X, pattern, **kwargs),axis=0)
            background = bgr(pattern, **kwargs)
            return signal + background
        return lambda X, pattern, **kwargs: model(X, pattern, **kwargs)

    def signal_model(self):#TODO: correct application of molecule brightness
        """
        Get the signal of the molecules
        :return: callable of signal model
        """
        
        psf = self.instrument.effPSF()
        I_default = 0
        if self.sample.config.n_molecules>0:
            I_default = self.sample.config.molecule_brightness#TODO * self.instrument.config.dwell_time * self.instrument.config.alpha
        def signal(X, pattern, **kwargs):
            I0 = kwargs.get('molecule_brightness', I_default)
            return I0 * psf(X, pattern,**kwargs)
        return lambda X, pattern, **kwargs: signal(X, pattern, **kwargs)

    def bgr_model(self):
        """
        Get the background model.
        :return: Callable of bgr model
        """
        bgrpsf = self.instrument.effBgr()
        def bgr(pattern, **kwargs):
            beta = kwargs.get('beta', self.sample.config.beta)
            return beta * bgrpsf(pattern,**kwargs)
        return lambda pattern, **kwargs: bgr(pattern, **kwargs)

#------------------------------------------------------------
class Instrument: #TODO: use more general PSF module???

    def __init__(self,param_dict={}): # TODO: don't copy all the values, but take from self.config...
        default_params = {
            # instrument default config
            'dwell_time' : DWELL_TIME 
            ,'alpha' : ALPHA  
            ,'Lambda0' : LAMBDA 
            ,'STED_fac' : STED_FACTOR 
            ,'NA' : NUMERICAL_APERTURE 
            ,'airy' : AIRY_FACTOR 
            ,'sig_to_FWHM' : SIG_TO_FWHM 
            ,'detFWHM' : DETECTION_FWHM 
            ,'dimensions' : DIMENSIONS
            ,'excPSF' : EXC_PSF
            ,'superposition' : SUPERPOSITION_TYPE 
            ,'detPSF' : DET_PSF # shape of detection PSF

        }
        self.config = ut.Configuration(default_params,param_dict)

    def emissionPSF(self, X, pattern, **kwargs):
        sw = standingWave(self.config)
        return sw.getPSF(X, pattern, **kwargs)

    def backgroundPSF(self, pattern, **kwargs):
        bgr = Background(self.config)
        return bgr.getPSF(pattern, **kwargs)

    def detectionPSF(self, pattern, **kwargs):
        det = Detection(self.config)
        return det.getPSF(pattern, **kwargs)

    def effPSF(self):
        """
        Convolve emission and detection.
        """
        return lambda x, pattern, **kwargs: self.emissionPSF(x,pattern,**kwargs) * self.detectionPSF(pattern,**kwargs)

    def effBgr(self):
        """
        Convolve background and detection.
        """
        return lambda pattern, **kwargs: self.backgroundPSF(pattern,**kwargs) * self.detectionPSF(pattern, **kwargs)[0]#_bgrPSF(pattern,**kwargs)


class SampleGenerator:

    def __init__(self, param_dict={}) -> None:
        # sample default config
        default_params = {
            'dimensions' : DIMENSIONS
            ,'n_molecules' : NUMBER_OF_MOLECULES
            ,'mol_pos' : None
            ,'molecule_brightness' : MOLECULE_BRIGHTNESS
            ,'beta' : BACKGROUND_RATE # number of background molecules
            ,'single_molecule_photon_budget' : SINGLE_MOLECULE_PHOTON_BUDGET
            ,'photon_budget': PHOTON_BUDGET #budget for each molecule, list of M numbers
            ,'blinking': BLINKING
            ,'bleach_idx': BLEACH_IDX
        }
        self.config = ut.Configuration(default_params, param_dict)        
        
    def generate_sample(self):
        #if self.config.mol_pos is None:
        #    self.config.mol_pos = self.get_rand_positions() # create random positions for molecules if no position is provided

        if self.config.photon_budget is None: # define statistically distributed photon budget
            rng = np.random.default_rng()
            self.config.photon_budget = np.zeros((self.config.n_molecules))
            for mo in range(0,self.config.n_molecules):
                self.config.photon_budget[mo] = np.int_(rng.normal(self.config.single_molecule_photon_budget, 0.5*self.config.single_molecule_photon_budget,1)) # normal distributed photon budget, std=15% of mean
                #photon_budget[mo] = self.single_molecule_photon_budget # photon budget is constant   

        sample = Sample(self.config)
        return sample

    def get_rand_positions(self, int_range=20):
        Mol_Pos = np.zeros((self.config.n_molecules,self.config.dimensions))
        rng = np.random.default_rng()
        for mo in range(0,self.config.n_molecules):
            for d in range(0,self.config.dimensions):
                #Mol_Pos[mo,d] = int_range * (rng.random() - .5)
                Mol_Pos[mo,d] = int_range * rng.random() # position molecules in [0,1]*int_range
        return Mol_Pos  

class Sample:

    def __init__(self, config):
        self.config = config
        self.photon_budget = np.copy(config.photon_budget) # photon budget of all molecules...
        self.photon_counter = np.zeros((config.n_molecules)) # internal counter to keep track of the emission/molecule
        
        self.bleach_idx = np.copy(config.bleach_idx)#np.zeros((config.n_molecules)) # k-th measurement tuple where the molecule bleached.
        #self.bleach_idx[:] = np.nan
        self.it_idx = 0 # to keep track of the current iteration idx
        rng = np.random.default_rng()
        self.s0 = int(2000*rng.random()) # get part of the seed randomized
        pass

    def clone(self):
        return Sample(self.config)

    def get_photons(self, pattern, signal, bgr, dwell_time, mean=False, cumulative=True):
        """
        Method in order to retrieve the photons from the sample.
        :param pattern: pattern object
        :param signal: eff PSF callable
        :param bgr: effBgr callable
        :param mean: option to disable poisson noise (True)
        :param cumulative: option to receive a cumulated signal of all emitters
        """
        sig_photons = self._get_sig_photons(pattern, signal, dwell_time, mean=mean)
        if cumulative==True:
            sig_photons = np.sum(sig_photons,axis=0)
        bgr_photons = self._get_bgr_photons(pattern, bgr, mean=mean)
        self.it_idx += 1
        return sig_photons, bgr_photons

    def _get_sig_photons(self,pattern,model,dwell_time,mean=False,blinking=False):
        """
        Method to retrieve signal from molecules.
        :return: array of photons, shape=(#number of molecules, #number of measurement sites)
        """
        K,_,_,_ = pattern.x0.shape

        phot_vec = np.zeros((self.config.n_molecules,int(K)))
        
        seed = self.s0 + self.it_idx #1145, if all rulers should share the same set of keys.
        key = jax.random.PRNGKey(seed)
        # first: get all photons
        if not self.is_bleached().all(): # if not all molecules bleached: continue
            if mean == False:
                #rng = np.random.default_rng()
                #self.X += 50*(rng.random((self.config.dimensions))-0.5) # add wobbling to nanoruler
                phot_vec = jax.random.poisson(key,model(self.config.mol_pos,pattern)) #obtain a photon number for each measurement site from the respective poisson distribution
            else:
                phot_vec = model(self.config.mol_pos,pattern) # for debugging purposes (no noise)
            
            phot_vec = self.add_blinking(phot_vec, dwell_time) # add variations in intensity to trace
            # second: set to zero according to budget
            local_bleach_idx = jnp.cumsum(phot_vec,axis=1) + self.photon_counter[:,jnp.newaxis] > self.photon_budget[:,jnp.newaxis] # find bleach point
            phot_vec = phot_vec.at[local_bleach_idx==True].set(0)
            #phot_vec[local_bleach_idx==True] *= 0
            
            # third update photon_counter and bleach_idx
            for i in range(self.config.n_molecules):
                if np.isnan(self.bleach_idx[i]) and local_bleach_idx[i].any():
                    self.bleach_idx[i] = int(K) + np.where(local_bleach_idx[i]==True)[0][0]
            self.photon_counter += np.sum(phot_vec,axis=1) # update photon counter: add all photons of one molecule for consecutive measurements
        return phot_vec

    def _get_bgr_photons(self, pattern, model, mean=False):
        # draw background photons for each measurement.
        # Those do not influence the bleaching/molecule budget
        rng = np.random.default_rng()
        mean_val = model(pattern)
        if mean==False: 
            bgr_phot = rng.poisson(mean_val)
        else:
            bgr_phot = mean_val
        return  bgr_phot

    def is_bleached(self):
        if np.any(np.isnan(self.bleach_idx)):
            return np.greater_equal(self.photon_counter,self.photon_budget)
        else:
            bleach_arr = np.array(self.bleach_idx) <= self.it_idx
            self.photon_budget[bleach_arr==True] = self.photon_counter[bleach_arr==True]
            return bleach_arr
    
    def add_blinking(self,phot_vec,dwell_time):
        if self.config.blinking:
            rng = np.random.default_rng()
            #f_blink = 2 # blinking frequency [Hz] #TODO: set in config
            #avg_n_blink = dwell_time * f_blink
            n_blink = int(((4 + 3*(rng.random()-0.5))))
            rnd_fluc = (1 + 0.2*(rng.random(n_blink)-0.5))
            sp = jnp.array_split(phot_vec,n_blink,axis=1)
            new_vec = jnp.empty((2,0))
            for idx, event in enumerate(rnd_fluc):
                new_vec = jnp.concatenate((new_vec,event*sp[idx]),axis=1)
            phot_vec = new_vec.astype(int)
        return phot_vec

class Trace:
    """
    Class to represent trace data, e.g. photon counts, bleaching indices etc.
    """

    def __init__(self) -> None:
        """
        Initialization
        :param config: configuration object
        """
        self.photons = np.empty((0)) # full emission (bgr and signal)
        self.sig_photons = np.empty((0)) # signal photons
        self.bgr_photons = np.empty((0)) # background photons
        #self.bleach_idx = np.empty((0)) # iteration in which the molecule bleached (be aware of block_size in config, since the molecule has been scanned block_size times in one iteration)

    def construct_trace(self, sig_photons, bgr_photons):
        """
        Method in order to retrieve the photons from the sample.
        :param sig_photons: signal photons, shape= (K)
        :param bgr_photons: shapem = (K)
        :return: Trace Object
        """
        self.sig_photons = np.asarray(sig_photons)
        self.bgr_photons = np.asarray(bgr_photons)
        self.photons = self.sig_photons + self.bgr_photons
        return self
