"""
Module that provides PSF models for the microscope.
"""
import jax as jax
import jax.numpy as jnp
import numpy as np

class PSFmodel:
    """
    Base class for all PSF models
    """

    def __init__(self, config):
        self.config = config
        pass

    def _sanity_check(self):
        """
        Method that performs a sanity check of the provided parameters.
        """
        pass

    def getPSF(self, X, pattern,**kwargs):
        return None

class PSFgenerator:

    def __init__(self):
        pass


class standingWave(PSFmodel):

    def __init__(self, config):
        super().__init__(config)
        pass

    def getPSF(self, X, pattern, **kwargs):
        """
        Method to create a standing evanescent wave pattern via TIR to illuminate the sample
        :return: incident intensity at z=0 for M molecules (no decay...)
        """
        lam = kwargs.get('Lambda0', self.config.Lambda0)
        epsilon = kwargs.get('p0', 0) # imperfection of intensity minimum
        #p0 = kwargs.get('p0',self.config.p0)
        X = X[jnp.newaxis,jnp.newaxis,:,:] # shape = (k,n,M,d)
        x0, c0, phi, k = pattern.x0, pattern.c0, pattern.phi, pattern.k
        
        pos_ext = 2*jnp.pi/lam * (X - x0) # shape (K, n, m, d)
        kx = jnp.einsum('knmd,knmd->knm', k , pos_ext) # scalar product of k and x, shape (K,n,M)
        
        arg = kx - phi
        E_field = c0*jax.lax.complex(jnp.cos(arg), jnp.sin(arg)) # shape=(K, n, M)
        inc_int = jnp.abs(jnp.sum(E_field,axis=(1)))**2 # coherent superposition of incident waves, shape=(n_M,n_K)
        
        res_field = epsilon * c0*jax.lax.complex(jnp.cos(arg-np.pi), jnp.sin(arg-np.pi)) #TODO: + or -?
        res_int = jnp.abs(jnp.sum(E_field,axis=(1)))**2
        
        inc_int = inc_int.T
        #import matplotlib.pyplot as plt
        #fig, ax = plt.subplots()
        #ax.plot(np.sum(inc_int,axis=0))
        #plt.show()
        return inc_int

class Background(PSFmodel):

    def __init__(self, config):
        super().__init__(config)
        pass

    def getPSF(self, pattern, **kwargs):
        """
        Method to create a standing evanescent wave pattern via TIR to illuminate the sample
        :return: average background counts
        """
        c0 = pattern.c0
        E_field = c0
        inc_int = (jnp.abs(jnp.sum(E_field,axis=(1)))**2)# incoherent superposition, shape=(n_M,n_K)
        inc_int = inc_int.T
        return np.ones((c0.shape[0],))

class Detection(PSFmodel):

    def __init__(self, config):
        super().__init__(config)
        pass

    def getPSF(self, pattern, **kwargs):
        """
        Method to calculate the detection PSF to weight
        measurement results.
        :param pattern: pattern object
        :return: array of detPSF values, shape=(K, M)
        """
        sigma = kwargs.get('sigma',self.config.detFWHM/self.config.sig_to_FWHM)
        x0 = pattern.x0
        if self.config.detPSF == 'gauss':
            fun = jnp.exp(-(x0/(jnp.sqrt(2)*sigma))**2)# shape (K, n, M, d)
        #val = jnp.einsum('nmkd,nmkd->nmk',fun, fun)
        val = jnp.sum(fun[:,:,:,0]*fun[:,:,:,1],axis=2)
        return val.T
