"""
Set global variables in this module.

This module contains global variables that are used throughout the program. These variables define various constants related to the sample, instrument, and measurement.

Author: Thomas Arne Hensel
Date: 03/2023
"""
import numpy as np

# sample constants
BLEACH_IDX = np.array([np.nan,np.nan])  # bleach idx of molecules
PHOTON_BUDGET = [3*10**5, 5*10**5]  # photon budgets for the sample
BACKGROUND_RATE = 1E-9  # average number of background molecules
NUMBER_OF_MOLECULES = 2  # number of molecules in the sample
MOLECULE_BRIGHTNESS = 1.0  # relative molecule brightness
BLINKING = False  # boolean to switch blinking on and off
SINGLE_MOLECULE_PHOTON_BUDGET = 3*10**5  # single molecule photon budget if no individual budgets are provided

# instrument constants
LAMBDA = 640  # wavelength of incident light (nm)
EXC_PSF = 'sine'  # type of excitation PSF
ALPHA = 3.5*10**2  # excitation rate (phot/(molecule s))
AIRY_FACTOR = 1.22  # factor of airy disk
NUMERICAL_APERTURE = 1.4  # numerical aperture of objective
DETECTION_FWHM = 350  # FWHM of detection PSF in nm (gaussian, 70um Pinhole, 200x magnification)
DIMENSIONS = 2  # number of dimensions of the sample
SUPERPOSITION_TYPE = 'incoherent'  # type of superposition of incident light of different dimensions
DET_PSF = 'gauss'  # shape of detection PSF
STED_FACTOR = 10  # factor to saturation: alphaSTED/alphaSaturation
SIG_TO_FWHM = 2.35482  # factor to convert sigma to FWHM for gaussian

# measurement constants
REPETITIONS = 1  # number of repetitions of a block, i.e. an atomic measurement cycle
ITERATION_DEPTH = 3  # number of iterations of measurement cycles
DWELL_TIME = 100 * 10**-6  # dwell time for one measurement in seconds
SCAN_RANGE = LAMBDA/2
BLOCK_SIZE = 1 # number of lines per block, e.g. 10 x-scans, 10 y-scans etc
PIXEL_NUMBER = 160
MEASUREMENT_MODE = 'phase'
SHRINK_FACTOR = 0.8 # shrink characteristic length of pattern in each iteration
LIGHT_FACTOR = 1.05 # alter brightness of intensity profile with each iteration