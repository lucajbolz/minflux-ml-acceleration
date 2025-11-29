"""
Main module for Multiflux simulation.
Takes arbitrary number of molecules. 
Contains different classes in order to simulate a realistic experiment.

Construct Experiment, run experiment and perform an analysis (Study)
"""
import os
script_name = os.path.basename(__file__)
import numpy as np
import jax as jax
import copy as copy

# own modules
from lib.constants import *
import lib.utilities as ut
from lib.simulation.setup import Setup, Trace, SampleGenerator, Instrument
from lib.simulation.measurement import Measurement, Pattern
from lib.data_handling.utilities import Converter

class Experiment:
    """
    Class to simulate an experiment.
    It can either perform an experiment from a Setup or
    create an experiment from a Record (e.g. experimental data)
    """

    def __init__(self,**kwargs)->None:
        """
        Initializes an instance of the Experiment class.

        :param kwargs: Optional keyword arguments.
        :type kwargs: dict

        :returns: None
        :rtype: None
        """
        sample = SampleGenerator().generate_sample()
        instrument = Instrument()
        default_setup = Setup(instrument=instrument,sample=sample)
        default_measurement = Measurement()

        self.setup=kwargs.get('setup',default_setup)
        self.measurement = kwargs.get('measurement', default_measurement)
        self.protocol = kwargs.get('protocol',Protocol())
        self.record = kwargs.get('record',Record())
        
        self.log = Log()
        self.log.register_entity()
        pass    

    def perform(self, mode='real'):
        """
        Method that performs an experiment. Do a Measurement with Setup according to Protocol.
        For each prompt in the protocol, apply settings to instrument and measurement. Then 
        generate a pattern and retrieve photons. Lastly, combine them to the measurement record
        and go on to the next prompt.

        :param mode: The mode of the experiment.
        :type mode: str

        :returns: None
        :rtype: None
        """
        #TODO write to log: notify log once a configuration has changed!
        for i,prompt in enumerate(self.protocol.prompts.values()): # prepare microscope for current prompt:
            do_measurement = self.interpret_prompt(prompt=prompt)

            if do_measurement:
                pattern = self.measurement.get_pattern(self.setup.instrument)
                self.record.append(pattern)
                trace = self.setup.get_trace(pattern,mode=mode)
                self.record.append(trace) 
        pass

    def interpret_prompt(self,prompt={}):
        """
        Method to interpret the measurement protocol.
        If the prompt is 'estimate', the machine updates certain parameters,
        e.g. center position and scan range for Minflux sequence.
        Otherwise, it tries to update the setup and measurement mode according to the prompt.
        """
        if 'set' in prompt.keys():
            new_set = prompt['set']
            try:
                self.setup.instrument.config.set_params(new_set['instrument'])
                param_dict = self.setup.instrument.config.__dict__
                self.setup.instrument = Instrument(param_dict=param_dict)
            except:
                pass
            try:
                self.measurement.config.set_params(new_set['measurement'])
                param_dict = self.measurement.config.__dict__
                self.measurement = Measurement(param_dict=param_dict)
            except:
                pass
            return True
        elif 'estimate' in prompt.keys():
            # obtain a simple estimate of the minimum to adjust the phase offset.
            # works currently only for three-point MF TODO: specify type of estimate to be performed!
            def get_new_phase_offset(exp,partitioned_record):
                triples = [axis[-1] for axis in partitioned_record.photons]
                n_ds = [triple[0]+triple[2]-2*triple[1] for triple in triples] #n_d = (n1+n2-2*n0), n1,n0,n2 = exp.record.photons
                if np.any([n_d==0 for n_d in n_ds]):
                    local_x = np.zeros(2,)
                else:
                    local_x = exp.measurement.config.L/4 * np.asarray([(triple[2]-triple[0])/n_d for triple,n_d in zip(triples,n_ds)]) #(n2-n1)/n_d
                #local_phi = (4*np.pi)/exp.setup.instrument.config.Lambda0 * local_x

                new_x = [offset - np.sign(loc)*min(np.abs(loc),exp.measurement.config.L/6) for offset,loc in zip(exp.measurement.config.phase_offset,local_x)]
                new_mean_x = [(min(exp.setup.sample.it_idx,10-1)*offset+x)/min(exp.setup.sample.it_idx+1,10) for x,offset in zip(new_x,exp.measurement.config.phase_offset)]
                return np.asarray(new_mean_x) #+ random.uniform(-1.,1.) * (4*np.pi)/exp.setup.instrument.config.Lambda0 # add some jitter to position estimate to make it more realistic
            try:
                record = Converter().reshape_record(self,export=False,show=False)
                record = Converter().partition_record(record, self.measurement.config.block_size)
                new_offset = get_new_phase_offset(self,record)
                self.measurement.config.phase_offset = new_offset
            except:
                pass # TODO: write to log!
            # do estimate and update measurement parameters
            return False
        pass


    def save(self, filename='experiment',folder=None):
        """
        Save an experiment to a yaml file
        """
        ut.Exporter().write_yaml(self,filename=filename,out_dir=folder)
        pass

    def load(self,path):
        """
        Load an experiment from a yaml.
        :param path: path to yaml file
        :return: experiment object
        """
        experiment = ut.Importer().load_yaml(path)
        prompts = {} # ordering prompts...
        unordered_prompts = experiment.protocol.prompts
        for idx in range(0,len(unordered_prompts)):
            prompts[f'{idx}']=unordered_prompts[f'{idx}']
        experiment.protocol.prompts = prompts
        return experiment

    def save_as_tif(self):
        pass

    def update(self,param_dict):
        """ Method to update an experiment.
        """
        for key in param_dict.keys():
            if hasattr(self.setup.instrument.config,key):
                self.setup.instrument.config.set_params({key:param_dict[key]})
            elif hasattr(self.setup.sample.config,key):
                self.setup.sample.config.set_params({key:param_dict[key]})
            elif hasattr(self.measurement.config,key):
                self.measurement.config.set_params({key:param_dict[key]})
            else:
                continue
        sample = SampleGenerator(param_dict=self.setup.sample.config.__dict__).generate_sample()
        instrument = Instrument(param_dict = self.setup.instrument.config.__dict__)
        self.setup = Setup(instrument=instrument,sample=sample)
        self.measurement = Measurement(param_dict=self.measurement.config.__dict__)
        #exp = Experiment(**{'setup':setup,'measurement':measurement,'protocol':protocol})
        pass

class ExperimentGenerator():

    def __init__(self,**kwargs):
        #Experiment.__init__(self,**kwargs)
        pass

    def get_default(self, **kwargs):
        """
        Create a default experiment. If a distance is provided, a random nanoruler with d
        is created. Otherwise, the same molecule position is used every time.
        """
        exp = Experiment()
        exp.update(kwargs)

        type = kwargs.get('type', None)
        sample_dict, meas_dict, inst_dict, prot_dict = {},{},{},{}
        
        if type=='line-scan':
            meas_dict = {'block_size': 1
                , 'L': LAMBDA/2
                , 'meas_mode': 'phase'
                , 'iteration_depth': 1
                }
            inst_dict = {'excPSF' : 'sine', 'Lambda0': LAMBDA}
            exp.update(sample_dict|meas_dict|inst_dict)
            for i in range(exp.measurement.config.iteration_depth*exp.measurement.config.repetitions):
                prot_dict = prot_dict | {
                                    f'{i}': {
                                        'set' : {
                                            'measurement': {}
                                            ,'instrument': {}
                                        }
                                    }
                                }
        elif type=='minflux':
            pre_loc_rep = 5
            meas_dict = {'block_size': 1
                ,'num_positions':3
                , 'meas_mode': 'minflux'
                ,'iteration_depth':4
                }
            inst_dict = {'excPSF' : 'sine', 'Lambda0': LAMBDA}
            exp.update(sample_dict|meas_dict|inst_dict)
            for i in range(exp.measurement.config.iteration_depth):
                if i<exp.measurement.config.iteration_depth-1:
                    for j in range(pre_loc_rep): # pre-localization phase TODO: estimation!
                        prot_dict = prot_dict | {
                                        f'{i*pre_loc_rep + j}': {
                                            'set' : {
                                                'measurement': {'L': exp.measurement.config.L * SHRINK_FACTOR**i}
                                                ,'instrument': {'alpha': exp.setup.instrument.config.alpha}#* LIGHT_FACTOR**i
                                            }
                                        }
                                    }
                        if i+j!=0:
                            prot_dict = prot_dict | {
                                        f'est{i*pre_loc_rep + j}': {
                                            'estimate' : {}
                                        }
                                    }
                    #append some prompt for pre-localization, depending on iteration_depth and repeat e.g. 100 times
                else:
                    for j in range(exp.measurement.config.repetitions):
                        prot_dict = prot_dict | {
                                        f'{i*pre_loc_rep+j}': {
                                            'set' : {
                                                'measurement': {'L': exp.measurement.config.L * SHRINK_FACTOR**i}
                                                ,'instrument': {'alpha': exp.setup.instrument.config.alpha}#* LIGHT_FACTOR**i
                                            }
                                        }
                                    }
                        if j%10==0 and i+j!=0:
                            prot_dict = prot_dict | {
                                        f'est{i*pre_loc_rep + j}': {
                                            'estimate' : {}
                                        }
                                    }
                    # append for final iteration and repeat repetition times.
                #TODO: create protocol for iterations
                # modify L and brightness
        exp.protocol = Protocol(param_dict = prot_dict)
        return exp

    def get_CRB_scan(self, mol_pos = np.array([[-10.,-10.],[10.,10.]]), meas_pos = np.array([[-10.,-10.],[10.,10.]])):
        sample_dict = {'n_molecules': 2
            , 'photon_budget': [2*10**4, 4*10**4]
            , 'beta': 0.004
            , 'mol_pos': mol_pos# has to be float!!!
            , 'molecule_brightness' : 1.0
            }
        meas_dict = {'block_size': 1
            , 'num_positions': int(meas_pos.size/2)
            , 'iteration_depth': 1
            , 'meas_mode': 'custom'
            , 'meas_pos' : meas_pos
            ,'dwell_time' : 100 * 10**-6
            }
        inst_dict = {'excPSF' : 'sine', 'alpha': 3e5}
        prot_dict = {}
        for i in range(0,1):
            prot_dict = prot_dict | {
                                f'{i}': {
                                        'set' : {
                                            'measurement': {'phi0': 0.}
                                            ,'instrument': inst_dict
                                        }
                                    }
                            }
        sample = SampleGenerator(param_dict=sample_dict).generate_sample()
        setup = Setup(sample=sample)
        measurement = Measurement(param_dict=meas_dict)
        protocol = Protocol(param_dict = prot_dict)
        exp = Experiment(**{'setup':setup,'measurement':measurement,'protocol':protocol})
        return exp

class Protocol:
    """
    Describe the different measurements. Used as a *recipe* during an experiment
    on what kind of measurement to perform next.
    """
    def __init__(self, param_dict={}):
        self.prompts = {}
        self.prompts.update(param_dict)
        pass

class Record(Trace,Pattern):
    """
    Class for measurement records.
    Inherits from Pattern-class and Trace-class
    Its attributes hold all information on the measurement sites,
    e.g. positions, incident brightness, PSF orientation etc.
    """
    def __init__(self)->None:
        Trace.__init__(self)
        Pattern.__init__(self)

    def truncate_record(self, record, n1, n2):
        """
        Method to truncate the pattern of a record up to the n-th element (n-th element
        is not contained!)
        :param record: measurement record object
        :param n1: start idx
        :param n2: stop idx
        :return: modified MeasurementRecord object
        """
        new_record = copy.copy(record)
        rec_dict = vars(new_record)
        for key in rec_dict.keys():
            val = rec_dict[key][n1:n2]
            setattr(new_record, key, val)
        #phot, lam, x0, c0, phi, phi0 = record.photons, record.lam, record.x0, record.c0, record.phi, record.phi0
        #new_record.photons, new_record.lam, new_record.x0, new_record.c0, new_record.phi, new_record.phi0 = phot[n1:n2], lam[n1:n2], x0[n1:n2], c0[n1:n2], phi[n1:n2], phi0[n1:n2]
        return new_record
    
    def pick_lines_from_record(self, record, n1, n2): # depricated 06.02.2023
        """
        Method to pick lines n1 through n2 from record.
        Used to construct a measurement record for a trace segment with a certain number of lines.
        line #n2 is not contained!
        """
        new_record = copy.copy(record)
        rec_dict = vars(new_record)
        for key in rec_dict.keys():
            arr = rec_dict[key].reshape((-1,record.config.pixel_count,2))
            val = val.reshape((-1,record.config.block_size,record.config.pixel_count,2))
            val = val.reshape((-1,2))
            setattr(new_record, key, val)
        return None

class Log:
    """
    Class that provides logging capabilities to document steps in the program.
    """

    def __init__(self)->None:
        pass

    def register_entity(self):
        """
        Method to register an object that has to be watched.
        Serves to document on-change behaviour of object.
        """
        pass

    def unregister_entity(self):
        pass

    def write_to_log(self):
        pass

    def save_log(self):
        pass