"""
Module that provides functionality in order to post-process data, here construct experiment object from input data.

@author: Thomas Arne Hensel, 2023
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import copy as copy

from lib.data_handling.utilities import Converter
from lib.data_handling.data_preprocessing import Extractor
from lib.simulation.experiment import Experiment, ExperimentGenerator


class Constructor:
    """
    Class that provides a constructor object in order to create experiments from loaded data. Can be full experiments
    (from simulations) or extracted data from real experiments.
    """

    def __init__(self,obj,collect_artefacts=True):
        """
        Initialize the Constructor object.

        :param obj: Experiment object or path to data (tif, tiff, yaml).
        :type obj: Experiment or str
        :param collect_artefacts: Whether to collect artefacts during extraction (default is True).
        :type collect_artefacts: bool
        """
        self.obj = obj
        self.ext = Extractor(self.obj,collect_artefacts=collect_artefacts)
        pass

    def get_experiments(self, batchsize=1, agnostic=True, fast_return=False):
        """
        Construct experiments from a file or experiment object, segmented at bleaching steps.
        Currently functional for phase scans!!!

        :param batchsize: Number of lines to be averaged with moving average.
        :type batchsize: int, optional
        :param agnostic: Boolean, default=True. Forget about all information from the experiment.
        :type agnostic: bool, optional
        :param fast_return: Optional direct return of the experiment object. The routine effectively only reshapes the record.
        :type fast_return: bool, optional
        :return: List of experiments.
        :rtype: list
        """
        exp = None
        if isinstance(self.obj,str):
            filename = os.fsdecode(self.obj)
            if ((filename.endswith('.tif') or filename.endswith('.tiff')) and (('c003' in filename) or ('source' in filename))):
                imarray = self.ext.get_array_from_tif(filename)
            elif filename.endswith('.yaml'):
                exp = Experiment().load(filename)
                imarray = Converter().reshape_record(exp,export=False,show=False).photons
                if agnostic:
                    exp = None
            else: # if string provided but doesn't match required format:
                return []
        else:
            exp = self.obj
            imarray = Converter().reshape_record(exp,export=False,show=False).photons
            if agnostic:
                exp = None
        if not agnostic and fast_return:
            exp.measurement.config.set_params({'seg_idcs': None})
            return [exp]
        avg_scan_seg, seg_idcs = [], []
        try:            
            avg_scan_seg, seg_idcs = self.ext._get_moving_avg_segments(imarray, self.ext.block_size, batchsize)
        except:
            print('failed!')

        experiments = []
        if np.any([np.concatenate(seg).size==0 for seg in avg_scan_seg]) or len(avg_scan_seg)!=3:
            return None
        else:
            N_bgr = self._get_bgr_estimate(avg_scan_seg)
            brightness = self._get_brightness_estimate(avg_scan_seg, N_bgr)
            beta = self._get_beta_estimate(N_bgr, brightness)
            if exp is None:
                exp = ExperimentGenerator().get_default(**{'type':'line-scan', 'repetitions': imarray.shape[0]//self.ext.block_size, 'block_size': self.ext.block_size})
                exp.perform(mode='strawman')
            for idx, seg in enumerate(avg_scan_seg):
                reps = imarray.shape[0]/(2 * self.ext.block_size)
                tmp_exp = copy.deepcopy(exp)
                tmp_exp.setup.sample.config.set_params({'n_molecules':2-idx, 'beta': beta})
                tmp_exp.measurement.config.set_params({'seg_idcs': seg_idcs[idx],'repetitions': int(reps)})
                #etc for instrument, measurement
                tmp_exp.record = self._construct_record(seg, tmp_exp, brightness[idx], restore_line_order=True, agnostic=agnostic)
                if agnostic:
                    tmp_exp.setup.sample.config.set_params({'molecule_brightness': 1.0}) # set molecule brightness to one if it has been estimated in c0
                experiments += [ tmp_exp ]
        if False:
            fig, ax = plt.subplots(figsize=(8,8))
            ax.pcolormesh(imarray,label='x')
            plt.show()
        return experiments # return list of experiments
    
    def _get_bgr_estimate(self,avg_scan_seg):
        """
        Retrieve the average number of background photons from 0M experiment.

        :param avg_scan_seg: List containing x and y segments.
        :type avg_scan_seg: list
        :return: Average number of background photons in each axis.
        :rtype: numpy.ndarray
        """
        return np.average(avg_scan_seg[2],axis=(2,1))
    
    def _get_beta_estimate(self,N_bgr,brightness):
        """
        Calculate the beta estimate.

        :param N_bgr: Average number of background photons in each axis.
        :type N_bgr: numpy.ndarray
        :param brightness: Estimated brightness.
        :type brightness: list
        :return: Beta estimate.
        :rtype: numpy.float64
        """
        beta = np.average(N_bgr)/(2*np.average(np.linalg.norm(brightness,axis=1)**2))
        return beta
    
    def _get_brightness_estimate(self,avg_scan_seg,N_bgr):
        """
        Calculate the brightness estimate.

        :param avg_scan_seg: List containing x and y segments.
        :type avg_scan_seg: list
        :param N_bgr: Average number of background photons in each axis.
        :type N_bgr: numpy.ndarray
        :return: Estimated brightness.
        :rtype: list
        """
        x_bgr, y_bgr = N_bgr[0], N_bgr[1]
        brightness = []
        for idx,seg in enumerate(avg_scan_seg):
            x_scans, y_scans = seg[0], seg[1]
            if idx==2:
                c0x, c0y = np.average(brightness,axis=1)
                brightness.append([c0x,c0y])
            else:
                c0x = np.sqrt(0.5 * (np.average(x_scans)-x_bgr)/(2-idx)) # factor 0.5 due to two incident plane waves
                c0y = np.sqrt(0.5 * (np.average(y_scans)-y_bgr)/(2-idx))
                brightness.append([c0x,c0y])
        return brightness
    
    def _construct_record(self, seg, exp, brightness, restore_line_order=False, agnostic=True):
        """
        Construct a measurement record from experimental or simulated data, e.g., after averaging.

        Takes a scan segment (e.g., 2M or 1M or 0M) and divides it into x and y blocks.
        Each segment is a list of [x_avg_scan, y_avg_scan].
        Those pairs form 1 record by combining all their measurement sites and counts (considering the block_size!).
        This procedure is repeated, and a dictionary of measurement records is returned.
        If a MeasurementRecord object is provided, the counts are substituted by seg, and the record is truncated.
        If agnostic mode is chosen, c0 is estimated from record; otherwise, the default from the provided experiment is chosen.

        :param seg: List of x/y segments.
        :type seg: list
        :param exp: Experiment object with information on how to construct the record.
        :type exp: Experiment
        :param brightness: Estimated brightness.
        :type brightness: list
        :param restore_line_order: Option to assemble blocks as in the original scan or keep plain x-y scans.
        :type restore_line_order: bool
        :param agnostic: If True, be agnostic about c0 and estimate from counts.
        :type agnostic: bool
        :return: Record object.
        :rtype: MeasurementRecord
        """
        x_scans, y_scans = seg[0], seg[1]
        x_br, y_br = brightness
        seg_idcs = exp.measurement.config.seg_idcs
        block_size = exp.measurement.config.block_size
        
        dl = len(x_scans)-len(y_scans)
        if dl != 0: # check whether x and y scans have equal length in first dimension and truncate accordingly (should already be fulfilled)
            raise Exception('#x_scans != #y_scans')
        if restore_line_order:
            seg_idcs2 = seg_idcs
        else:
            seg_idcs2 = np.array([0,0])
        
        #TODO: correct estimate of c0 with subtraction of N_bgr
        new_exp = copy.deepcopy(exp)
        new_rec = Converter().reshape_record(new_exp,export=False)
        new_rec = Converter().partition_record(new_rec, block_size, seg_idcs=None, truncate=False)
        if agnostic:
            new_rec.c0[0] = np.full(new_rec.c0[0].shape, x_br)
            new_rec.c0[1] = np.full(new_rec.c0[1].shape, y_br)
        rec_dict = vars(new_rec)

        for key in rec_dict.keys():#populate record with truncated entries
            #if rec_dict[key].size!=0:
            x_scans , y_scans = rec_dict[key]
            siz = np.array([scan.size for scan in [x_scans, y_scans]])
            if np.all(siz!=0) and siz.size!=0:
                val = self._assemble_blocks(x_scans[seg_idcs[0]:seg_idcs[1]], y_scans[seg_idcs[0]:seg_idcs[1]], seg_idcs2)#
                setattr(new_rec, key, val)
        if agnostic: # if agnostic, populate photons from segments (not agnostic: entries are already poulated)
            new_rec.photons = self._assemble_blocks(seg[0], seg[1], seg_idcs2)
            new_rec.sig_photons = 0 * new_rec.photons
            new_rec.bgr_photons = 0 * new_rec.photons
        return new_rec

    def _assemble_blocks(self, x_scans, y_scans, seg_idcs):
        """
        Re-assemble the x- and y-segments to an array with alternating x- and y-blocks
        according to the segment indices.

        :param x_scans: List of x-scans.
        :type x_scans: list
        :param y_scans: List of y-scans.
        :type y_scans: list
        :param seg_idcs: Segment indices.
        :type seg_idcs: numpy.ndarray
        :return: Assembled array with alternating x- and y-blocks.
        :rtype: numpy.ndarray
        """
        start, end = (-seg_idcs%self.ext.block_size)[0], (seg_idcs%self.ext.block_size)[1]
        xy_block = np.empty((0,))
        if start!=0:
            x_init_block = np.concatenate(x_scans[0:start],axis=0)
            y_init_block = np.concatenate(y_scans[0:start],axis=0)
            xy_block = np.concatenate((x_init_block,y_init_block),axis=0)     
        
        for idx0,idx in enumerate(range(start,len(x_scans)-end,self.ext.block_size)): # assemble the record
            x_block = np.concatenate(x_scans[idx:idx+self.ext.block_size],axis=0)
            y_block = np.concatenate(y_scans[idx:idx+self.ext.block_size],axis=0)
            tmp_xy_block = np.concatenate((x_block,y_block),axis=0)
            if idx0==0 and xy_block.size==0:
                xy_block = tmp_xy_block
            else:
                xy_block = np.concatenate((xy_block, tmp_xy_block),axis=0)
        if end != 0:
            x_block = np.concatenate(x_scans[-end:],axis=0)
            y_block = np.concatenate(y_scans[-end:],axis=0)
            tmp_xy_block = np.concatenate((x_block,y_block),axis=0)
            xy_block = np.concatenate((xy_block, tmp_xy_block),axis=0)
        return xy_block