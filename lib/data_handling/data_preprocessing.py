"""
Module that provides functionality in order to pre-process data, here extract relevant data from input.

@author: Thomas Arne Hensel, 2023
"""
import os
import numpy as np
import ruptures as rpt
from PIL import Image
import copy as copy

from lib.data_handling.utilities import Converter
from lib.plotting.artefacts import Artefacts, Trace_Figures
import lib.utilities as ut
from lib.constants import *

class Extractor:
    """
    Class to extract data from files to hand the data over to the constructor class.
    Currently functional for phase scans.
    """

    def __init__(self, obj=None,collect_artefacts=True)->None:
        """
        Initialize the Extractor.

        :param obj: Object/file to be analyzed (either experiment-object, yaml or tif).
        :type obj: str or None
        :param collect_artefacts: Optional boolean. If True, collect artefacts; otherwise, don't.
        :type collect_artefacts: bool
        """
        self._set_params(obj)
        self.artefacts = Artefacts()
        self.collect_artefacts=collect_artefacts
        return None

    def _set_params(self,obj):
        """
        Fetch the relevant experimental parameters from the parameter file of the measurement.

        :param obj: Object/file to be analyzed (either experiment-object, yaml or tif).
        :type obj: str or None
        """
        #PixelCount, pixelSize, imageSize, scanDevs x-phase,lineStep 0  times 10 (blocksize)
        self.block_size = BLOCK_SIZE # [1] number of lines/scans that are
        self.dwell_time = DWELL_TIME # [s]
        self.scan_range = SCAN_RANGE # [nm] 'imageSize'
        if isinstance(obj,str):
            filename = os.fsdecode(obj)
            if filename.endswith('.tif') or filename.endswith('.tiff'): # TODO: read from parameter txt-file
                imarray = self.get_array_from_tif(filename)
                self.pixel_count = imarray.shape[1] # [1]
            elif filename.endswith('.yaml'):
                exp = ut.Importer().load_yaml(filename)
                self.block_size = exp.measurement.config.block_size
                self.dwell_time = exp.measurement.config.dwell_time
                self.pixel_count = exp.measurement.config.num_positions # [1]
                self.scan_range = exp.measurement.config.L # [nm]
        elif obj is None: # default parameters if Extractor is initialized without passing an object
            pass
        else:
            try:
                exp = obj
                self.block_size = exp.measurement.config.block_size
                self.dwell_time = exp.measurement.config.dwell_time
                self.pixel_count = exp.measurement.config.num_positions # [1]
                self.scan_range = exp.measurement.config.L
            except:
                raise Exception('Extractor could not be called, please pass suitable object!')
        return None

    def get_array_from_tif(self,path):
        """
        Load a TIF image and convert it into a numpy array.

        :param path: Absolute path to the file.
        :type path: str
        :return: ndarray - Shape (2,)
        """
        im = Image.open(path)
        imarray = np.array(im)
        return imarray
    
    def get_bleach_idx(self, signal,segment_size = None):
        """
        Detect bleaching steps in a single record and divide it into 2M, 1M, 0M traces.

        :param signal: MeasurementRecord
        :type signal: MeasurementRecord obj
        :param segment_size: Optional size of the segment.
        :type segment_size: int or None
        :return: List of new records of the partial traces corresponding to 0M, 1M or 2M.
        :rtype: list
        """
        norm_sig=signal/max(signal)
        if segment_size is None:
            segment_size = np.int_(2*self.block_size)
        model = 'l2'
        algo = rpt.Pelt(model=model, min_size = segment_size, jump = 2).fit(norm_sig)
        bleach_idx = algo.predict(pen=0.5)#n_bkps=2) 4
        #rpt.show.display(signal, [], bleach_idx, figsize=(10, 6))
        #plt.show()
        return bleach_idx

    def _get_moving_avg_segments(self, imarray, block_size, batchsize, bleach_idx = None, n_cut = 2):
        """
        Process an image array. Split array into x- and y-axis. Get a moving average of the arrays with blocksize lines.
        Segment the averaged traces by bleaching steps.

        :param imarray: 2D array from image data.
        :type imarray: ndarray
        :param block_size: Block size.
        :type block_size: int
        :param batchsize: Integer, moving avg batch size.
        :type batchsize: int
        :param bleach_idx: Optional list of indices to segment an array by hand.
        :type bleach_idx: list or None
        :param n_cut: Number of lines to throw away before and after the bleaching step.
        :type n_cut: int
        :return: List of averaged segments of shape (bleach_steps, no of axes, averaged lines) and start/end idcs of each segment (idx of line of imarray!)
        :rtype: list
        """
        x_imarray, y_imarray = Converter().partition_array(imarray, block_size)
        # define moving avg
        def moving_average(a, n=10): # moving average of scans
            ret = np.cumsum(a, axis=0, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n
        # segmentation: get bleach_idx and truncate segments to same length in each axis.
        segments = []
        if bleach_idx is None:
            bleach_idx = []
            # temporary 3-line averaging to detect bleachingsteps more reliably
            avg_xscans = moving_average(x_imarray,n=3) # average scans of one block.
            avg_yscans = moving_average(y_imarray,n=3)
            avg_scans = [avg_xscans, avg_yscans]
            for i,scan in enumerate(avg_scans):
                line_avg = np.average(scan,axis=1)
                curr_idx = self.get_bleach_idx(line_avg, segment_size = None)
                bleach_idx.append(curr_idx)
            new_idx = np.array(bleach_idx).T
            new_idx = np.insert(new_idx,0,np.array([0,0]),axis=0)
            min_idx = []
            line_avgs = np.average([x_imarray,y_imarray],axis=2)
            for i in range(len(new_idx)-1):
                new_idcs = np.array([np.max([n_cut,np.max(new_idx[i])+n_cut]),np.max([n_cut,np.min(new_idx[i+1])-n_cut])])
                #x_median, y_median = np.median(x_line_avg), np.median(y_line_avg)#calculate medians of segment
                max_size = (2-i)*10 + 30
                if new_idcs.size==2 and np.diff(new_idcs)>max_size: # find a sub-segment with minimal variance and bias
                    step = 1
                    tmp_idcs = np.arange(new_idcs[0], new_idcs[1] - max_size, step)
                    #min_combined_var = float('inf')
                    min_std = float('inf')
                    for idx in tmp_idcs:
                        sub_seg = line_avgs[:,idx:idx+max_size]
                        curr_std =  np.linalg.norm(np.std(sub_seg,axis=1))#calculate average of standard deviation in both axes
                        if curr_std < min_std:
                            min_std = curr_std
                            new_idcs = np.array([idx,idx+max_size])
                min_idx.append([new_idcs])
            min_idx = np.concatenate(min_idx, axis=0)
            for i, scan in enumerate([x_imarray, y_imarray]):#avg_scans
                tmp_segments = np.split(scan,min_idx.flatten())[1::2]
                segments.append(tmp_segments)
        # do the averaging...
        batchsizes = [[min(seg.shape[0]-1, batchsize) for seg in axis] for axis in segments]
        avg_segments = []
        avg_segments = [[moving_average(seg,n=batchsizes[i][j]) for j,seg in enumerate(axis)] for i, axis in enumerate(segments)]
        avg_segments = [[avg_segments[0][i],avg_segments[1][i]] for i in range(len(avg_segments[0]))]
        seg_idcs = [min_idx[i]-np.array([0,offset-1]) for i,offset in enumerate(batchsizes[0])] # corresponds to lines in original records (by axis), reduced via moving_average

        if self.collect_artefacts:
            fig, _ = Trace_Figures().fig_segmented_trace(line_avgs, seg_idcs)
            self.artefacts.add_figures([fig],['bleaching_steps'])

            fig, _ = Trace_Figures().fig_raw_trace(imarray, x_imarray, y_imarray)
            self.artefacts.add_figures([fig],['extracted_trace'])

        return avg_segments, seg_idcs