"""
Utility function shared among the data_handling module.
@author: Thomas Arne Hensel, 2023
"""

import numpy as np
import copy as copy
import matplotlib.pyplot as plt
from PIL import Image
from lib.utilities import Labeler

class Converter:
    """
    Convert back and forth between different data formats, styles, and shapes.

    Methods:
    - __init__(): Constructor for the Converter class.
    - partition_record(record, block_size, seg_idcs=None, truncate=False): Partition an array into x and y-scans in the record.
    - partition_array(array, block_size, seg_idcs=None, truncate=False): Partition an array into x and y-scans.
    - join_array(array, block_size, seg_idcs=None, truncate=False): Join x and y-scans back into a single array.

    """
    def __init__(self):
        pass

    def partition_record(self, record, block_size, seg_idcs=None, truncate=False):
        """
        Partition an array into x and y-scans in the record.

        :param record: Reshaped record that has to be partitioned into x-y axis.
        :type record: Record
        :param block_size: Size of the data block.
        :type block_size: int
        :param seg_idcs: Segment indices, defaults to None.
        :type seg_idcs: tuple, optional
        :param truncate: Option to produce imarrays of equal length, defaults to False.
        :type truncate: bool, optional
        :return: Partitioned record.
        :rtype: Record
        """
        rec = copy.deepcopy(record)
        dict = record.__dict__
        for key in dict.keys():
            siz = np.array([el.size for el in dict[key]])
            if np.all(siz!=0) and siz.size!=0:
                arr = dict[key]
                [x_array, y_array] = self.partition_array(arr, block_size, seg_idcs=seg_idcs, truncate=truncate)
                setattr(rec, key, [x_array, y_array])
            else:
                continue
        return rec

    def partition_array(self, array, block_size, seg_idcs=None, truncate=False):
        """
        Partition an array into x and y-scans.

        :param array: Array to be partitioned.
        :type array: numpy.ndarray
        :param block_size: Size of the data block.
        :type block_size: int
        :param seg_idcs: Segment indices, defaults to None.
        :type seg_idcs: tuple, optional
        :param truncate: Option to produce imarrays of equal length, defaults to False.
        :type truncate: bool, optional
        :return: List containing x and y arrays.
        :rtype: list of numpy.ndarray
        """
        # partition imarray into axes
        line_number = array.shape[0]
        if seg_idcs is None:
            start, end = 0, 0
        else:
            start, end = (-seg_idcs%block_size)[0], (seg_idcs%block_size)[1]
            x_array = array[0:start]
            y_array = array[start:2*start]
        block_number = int((line_number-start-end)/(block_size))
        #if block_number > 0:            
        for i in range(0,block_number-1,2):
            x_block = array[2*start+i*block_size:2*start + (i+1)*block_size]
            y_block = array[2*start + (i+1)*block_size: 2*start + (i+2)*block_size]
            if x_block.size != y_block.size and truncate==True:
                continue
            elif i==0 and start == 0:
                x_array = x_block
                y_array = y_block
            else:
                x_array = np.concatenate((x_array,x_block),axis=0)
                y_array = np.concatenate((y_array,y_block),axis=0)
        if end != 0:
            x_block = array[-2*end:-end]
            y_block = array[-end:]
            x_array = np.concatenate((x_array,x_block),axis=0)
            y_array = np.concatenate((y_array,y_block),axis=0)
        return [x_array, y_array]
    
    def join_array(self, array, block_size, seg_idcs=None, truncate=False):
        """
        Join x and y-scans back into a single array.

        :param array: List containing x and y arrays.
        :type array: list of numpy.ndarray
        :param block_size: Size of the data block.
        :type block_size: int
        :param seg_idcs: Segment indices, defaults to None.
        :type seg_idcs: tuple, optional
        :param truncate: Option to produce imarrays of equal length, defaults to False.
        :type truncate: bool, optional
        :return: Joined array.
        :rtype: numpy.ndarray
        """
        x_array, y_array = array
        array_len = len(x_array) + len(y_array)
        result = np.empty((array_len, x_array.shape[1]), dtype=x_array.dtype)
        # partition imarray into axes
        line_number = array_len
        if seg_idcs is None:
            start, end = 0, 0
        else:
            start, end = (-seg_idcs%block_size)[0], (seg_idcs%block_size)[1]
            result[0:start] = x_array[0:start]
            result[start:2*start] = y_array[0:start]
        block_number = int((line_number-start-end)/(block_size))
        for i in range(0,block_number-1,2):
            result[2*start+i*block_size:2*start + (i+1)*block_size]=x_array[2*start+i*block_size:2*start + (i+1)*block_size]
            result[2*start + (i+1)*block_size: 2*start + (i+2)*block_size]=y_array[2*start+i*block_size:2*start + (i+1)*block_size]
        if end != 0:
            result[-2*end:-end]=x_array[-2*end:-end]
            result[-end:]=y_array[-2*end:-end]
        return result

    def reshape_record(self,experiment,export=False,show=False):
        """
        Retrieve an image array from an experiment-object.

        Creates a deep-copy of the record attribute of the experiment object,
        modifies the record, and returns it.

        :param experiment: Experiment object.
        :type experiment: Experiment
        :param export: Option to export to a tif file, defaults to False.
        :type export: bool, optional
        :param show: Option to display the reshaped array, defaults to False.
        :type show: bool, optional
        :return: Record object.
        :rtype: Record
        """
        exp = copy.deepcopy(experiment)
        # reshape to line by line x,y array
        xypixels = experiment.measurement.config.num_positions
        dict = exp.record.__dict__
        for key in dict.keys():
            siz = np.array([el.size for el in dict[key]])
            if np.all(siz!=0):
                curr_shape = list(dict[key].shape)
                new_shape = tuple([-1,xypixels] + curr_shape[1:])
                dict[key] = np.reshape(dict[key],new_shape)
                setattr(exp.record, key, dict[key])
        if show:
            fig, ax = plt.subplots(figsize=(8,8))
            ax.pcolormesh(exp.record.photons,label='x',linewidth=0,rasterized=True)
            plt.show()
        # export array to tif
        if export:
            data = Image.fromarray(exp.record.photons)
            filename = Labeler().stamp('full-record')
            data.save(filename + '.tif')
        return exp.record