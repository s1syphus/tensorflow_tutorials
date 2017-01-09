"""
Dataset class

Could use a better description at some point
"""

from abc import ABCMeta
from abc import abstractmethod
import os
import tensorflow as tf


class Dataset(object):
    """A simple class for handling data sets."""
    __metaclass__ = ABCMeta

    def __init__(self, name, subset, data_dir):
        """Initialize dataset using a subset and the path to the data."""
        assert subset in self.available_subsets(), self.available_subsets()
        self.name = name
        self.subset = subset
        self.data_dir = data_dir

    @abstractmethod
    def num_classes(self):
        """Returns the number of classes in the data set."""
        pass

    @abstractmethod
    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset."""
        pass

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation']

    def data_files(self):
        """
            Returns a python list of all (sharded) data subset files.
            Returns:
                python list of all (sharded) data set files.
            Raises:
            ValueError: if there are not data_files matching the subset.
        """
        # tf_record_pattern = os.path.join(self.data_dir, '%s-*' % self.subset)
        tf_record_pattern = os.path.join(self.data_dir, '*')
        data_files = tf.gfile.Glob(tf_record_pattern)
        return data_files

    def reader(self):
        """
        Return a reader for a single entry from the data set.
            See io_ops.py for details of Reader class.
            Returns:
        Reader object that reads the data set.
        """
        return tf.TFRecordReader()
