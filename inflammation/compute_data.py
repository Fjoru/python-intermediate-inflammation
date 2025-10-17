"""Module containing mechanism for calculating standard deviation between datasets.
"""

import glob
import os
import numpy as np

from inflammation import models, views

class CSVDataSource:
    """Class containing the inflammation data from CSV files."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):
        """Loads all inflammation data from CSV files within a directory.

        :returns: A list of 2D numpy arrays, each containing inflammation data
        """
        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.csv'))
        
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data CSV files found in path {self.data_dir}")
        data = map(models.load_csv, data_file_paths)
        return data


class JSONDataSource:
    """Class containing the inflammation data from JSON files."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):
        """Loads all inflammation data from JSON files within a directory.

        :returns: A list of 2D numpy arrays, each containing inflammation data
        """
        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.json'))
        
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data JSON files found in path {self.data_dir}")
        data = map(models.load_json, data_file_paths)
        return data

def compute_standard_deviation_by_day(data):
    """Calculates the standard deviation by day between datasets."""
    means_by_day = map(models.daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))
    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)

    return daily_standard_deviation



def analyse_data(data_class):
    """Calculates the standard deviation by day between datasets.

    Gets all the inflammation data from CSV files within a directory,
    works out the mean inflammation value for each day across all datasets,
    then plots the graphs of standard deviation of these means."""
    data = data_class.load_inflammation_data()

    daily_standard_deviation = compute_standard_deviation_by_day(data)


    return daily_standard_deviation
