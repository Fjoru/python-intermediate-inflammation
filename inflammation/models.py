"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains
inflammation data for a single patient taken over a number of days
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    :returns: A 2D numpy array of inflammation data
    """
    return np.loadtxt(fname=filename, delimiter=",")


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.

    :param data: A 2D numpy array of inflammation data
    :returns: A 1D numpy array of daily mean inflammation values
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.

    :param data: A 2D numpy array of inflammation data
    :returns: A 1D numpy array of daily max inflammation values
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.

    :param data: A 2D numpy array of inflammation data
    :returns: A 1D numpy array of daily min inflammation values
    """
    return np.min(data, axis=0)


def patient_normalise(data: np.ndarray) -> np.ndarray:
    """Normalise patient data from a 2D inflammation data array.

    NaNs and negative values are replaced with 0s.

    :param data: A 2D numpy array of inflammation data
    :returns: A 2D numpy array of normalised inflammation data
    """
    if np.any(data < 0):
        raise ValueError("Inflammation values should not be negative")
    
    max = np.max(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / max[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised
