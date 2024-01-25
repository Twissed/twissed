"""stats.py file

    Python package for beam dynamics analysis in laser-plasma acceleration
    author:: Damien Minenna <damien.minenna@cea.fr>
    date = 21/07/2023
"""

import numpy as np
import pandas as pd
import scipy.constants as const
from typing import Union

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("WARNING: matplotlib package not found")

# twissed
from .tools import deprecated

# TODO: Add error weighted_mad, weighted_std, ...


def weighted_avg(
    a: Union[float, np.ndarray], weights: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    r"""
    Calculate the weighted average of array a.

    .. math::

        \langle {\bf a} \rangle = \frac{\sum_i w_i a_i}{\sum_i w_i} \, ,

    Args:
        arr (Union[float, np.ndarray]): 1D numpy array
        weights (Union[float, np.ndarray]): 1D numpy array

    Returns:
        Union[float, np.ndarray]: weighted average.
    """
    # Check arrays size
    if np.shape(np.asarray(a)) != np.shape(np.asarray(weights)):
        raise ValueError("ERROR: array and weights do not have the same length!")

    # Check if input contains data
    if not np.any(weights) and not np.any(a):
        raise ValueError("ERROR: Passed array is empty!")

    else:
        # Calculate the weighted average
        return np.average(a, weights=weights)


def weighted_std(
    arr: Union[float, np.ndarray], weights: Union[float, np.ndarray], verbose=True
) -> Union[float, np.ndarray]:
    """
    Calculate the weighted standard deviation.

    Args:
        arr (Union[float, np.ndarray]): 1D numpy array
        weights (Union[float, np.ndarray]): 1D numpy array

    Returns:
        Union[float, np.ndarray]: weighted standard deviation.
    """
    try:
        # Check arrays size
        if np.shape(np.asarray(arr)) != np.shape(np.asarray(weights)):
            raise ValueError("ERROR: array and weights do not have the same length!")

        # Check if input contains data
        if not np.any(weights) and not np.any(arr):
            raise ValueError("ERROR: Passed array is empty!")
        else:
            # Calculate the weighted standard deviation
            average = np.average(arr, weights=weights)
            variance = np.average((arr - average) ** 2, weights=weights)
            return np.sqrt(variance)
    except:
        if verbose:
            print("ERROR: Cannot used weighted_std!")
        return -1


def weighted_med(
    arr: Union[float, np.ndarray], weights: Union[float, np.ndarray], verbose=True
) -> Union[float, np.ndarray]:
    """
    Compute the weighted median

    Args:
        a (Union[float, np.ndarray]): 1D numpy array
        weights (Union[float, np.ndarray]): 1D numpy array

    Returns:
        Union[float, np.ndarray]: weighted median
    """
    try:
        quantile = 0.5
        if not isinstance(arr, np.matrix):
            arr = np.asarray(arr)

        if not isinstance(weights, np.matrix):
            weights = np.asarray(weights)

        if arr.shape != weights.shape:
            raise ValueError("ERROR: array and weights do not have the same length!")

        ind_sorted = np.argsort(arr)
        sorted_data = arr[ind_sorted]
        sorted_weights = weights[ind_sorted]

        Sn = np.cumsum(sorted_weights)
        # Center and normalize the cumsum (i.e. divide by the total sum)
        Pn = (Sn - 0.5 * sorted_weights) / Sn[-1]
        # Get the value of the weighted median
        return np.interp(quantile, Pn, sorted_data)

    except:
        if verbose:
            print("ERROR: Cannot used weighted_med!")
        return -1


def weighted_mad(a, w):
    """
    Compute the weighted median absolute

    Args:
        a (float): 1D numpy array
        weights (float): 1D numpy array

    Returns:
        float: median
    """
    med = weighted_med(a, w)
    mad = weighted_med(np.abs(a - med), w)
    return mad
