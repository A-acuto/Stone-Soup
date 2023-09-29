"""
Navigation functions
--------------------

Functions used within multiple navigation classes in Stone Soup
"""
import numpy as np

from . import dotproduct
from ..types.array import StateVector, StateVectors

# here I would place the various components regarding functions that I might
# need for navigation

def EarthSpeedFlatSq(dx, dy):
    r"""Calculate the Earth speed flat vector

    Parameters
    ----------
    dx : float, array like
        :math:`dx` Earth velocity component

    dy : float, array like
        :math:`dy` Earth velocity component

    Returns
    -------

    """
    return np.pow(dx,2) + np.pow(dy,2)

def EarthSpeedSq(dx, dy, dz):
    r"""

    Parameters
    ----------
    dx : float, array like
        :math:`dx` Earth velocity component

    dy : float, array like
        :math:`dy` Earth velocity component

    dz : float, array like
        :math:`dz` Earth velocity component

    Returns
    -------

    """

    return np.pow(dx, 2) + np.pow(dy, 2) + np.pow(dz, 2)

def EarthSpeedFlat(dx, dy):
    r"""

    Parameters
    ----------
    dx :

    dy :
    Returns
    -------

    """

    return np.sqrt(np.pow(dx, 2) + np.pow(dy, 2))

