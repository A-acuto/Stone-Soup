from abc import abstractmethod
from typing import Union

import numpy as np

from stonesoup.base import Base, Property
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition import TransitionModel
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.detection import Detection
from stonesoup.types.numeric import Probability
from stonesoup.types.state import State


class Proposal(Base):

    @abstractmethod
    def rvs(self, *args, **kwargs):
        r"""Proposal noise/sample generation function

        Generates samples from the proposal.

        Returns
        -------
         : 2-D array of shape (:attr:`ndim`, ``num_samples``)
            A set of Np samples, generated from the model's noise
            distribution.
        """
        raise NotImplementedError
