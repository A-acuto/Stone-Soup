"""

Comparing various filters in no-data association scenarios
==========================================================

In this example we consider the complex case of running
tracker algorithms while on measurements involving no-
data associations, in particular coming from GPS or
navigation scenario.
In this example, we show how to set up various trackers
using differents filters as Unscented, extended kalman
filters as well as particle filters.

"""

# %%
# The example layout follows:
# 1) load the GPS data and visualise them
# 2) instantiate the various tracker components
# 3) run the tracker on the data
# 4) visualise the tracker results


# %%
# 1) load the GPS data and visualise them
# ---------------------------------------
# Somewhere find and generate the data as groundtruth
# in our simulation. Then from that we should be able to
# generate some detections

# general imports
import numpy as np
import datetime
from datetime import datetime
from datetime import timedelta

# General Stone soup transition model
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05), ConstantVelocity(0.05)])

# %%
# 2) instantiate the various tracker components
# ---------------------------------------------
# We have out detections and measurements now
# we want to set up the various tracker components.
# We consider a class of :class:`.ExtendedKalmanPredictor`
# :class:`.UnscentedKalmanPredictor` and
# :class:`.ParticlePredictor`

# Load the various components regarding the extended
# kalman filter
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater

predictor_EKF = ExtendedKalmanPredictor(transition_model)
updater_EKF = ExtendedKalmanUpdater(measurement_model=None)

# Load the components for the Unscented Kalman Filter
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.updater.kalman import UnscentedKalmanUpdater

predictor_UKF = UnscentedKalmanPredictor(transition_model)
updater_UKF = UnscentedKalmanUpdater(measurement_model=None)

# Load the components for the Particle filter
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.resampler.particle import SystematicResampler

resampler = SystematicResampler()
updater_PF = ParticleUpdater(measurement_model= None,
                             resampler=resampler)
predictor_PF = ParticlePredictor(transition_model)

# technically no data association in this case

# load a deleter
from stonesoup.deleter.time import UpdateTimeDeleter
deleter = UpdateTimeDeleter(timedelta(seconds=5), delete_last_pred= True)

# Load the various initiators
from stonesoup.initiator.simple import SimpleMeasurementInitiator, GaussianParticleInitiator
from stonesoup.types.state import GaussianState






