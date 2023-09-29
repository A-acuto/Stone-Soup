# %%
# Comparing different filters on navigation transition models
# ===========================================================
#
# This example compares the performances of various filters
# in tracking objects with navigation-like transition models.
# We are interested in this scenario to show how
# navigation models can be implemented in Stone Soup and
# how the various filters (different Kalman flavours) and
# particle filters performs in tracking the target(s).
#
# Insert here the explanation of the transition model
#
#
# This example follows this schema:
# 1) instantiate the target ground truths
# 2) prepare and load the various filters components
# 3) run the simulation and collect measurements
# 4) run the trackers and plot the relevant statistics.

# %%
# 1) Instantiate the target ground truth path
# -------------------------------------------
#
# Here we show how to start the various simulation
# components

# general imports
import numpy as np
import datetime
from datetime import datetime

# Load various stone soup components

np.random.seed(1908) # fix a random seed
simulation_steps = 35
import sys
from stonesoup.models.transition.linear import CombinedGaussianTransitionModel, ConstantAcceleration

# In combined Gaussian transition model we put the various models
transition_model = CombinedGaussianTransitionModel([ConstantAcceleration(0.01),
                                                    ConstantAcceleration(0.01),
                                                    ConstantAcceleration(0.01),])
                                                    # HERE SHOULD GO THE ROTATION BIT])

print(transition_model)
sys.exit()
# %%
# 2) Prepare the various filters components
# -----------------------------------------
#
# We have instantiate the ground truth using
# the relevant transition model and
# measurement model for navigation purposes
# we have also gathered information using the
# sensor. Now we can prepare the various
# trackers components, we consider an
# extended kalman filter, an
# unscented kalman filter and a
# particle filter

# Load the Kalman components
from stonesoup.updater.kalman import UnscentedKalmanUpdater, ExtendedKalmanUpdater
from stonesoup.predictor.kalman import UnscentedKalmanPredictor, ExtendedKalmanPredictor

from stonesoup.updater.particle import ParticleUpdater
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.resampler.particle import ESSResampler

EKF_predictor = ExtendedKalmanPredictor()
UKF_predictor = UnscentedKalmanPredictor()

EFK_updater = ExtendedKalmanUpdater(measurement_model=None)
UKF_updater = UnscentedKalmanUpdater(measurement_model=None)

PF_predictor = ParticlePredictor()
resampler = ESSResampler()
PF_updater = ParticleUpdater(measurement_model=None,
                             resampler=resampler)


