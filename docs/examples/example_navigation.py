#!/usr/bin/env python
# coding: utf-8

"""
==========================================
Example using navigation measurement model
==========================================
"""

# %%
# In this example, we present how to perform the
# target tracking task in the context of
# navigation measurement model. We simulate an
# three dimensional target, moving in 3D and with
# Euler angles describing the target rotation and
# orientation during the flight. This example aims to
# provide an idea of how to use the :class:`~.AccelerometerGyroscopeMeasurementModel`
# to calculate the accelerations and angles rotation
# of the target. This measurement model combines a
# 3D :class:`~.ConstantAcceleration` transition model for
# the trajectory and a 3D :class`~.ConstantVelocity` transition
# model to describe the target internal movements.
# Overall this model has a 15 dimensions. The Euler angles
# are the heading (:math:`\psi`), the pitch (:math:`\theta`)
# and the roll (:math:`\phi`).
#
# This example follows this points:
# 1) describe the transition model;
# 2) obtain the ground truth and measurements;
# 3) instantiate the tracker components;
# 4) run the tracker and obtain the final track.
#

# %%
# 1) Describe the transition model
# --------------------------------
# As we have previously said we want a 15 dimensions
# transition model, in the simplest form we can combine
# :class:`~.ConstantAcceleration` and :class:`~.ConstantVelocity`
# transition model. A more complex approach could involve using
# the Ornstein-Uhlenbeck model which revers the angle rates to
# adjustments into the direction of flight. However, this
# implementation is not yet available in Stone Soup.
#

# %%
# General imports
# ^^^^^^^^^^^^^^^
import numpy as np
from datetime import datetime, timedelta

# %%
# Stone Soup and transition models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
from stonesoup.types.detection import Detection
from stonesoup.types.state import State, StateVector, StateVectors, GaussianState
from stonesoup.models.transition.linear import CombinedGaussianTransitionModel, \
    ConstantVelocity, ConstantAcceleration
from stonesoup.functions.navigation import getEulersAngles

# %%
# Simulation parameters setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Lets assume a target sensor with these specifics
radius = 5000   # meters
speed = 200     # meters
center = np.array([0, 0, 1000])  # 3D center
n_timesteps = 100

timesteps = np.linspace(1, n_timesteps+1, n_timesteps+1)
simulation_start = datetime.now().replace(microsecond=0)
np.random.seed(1908) # fix a random seed

# %%
# Describe the ground truth
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# In this case lets create a ground truth
# considering all the relevant components.
#

# Create a function to create the groundtruth paths
def describe_target_motion(target_speed: float,
                           target_radius: float,
                           starting_position: np.array,
                           start_time: datetime,
                           number_of_timesteps: np.array
                           ) -> (list, set):

    """
        Auxuliary function to create the target dynamics in the
        specific case of circular motion.

    Parameters:
    -----------
    target_speed: float
        Speed of the target;
    target_tadius: float
        radius of the circular trajectory;
    starting_position: np.array
        starting point of the target, latitude, longitude
        and altitude;
    start_time: datetime,
        start of the simulation;
    number_of_timesteps: np.array
        simulation lenght

    Return:
    -------
    (list, set):
        list of timestamps of the simulation and
        groundtruths path.
    """

    # Instantiate the 15 dimension object describing
    # the positions, dynamics and angles of the target
    target_dynamics = np.zeros((15))

    # Generate the groundTruthpath
    truths = GroundTruthPath([])

    # instantiate a list for the timestamps
    timestamps = []

    # indexes of the array
    position_indexes = [0, 3, 6]
    velocity_indexes = [1, 4, 7]
    acceleration_indexes = [2, 5, 8]
    angles_indexes = [9, 11, 13]
    vangles_indexes = [10, 12, 14]

    # loop over the timestep
    for i in number_of_timesteps:
        theta = target_speed * i / target_radius + 0

        # positions
        target_dynamics[position_indexes] += target_radius * \
                                      np.array([np.cos(theta), np.sin(theta), 0]) + \
                                      starting_position

        # velocities
        target_dynamics[velocity_indexes] += target_speed * \
                                      np.array([-np.sin(theta), np.cos(theta), 0])

        # acceleration
        target_dynamics[acceleration_indexes] += ((-target_speed * target_speed) / target_radius) * \
                           np.array([np.cos(theta), np.sin(theta), 0])

        # Now using the velocity and accelerations get the Euler angles
        angles, dangles = getEulersAngles(target_dynamics[velocity_indexes],
                                          target_dynamics[acceleration_indexes])

        # add the Euler angles and their time derivative
        # please check that are all angles
        target_dynamics[angles_indexes] += angles
        target_dynamics[vangles_indexes] += dangles

        # append all those as ground state
        truths.append(GroundTruthState(state_vector=target_dynamics,
                                       timestamp=start_time +
                                                 timedelta(seconds=int(i))))
        # restart the array
        target_dynamics = np.zeros((15))
        timestamps.append(start_time + timedelta(seconds=int(i)))

    return (timestamps, truths)


# Instantiate the transition model
transition_model = CombinedGaussianTransitionModel([ConstantAcceleration(1.5),
                                     ConstantAcceleration(1.5),
                                     ConstantAcceleration(1.5),
                                     ConstantVelocity(0.5),
                                     ConstantVelocity(0.5),
                                     ConstantVelocity(0.5)
                                     ])

# %%
# 2) obtain the ground truth and measurements;
# --------------------------------------------
# We have created a function to describe the
# target dynamics following using the Euler angles
# obtained from the target acceleration and velocity
# :class:`~.getEulerAngles`. We have also instantiated
# the 15 dimension transition model using a constant
# acceleration model for the 3D dynamics and a
# constant velocity for modelling the Euler angles
# dynamics.
# Now we can start collecting both the grountruths and
# the measurement using the :class:`~.AccelerometerGyroscopeMeasurementModel`.
# This measurement model combines the specific force
# measured by the accelerometer instrument on to the
# target, accounting from the inertia movements.
# As well, this model measures the angular rotation
# values due to the target inertia movements.
#

# %%
# Get the groungtruth paths
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
timestamps, truths = describe_target_motion(speed,
                                            radius,
                                            center,
                                            simulation_start,
                                            timesteps)

# %%
# Load and instantiate the measurement model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This model requires a 15 dimension covariance matrix,
# a mapping of the x,y,z and a reference frame in
# latitude, longitude and altitude.
#
from stonesoup.models.measurement.nonlinear import AccelerometerGyroscopeMeasurementModel

# Instantiate the measurement model
covariance_diagonal = np.repeat(100, 15)
reference_frame = StateVector([55, 0, 0])  # Latitude, longitude, Altitude
meas_model = AccelerometerGyroscopeMeasurementModel(
    ndim_state=15,
    mapping=(0, 3, 6),
    noise_covar=np.diag(covariance_diagonal)**2,
    reference_frame=reference_frame)

measurement_set = []

# Now create the measurements
for truth in truths:
    measurement = meas_model.function(truth, noise=True)
    measurement_set.append(Detection(state_vector=measurement,
                                      timestamp=truth.timestamp,
                                      measurement_model=meas_model))


# %%
# 3) instantiate the tracker components;
# --------------------------------------
# We have the truths and the detections,
# in this simple example we do not
# include measurement clutter. Now we can
# prepare the various components of the tracker.
# In this example we consider an UnscentedKalmanFilter.

# %%
# Load the filter components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.updater.kalman import UnscentedKalmanUpdater

predictor = UnscentedKalmanPredictor(transition_model)
updater = UnscentedKalmanUpdater(None)

covar_starting_position = np.repeat(200, 15)

# Instantiate the prior, with a known location.
prior = GaussianState(
    state_vector=StateVector([[4.99051886e+03],
                                                   [-1.23107435e+01],
                                                   [ 7.98483018e+00],
                                                   [ 3.07768587e+02],
                                                   [1.99620754e+02],
                                                   [-4.92429739e-01],
                                                   [ 1.00000000e+03],
                                                   [ 0.00000000e+00],
                                                   [ 0.00000000e+00],
                                                   [-9.35289991e+01],
                                                   [ 2.29183118e+00],
                                                   [ 0.00000000e+00],
                                                   [ 0.00000000e+00],
                                                   [ 0.00000000e+00],
                                                   [ 0.00000000e+00]]),
    covar=np.diag(covar_starting_position),
    timestamp=timestamps[0]
)

# %%
# 4) run the tracker and obtain the final track.
# ----------------------------------------------
# We have the tracker components and the starting
# (prior) knowledge, now we can loop over the
# various measurements and using a
# :class:`~.SingleHypothesis` we can perform the
# tracking.
#

# Load these components to do the tracking
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis

track = Track()

# Loop over the measurement
for k, measurement in enumerate(measurement_set):
    predictions = predictor.predict(prior, timestamp=measurement.timestamp)
    hyps = SingleHypothesis(predictions, measurement)
    post = updater.update(hyps)
    track.append(post)
    prior = track[-1]

# %%
# Load the plotter
# ^^^^^^^^^^^^^^^^

from stonesoup.plotter import Plotterly

plotter = Plotterly()

plotter.plot_ground_truths(truths, mapping=[0, 3])
plotter.plot_measurements(measurement_set, mapping=[0, 3])
plotter.plot_tracks(track, mapping=[0, 3])
plotter.fig.show()

# %%
# Conclusion
# ----------
# In this example we have shown how it is
# possible to measure the trajectory of a
# 3D target using measurements from the
# accelerometers and rotation angles.
