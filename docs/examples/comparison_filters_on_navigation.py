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

# %%
# General imports
# ^^^^^^^^^^^^^^^

import numpy as np
from datetime import datetime, timedelta
import sys

# %%
# Stone Soup imports
# ^^^^^^^^^^^^^^^^^^

from stonesoup.models.transition.linear import CombinedGaussianTransitionModel, \
    ConstantAcceleration, OrnsteinUhlenbeck
from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.models.measurement.nonlinear import NonLinearGaussianMeasurement
from stonesoup.functions.navigation import getEulersAngles

# Simulation parameters
np.random.seed(1908) # fix a random seed
simulation_steps = 35
start_time = datetime.now().replace(microsecond=0)


# %%
# Simulate ground truths
# ^^^^^^^^^^^^^^^^^^^^^^
# For this example we consider a different
# approach for describing the ground truth.
# We evaluate the target motion on a circular
# trajectory by modelling by hand all the
# dinamics.

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

    # Generate the groundTruthpat
    truths = GroundTruthPath([])

    # instantiate a list for the timestamps
    timestamps = []

    # loop over the timestep
    for i in number_of_timesteps:
        theta = target_speed * i / target_radius + 0

        # positions
        target_dynamics[[0, 3, 6]] += target_radius * \
                                      np.array([np.cos(theta), np.sin(theta), 0]) + \
                                      starting_position

        # velocities
        target_dynamics[[1, 4, 7]] += target_speed * \
                                      np.array([-np.sin(theta), np.cos(theta), 0])

        # acceleration
        target_dynamics[[2, 5, 8]] += ((-target_speed * target_speed) / target_radius) * \
                           np.array([np.cos(theta), np.sin(theta), 0])

        # Now using the velocity and accelerations get the Euler angles
        angles, dangles = getEulersAngles(target_dynamics[[1, 4, 7]],
                                          target_dynamics[[2, 5, 8]])

        # add the Euler angles and their time derivative
        # please check that are all angles
        target_dynamics[[9, 11, 13]] += angles
        target_dynamics[[10, 12, 14]] += dangles

        # append all those as ground state
        truths.append(GroundTruthState(state_vector=target_dynamics,
                                       timestamp=timedelta(seconds=int(i))))
        # restart the array
        target_dynamics = np.zeros((15))
        timestamps.append(start_time + timedelta(seconds=int(i)))

    return (timestamps, truths)

# Lets assume a sensor with these specifics
radius = 5000
speed = 200
center = np.array([0, 0, 1000])

timestamps, groundtruths = describe_target_motion(speed, radius, center, start_time, np.arange(simulation_steps))

from stonesoup.plotter import Plotterly
plotter = Plotterly()

plotter.plot_ground_truths(groundtruths, [0, 3])
plotter.fig.show()

sys.exit()


# In combined Gaussian transition model we put the various models
transition_model = CombinedGaussianTransitionModel([OrnsteinUhlenbeck(0.1, 1e-4),
                                                    OrnsteinUhlenbeck(0.1, 1e-4),
                                                    OrnsteinUhlenbeck(0.1, 1e-4)])
                                                    # HERE SHOULD GO THE ROTATION BIT])

# measurement_model = NonLinearGaussianMeasurement(
#     ndim_state=6,
#     mapping=(0, 2, 4),
#     noise_covar=np.array(np.diag([1, 0, 1, 0, 1, 0]))
# )

print(transition_model)
# print(measurement_model)
timesteps = [start_time]

init_target_state = GaussianState(
    state_vector=StateVector(np.array([0, 1, 0, 1, 0, 1])),
    covar=CovarianceMatrix(np.diag([1, 1, 1, 1, 1, 1])),
    timestamp=timesteps[0])


truth = GroundTruthPath([init_target_state])


for k in range(1, simulation_steps + 1):

    timesteps.append(start_time+timedelta(seconds=k))  # add next timestep to list of timesteps
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=timesteps[k]))


for k in range(1, simulation_steps + 1):
    timesteps.append(start_time + timedelta(seconds=k))

# # Simulate measurements
# # =====================
# scans = []
#
# for k in range(simulation_steps):
#     measurement_set = set()
#
#     # True detections
#     for truth in truths:
#         # Generate actual detection from the state with a 10% chance that no detection is received.
#         if np.random.rand() <= prob_detect:
#             measurement = measurement_model.function(truth[k], noise=True)
#             measurement_set.add(TrueDetection(state_vector=measurement,
#                                               groundtruth_path=truth,
#                                               timestamp=truth[k].timestamp,
#                                               measurement_model=measurement_model))
#
#         # Generate clutter at this time-step
#         truth_x = truth[k].state_vector[0]
#         truth_y = truth[k].state_vector[2]
#
#     # Clutter detections
#     for _ in range(np.random.poisson(clutter_rate)):
#         x = uniform.rvs(-10, 30)
#         y = uniform.rvs(0, 25)
#         measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=truth[k].timestamp,
#                                     measurement_model=measurement_model))
#     scans.append((timestamps[k], measurement_set))
#
#
# class SimpleDetector(DetectionReader):
#     @BufferedGenerator.generator_method
#     def detections_gen(self):
#         for timestamp, measurement_set in scans:
#             yield timestamp, measurement_set

from stonesoup.plotter import Plotterly
plotter = Plotterly()

plotter.plot_ground_truths(truth, [0, 2])
plotter.fig.show()

sys.exit()

# for the measurements consider a scan



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



