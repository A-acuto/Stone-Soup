#!/usr/bin/env python
# coding: utf-8

"""
==========================================================
Comparing different filters on navigation transition model
==========================================================
"""

# This example compares the performances of various filters
# in tracking objects with navigation-like transition model and
# measurements models.
# We are interested in this scenario to show how
# navigation models can be implemented in Stone Soup and
# how the various filters (different Kalman flavours) and
# particle filters performs in tracking the target.
#
# In an different example we have explained the
# measurement components coming from
# :class:`~.AccelerometerGyroscopeMeasurementModel` in
# Stone soup. This example will do a 1-to-1 comparison
# using Extended Kalman filter (EKF), Unscented Kalman Filter
# (UKF) and Particle filter (PF) in this single target scenario.
#
# This example follows this schema:
# 1) instantiate the target ground truths and run it;
# 2) prepare and load the various filters components;
# 3) run the trackers and obtain the tracks;
# 4) run a metric manager and compare the track
# performances.


# %%
# General imports
# ^^^^^^^^^^^^^^^

import numpy as np
from datetime import datetime, timedelta
from scipy.stats import multivariate_normal

# %%
# Stone Soup imports
# ^^^^^^^^^^^^^^^^^^

from stonesoup.models.transition.linear import CombinedGaussianTransitionModel, \
    ConstantAcceleration, ConstantVelocity, CombinedLinearGaussianTransitionModel
from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.functions.navigation import getEulersAngles


# Simulation parameters
np.random.seed(1908) # fix a random seed
simulation_steps = 100
timesteps = np.linspace(1, simulation_steps+1, simulation_steps+1)
start_time = datetime.now().replace(microsecond=0)
# Lets assume a sensor with these specifics
radius = 5000
speed = 200
center = np.array([0, 0, 1000]) # latitude, longitude, altitutde (meters)

# %%
# 1) Instantiate the target ground truth path
# -------------------------------------------
# For this example we consider a different
# approach for describing the ground truth.
# We evaluate the target motion on a circular
# trajectory by modelling by hand all the
# dynamics and the rotational angles.

from stonesoup.types.detection import TrueDetection

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

# In combined Gaussian transition model we put the various models
transition_model = CombinedLinearGaussianTransitionModel([ConstantAcceleration(1.5),
                                     ConstantAcceleration(1.5),
                                     ConstantAcceleration(1.5),
                                     ConstantVelocity(0.5),
                                     ConstantVelocity(0.5),
                                     ConstantVelocity(0.5)
                                     ])
transition_model_pf = CombinedLinearGaussianTransitionModel([ConstantAcceleration(0.5),
                                     ConstantAcceleration(0.5),
                                     ConstantAcceleration(0.5),
                                     ConstantVelocity(0.5),
                                     ConstantVelocity(0.5),
                                     ConstantVelocity(0.5)
                                     ])

# %%
# Load the measurement model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now we can collect the ground truths and the
# measurements

from stonesoup.models.measurement.nonlinear import AccelerometerGyroscopeMeasurementModel

# Instantiate the measurement model
covariance_diagonal = np.repeat(100, 15)
reference_frame = StateVector([55, 0, 0])  # Latitude, longitude, Altitude
meas_model = AccelerometerGyroscopeMeasurementModel(
    ndim_state=15,
    mapping=(0, 3, 6),
    noise_covar=np.diag(covariance_diagonal)**2,
    reference_frame=reference_frame)

timestamps, groundtruths = describe_target_motion(speed, radius, center, start_time,
                                                  timesteps)
measurements_set = []

# Now create the measurements
for truth in groundtruths:
    measurement = meas_model.function(truth, noise=True)
    measurements_set.append(Detection(state_vector=measurement,
                                      timestamp=truth.timestamp,
                                      measurement_model=meas_model))


# %%
# 2) prepare and load the various filters components;
# ---------------------------------------------------
# So far we have generate the target track and the
# measurements measured using the accelerometer. Now
# We can load the various filter components for the
# EKF, UKF and PF. We need to instantiate the
# components as the prior and the tracks.

# Load the Kalman components
from stonesoup.updater.kalman import UnscentedKalmanUpdater, ExtendedKalmanUpdater, \
    KalmanUpdater
from stonesoup.predictor.kalman import UnscentedKalmanPredictor, ExtendedKalmanPredictor, \
    KalmanPredictor

from stonesoup.updater.particle import ParticleUpdater
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.resampler.particle import ESSResampler, ResidualResampler, MultinomialResampler, SystematicResampler

# Kalman filter
KF_predictor = KalmanPredictor(transition_model)
KF_updater = KalmanUpdater(measurement_model=None)

# Extended Kalman filter
EKF_predictor = ExtendedKalmanPredictor(transition_model)
EFK_updater = ExtendedKalmanUpdater(measurement_model=None)

# Unscented Kalman filter
UKF_predictor = UnscentedKalmanPredictor(transition_model)
UKF_updater = UnscentedKalmanUpdater(measurement_model=None)

# Particle filter
PF_predictor = ParticlePredictor(transition_model_pf)
#subresampler = ResidualResampler(residual_method='systematic')
resampler = SystematicResampler() #(resampler=subresampler)
PF_updater = ParticleUpdater(measurement_model=None,
                             resampler=resampler)

# Create a starting covarinace
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

from stonesoup.types.state import ParticleState
from stonesoup.types.numeric import Probability
from stonesoup.types.particle import Particle

number_particles=500

samples = multivariate_normal.rvs(np.array(prior.state_vector).reshape(-1),
                                  np.diag(covar_starting_position),
                                  size=number_particles)

particles = [Particle(sample.reshape(-1, 1),
                      weight= Probability(1/number_particles))
             for sample in samples]

# Particle prior
particle_prior = ParticleState(state_vector=None,
                               particle_list=particles,
                               timestamp=timestamps[0])


from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis

track_ukf, track_ekf, track_kf, track_pf = Track(), Track(), Track(), Track()

# Loop over the measurement
updaters = [UKF_updater, EFK_updater, UKF_updater, PF_updater]
predictors = [UKF_predictor, EKF_predictor, KF_predictor, PF_predictor]
tracks = [track_ukf, track_ekf, track_kf, track_pf]
priors = [prior, prior, prior, particle_prior]

# %%
# 3) run the trackers and obtain the tracks;
# ------------------------------------------
# Now we are ready to run the various trackers
# and generate some tracks.
# Then we store the various objects into
# a metric manager to evaluate the tracking
# performances.

# Loop over the various trackers
for predictor, updater, tracks, prior in zip(predictors, updaters, tracks, priors):
    for k, measurement in enumerate(measurements_set):
        print(k)
        predictions = predictor.predict(prior, timestamp=measurement.timestamp)
        hyps = SingleHypothesis(predictions, measurement)
        post = updater.update(hyps)
        tracks.append(post)
        prior = tracks[-1]

from stonesoup.plotter import Plotterly
plotter = Plotterly()

plotter.plot_ground_truths(groundtruths, mapping=[0, 3])
plotter.plot_tracks(track_ukf, mapping=[0, 3], track_label='UKF')
plotter.plot_tracks(track_ekf, mapping=[0, 3], track_label='EKF')
plotter.plot_tracks(track_kf, mapping=[0, 3], track_label='KF')
plotter.plot_tracks(track_pf, mapping=[0, 3], track_label='PF')
plotter.plot_measurements(measurements_set, mapping=[0, 3])
plotter.fig.show()


from stonesoup.metricgenerator.basicmetrics import BasicMetrics

# load the metrics for the Kalman and Particle filter
basic_UKF = BasicMetrics(generator_name='Unscented Kalman Filter', tracks_key='UKF_tracks',
                        truths_key='truths')
basic_EKF = BasicMetrics(generator_name='Extended Kalman Filter', tracks_key='EKF_tracks',
                         truths_key='truths')
basic_KF = BasicMetrics(generator_name='Kalman Filter', tracks_key='KF_tracks',
                        truths_key='truths')
basic_PF = BasicMetrics(generator_name='Particle Filter', tracks_key='PF_tracks',
                        truths_key='truths')

# Load the OSPA metric managers
from stonesoup.metricgenerator.ospametric import OSPAMetric
ospa_UKF_truth = OSPAMetric(c=40, p=1, generator_name='OSPA_UKF_truths',
                           tracks_key='UKF_tracks',  truths_key='truths')
ospa_EKF_truth = OSPAMetric(c=40, p=1, generator_name='OSPA_EKF_truths',
                            tracks_key='EKF_tracks',  truths_key='truths')
ospa_KF_truth =  OSPAMetric(c=40, p=1, generator_name='OSPA_KF_truths',
                            tracks_key='KF_tracks',  truths_key='truths')
ospa_PF_truth = OSPAMetric(c=40, p=1, generator_name='OSPA_PF_truths',
                           tracks_key='PF_tracks',  truths_key='truths')

# Define a data associator between the tracks and the truths
from stonesoup.dataassociator.tracktotrack import TrackToTruth
associator = TrackToTruth(association_threshold=30)

from stonesoup.metricgenerator.manager import MultiManager
metric_manager = MultiManager([basic_UKF,
                               basic_EKF,
                               basic_KF,
                               basic_PF,
                               ospa_UKF_truth,
                               ospa_EKF_truth,
                               ospa_KF_truth,
                               ospa_PF_truth],
                              associator)

metric_manager.add_data({'UKF_tracks': track_ukf,
                         'PF_tracks': track_pf,
                         'EKF_tracks': track_ekf,
                         'KF_tracks': track_kf,
                         'truths': groundtruths
                         }, overwrite=False)
# %%
# 4) run a metric manager and compare the tracking performances
# -------------------------------------------------------------
# We have all the components and now we can measure how well the
# various filter perform.
#

# Loaded the plotter for the various metrics.
from stonesoup.plotter import MetricPlotter

metrics = metric_manager.generate_metrics()

graph = MetricPlotter()

graph.plot_metrics(metrics, generator_names=['OSPA_UKF_truths',
                                             'OSPA_EKF_truths',
                                             'OSPA_PF_truths',
                                             'OSPA_KF_truths'],
                   color=['green', 'blue', 'orange', 'violet'])
graph.axes[0].set(ylabel='OSPA metrics', title='OSPA distances over time')
graph.fig.show()
