"""
So lets do a data fusion example- use a standard radar and a
flying UAV considering a case of two targets moving in a simpler manner
and then we do a proper measurement fusion and simpler tracking using
a kalman and particle filter.

"""

# %%%
# Data fusion: comparison between Kalman and particle filters
# -----------------------------------------------------------
#
# In this example we present the case of data fusion, in
# detail measurement fusion, from two sensors in the context
# of multi-target tracking and we compare the performances
# of two separate filters, Kalman and Particle.
#
# The example layout is as follows:
# 1) define the targets trajectories and the sensors
# 2) define the various filter components and build the trackers
# 3) Run the measurement fusion algorithm and run the tracker
# 4) Plot the tracks and the track performances
#

# %%
# 1) Define the targets trajectories and the sensors
# --------------------------------------------------
# Let's define the targets trajectories, assuming a simpler case
# of a straight movement and the sensor specifics. We consider a
# class radar elevation bearing range measuring from an altitude position
# and moving on a trajectory a radar in the same cartesian space as the
# targets. The targets follow a simple trajectory.
#

# General imports
import numpy as np
import datetime
from datetime import datetime
from datetime import timedelta

# Stone Soup general imports
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.state import GaussianState
from stonesoup.types.array import CovarianceMatrix
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator

# Simulation parameters setup
start_time = datetime(2023, 8, 1, 10,0,0) # For simplicity fix a date
number_of_steps = 25  # Number of timestep for the simulation
np.random.seed(1908)  # Random seed for reproducibility

# Instantiate the target transition model 3D case
transition_model = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(0.00), ConstantVelocity(0.00)]) #, ConstantVelocity(0.00)])

# Define the initial target state
initial_target_state_1 = GaussianState([25, 1, 50, -0.5 ], #10., 0.
                                     np.diag([1, 0.1, 1, 0.1]) ** 2, #,0.1,0.1
                                     timestamp=start_time)
# Define the initial target state
initial_target_state_2 = GaussianState([25, 1, -50, 0.5],  #, 10, 0.
                                     np.diag([1, 0.1, 1, 0.1]) ** 2, #,0.1,0.1
                                     timestamp=start_time)

# Create a ground truth simulator, specify the number of initial targets as
# 0 so no new targets will be created a side from the two provided
ground_truth_simulator = MultiTargetGroundTruthSimulator(
    transition_model= transition_model,
    initial_state= GaussianState([10,1, 0,0.5],  #, 10, 0.0
                                 np.diag([5, 0.1, 5, 0.1]),  #,1,1
                                 timestamp= start_time),
    birth_rate= 0.0,
    death_probability= 0.0,
    number_steps= number_of_steps,
    preexisting_states= [initial_target_state_1.state_vector,
                         initial_target_state_2.state_vector],
    initial_number_targets= 0)

# Load a clutter model
from stonesoup.models.clutter.clutter import ClutterModel

# Define the clutter model, this will be the same for both sensors
clutter_model = ClutterModel(
    clutter_rate=1.2,
    distribution=np.random.default_rng().uniform,
    dist_params=((0,150), (-105,105))) #, (0, 15)))

# Define the clutter area and spatial density for the tracker
clutter_area = np.prod(np.diff(clutter_model.dist_params))
clutter_spatial_density = clutter_model.clutter_rate/clutter_area

# Instantiate the radars to collect measurements - Use a BearingRange radar
from stonesoup.sensor.radar.radar import RadarElevationBearingRange, RadarBearingRange

# Let's assume that both radars have the same noise covariance for simplicity
# These radars will have the +/-0.005 degrees accuracy in bearing and 2 meters in range
radar_noise = CovarianceMatrix(np.diag([np.deg2rad(0.005), 0.05]))  # , np.deg2rad(0.005)


# Define the specifications of the two radars
radar1 = RadarBearingRange(
    ndim_state= 4,
    position_mapping= (0,2), ##(0, 2, 4),
    noise_covar= radar_noise,
    clutter_model= clutter_model,
    max_range= 3000)

radar2 = RadarBearingRange( #RadarElevationBearingRange(
    ndim_state= 4,
    position_mapping= (0, 2),#(0, 2, 4),
    noise_covar= radar_noise,
    clutter_model= clutter_model,
    max_range= 3000)

# Import the platform to place the sensors on
from stonesoup.platform.base import FixedPlatform
from stonesoup.platform.base import MovingPlatform

# Instantiate the first sensor platform and add the sensor
sensor1_platform = FixedPlatform(
    states=GaussianState([10, 0, 5, 0],#, 10, 0
                         np.diag([1, 0, 1, 0, ])), #1, 0
    position_mapping= (0, 2), #, 4),
    sensors= [radar1])

sensor2_platform = MovingPlatform(
    states=GaussianState([120, 0,-50, 1.5], #, 50, 0],
    np.diag([1, 0, 5, 1])), #, 1, 0
    position_mapping= (0, 2),
    velocity_mapping=(0, 2),
    transition_model=transition_model,
    sensors=[radar2])


# Load the platform detection simulator - Let's use a simulator for each track
# Instantiate the simulators
from stonesoup.simulator.platform import PlatformDetectionSimulator

radar_simulator1 = PlatformDetectionSimulator(
    groundtruth= ground_truth_simulator,
    platforms= [sensor1_platform])

radar_simulator2 = PlatformDetectionSimulator(
    groundtruth= ground_truth_simulator,
    platforms = [sensor2_platform]
)

# Load the stone soup plotter
from stonesoup.plotter import Plotterly, Plotter, Dimension

# Lists to hold the detections from each sensor
# s1_detections = []
# s2_detections = []
# radar_path = []
# g1 = radar_simulator1.detections_gen()
# g2 = radar_simulator2.detections_gen()
#
# for _ in range(number_of_steps):
#     s1_detections.append(next(g1)[1])
#     s2_detections.append(next(g2)[1])
#     radar_path.append(radar_simulator2.platforms[0].position)
# # Generate the ground truth
# truths = set(ground_truth_simulator.groundtruth_paths)

# Plot the groundtruth and detections from the two radars
#plotter = Plotter(dimension=Dimension.THREE)
# plotter = Plotterly()
# plotter.plot_ground_truths(truths, [0, 2, 4])
# plotter.plot_measurements(s1_detections, [0, 2, 4])
# plotter.plot_measurements(s2_detections, [0, 2, 4])
# plotter.plot_sensors({sensor1_platform ,sensor2_platform}, [0, 1, 2]) # improve the plotting
#
# # plotter.plot_measurements(s1_detections, [0, 2, 4], marker= dict(color='orange', symbol='305'),
# #                            measurements_label='Sensor 1 measurements')
# # plotter.plot_measurements(s2_detections, [0, 2], marker= dict(color='blue', symbol='0'),
# #                           measurements_label='Sensor 2 measurements')
# # plotter.plot_sensors({sensor1_platform,sensor2_platform}, [0,1], marker= dict(color='black', symbol= '1',
# #                                                                               size=10))
# plotter.fig.show()

# %%
# 2) define the various filter components and build the trackers
# --------------------------------------------------------------
# We have presented the scenario with two separate targets moving
# and two sensors collecting the measurements. Now, we focus on
# building the various tracker components: we use a :class:`~.DistanceHypothesiser`
# hypothesiser using :class:.`Mahalanobis` distance measure.
# We consider an Extended Kalman filter and a
# particle filter.
#

# We use a Distance hypothesiser
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment, GlobalNearestNeighbour
from stonesoup.deleter.time import UpdateTimeStepsDeleter, UpdateTimeDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator

# Load the Kalman filter components
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.predictor.kalman import ExtendedKalmanPredictor

# prepare the particle components
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.resampler.particle import ESSResampler

from stonesoup.initiator.simple import SimpleMeasurementInitiator, GaussianParticleInitiator

# Design the components
# load the Kalman filter predictor
KF_predictor = ExtendedKalmanPredictor(transition_model)

# load the Kalman filter updater
KF_updater = ExtendedKalmanUpdater(measurement_model= None)

# define the hypothesiser
hypothesiser_KF = DistanceHypothesiser(
    predictor=KF_predictor,
    updater= KF_updater,
    measure= Mahalanobis(),
    missed_distance= 5)

# define the distance data associator
data_associator_KF = GNNWith2DAssignment(hypothesiser_KF)

# define a track deleter based on time measurements
deleter = UpdateTimeDeleter(timedelta(seconds=3), delete_last_pred= True)

# create an track initiator placed on the target tracks origin
KF_initiator = MultiMeasurementInitiator(
    prior_state=GaussianState([10,0,10,0], #,10,0
                              np.diag([1,1,1,1])), #,1,0
    measurement_model=None,
    deleter= deleter,
    updater= KF_updater,
    data_associator= data_associator_KF)

# Instantiate the predictor, particle resampler and particle
# filter updater
PF_predictor = ParticlePredictor(transition_model)
resampler = ESSResampler()
PF_updater = ParticleUpdater(measurement_model= None,
                             resampler= resampler)

hypothesiser_PF = DistanceHypothesiser(
    predictor= PF_predictor,
    updater= PF_updater,
    measure= Mahalanobis(),
    missed_distance= 5)

# define the data associator
data_associator_PF = GNNWith2DAssignment(hypothesiser_PF)

# To instantiate the track initiator we define a prior state
# as gaussian state with the target track origin
initiator_particles= SimpleMeasurementInitiator(
    prior_state=GaussianState([10, 0, 10,0], # , 10, 0
                 np.diag([5, 0.1, 5, 0.1]) ** 2), #,0.1,0.1
    measurement_model= None,
    skip_non_reversible= True)

# Particle filter initiator
PF_initiator = GaussianParticleInitiator(
    initiator= initiator_particles,
    number_particles= 1000)

# Load the multitarget tracker
from stonesoup.tracker.simple import MultiTargetTracker

# Load a detection reader
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.reader.base import DetectionReader

class DummyDetector(DetectionReader):
    def __init__(self, *args, **kwargs):
        self.current = kwargs['current']

    @BufferedGenerator.generator_method
    def detections_gen(self):
        yield self.current

# Instantiate the Kalman Tracker, without
# specifying the detector
KF_tracker = MultiTargetTracker(
    initiator= KF_initiator,
    deleter=deleter,
    data_associator= data_associator_KF,
    updater= KF_updater,
    detector= None)

# Instantiate the Particle filter as well
PF_tracker = MultiTargetTracker(
    initiator= PF_initiator,
    deleter= deleter,
    data_associator= data_associator_PF,
    updater= PF_updater,
    detector= None)

# %%
# 3) Run the measurement fusion algorithm and the trackers
# --------------------------------------------------------
# We have instantiated all the relevant components for the two
# filters and now we can run the simulation to generate the
# various detections, clutter and associations.
# The final tracks will be passed onto a metric generator
# plotter. We start composing the various
# metrics statistics available.

# Load the metric manager
from stonesoup.metricgenerator.basicmetrics import BasicMetrics

basic_KF = BasicMetrics(generator_name='Kalman Filter', tracks_key='KF_tracks', truths_key='truths')
basic_PF = BasicMetrics(generator_name='Particle Filter', tracks_key='PF_tracks', truths_key='truths')

# Load the OSPA metric managers
from stonesoup.metricgenerator.ospametric import OSPAMetric
ospa_KF_truth = OSPAMetric(c=40, p=1, generator_name='OSPA_KF_truths',
                           tracks_key= 'KF_tracks',  truths_key='truths')
ospa_PF_truth = OSPAMetric(c=40, p=1, generator_name='OSPA_PF_truths',
                           tracks_key= 'PF_tracks',  truths_key='truths')

# Define a data associator between the tracks and the truths
from stonesoup.dataassociator.tracktotrack import TrackToTruth
associator = TrackToTruth(association_threshold= 30)

from stonesoup.metricgenerator.manager import MultiManager
metric_manager = MultiManager([basic_KF,
                               basic_PF,
                               ospa_KF_truth,
                               ospa_PF_truth],
                              associator)

# define the Detections from the two sensors
s1_detections = []
s2_detections = []
radar_path = []
# instantiate the detections generator function from the simulators
g1 = radar_simulator1.detections_gen()
g2 = radar_simulator2.detections_gen()

# identify the tracks
kf_tracks = set()
pf_tracks = set()
truths = set()
full_detections = []

# loop over the various timesteps
for t in range(number_of_steps):
    print(t)
    detections_1 = next(g1)
    s1_detections.extend(detections_1[1])
    detections_2 = next(g2)
    s2_detections.extend(detections_2[1])
    full_detections.extend(detections_1[1])
    full_detections.extend(detections_2[1])

    for detections in [detections_1, detections_2]:

        # Run the Kalman tracker
        KF_tracker.detector = DummyDetector(current=detections)
        KF_tracker.__iter__()
        _, tracks = next(KF_tracker)
        kf_tracks.update(tracks)

        # Run the Particle Tracker
        PF_tracker.detector = DummyDetector(current=detections)
        PF_tracker.__iter__()
        _, tracks = next(PF_tracker)
        pf_tracks.update(tracks)

        metric_manager.add_data({'KF_tracks': kf_tracks}, overwrite=False)
        metric_manager.add_data({'PF_tracks': pf_tracks}, overwrite=False)

truths = set(ground_truth_simulator.groundtruth_paths)
metric_manager.add_data({'truths': truths,
                         'detections': full_detections}, overwrite=False)

# %%
# 4) Plot the tracks and the track performances
# ---------------------------------------------
# We have obtained the tracks and the ground truths
# from the various detectors and trackers. It is time
# to visualise the tracks and load the metric manager
# to evaluate the performances.

plotter = Plotterly()
plotter.plot_measurements(s1_detections, [0,2])
plotter.plot_measurements(s2_detections, [0,2])
plotter.plot_tracks(kf_tracks, [0,2,4], line= dict(color='black'), track_label='Kalman Filter')
plotter.plot_tracks(pf_tracks, [0,2], line= dict(color='red'), track_label='PF Filter')
plotter.plot_ground_truths(truths, [0,2])
plotter.fig.show()

# Loaded the plotter for the various metrics.
from stonesoup.plotter import MetricPlotter

metrics = metric_manager.generate_metrics()
graph = MetricPlotter()
graph.plot_metrics(metrics, generator_names=['OSPA_KF_truths',
                                             'OSPA_PF_truths'],
                   color=['green', 'orange'])
graph.axes[0].set(ylabel='OSPA metrics', title='OSPA distances over time')
graph.fig.show()


# %%
# This concludes this example where we have shown
# how to perform a measurement fusion using two
# radars and we have shown the performances
# of the tracks obtained by a Kalman and a particle
# filters.