"""
    So this is a data fusion example including DSTL track fusion and chernoff updater

    No in reality better focus on comparison between KF and PF in the contest of data fusion

    what needs to be done?
    add two sensors with the same capabilities and track multitarget objects

    bearing range radars

    general setup
    1) create the ground truth simulator

"""

# General imports
import numpy as np
from datetime import datetime
from copy import deepcopy

# Stone soup imports
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import State
from stonesoup.platform.base import FixedPlatform


# Simulation parameters setup
start_time = datetime.now()
number_of_steps = 50
np.random.seed(1908)
birth_rate = 0.1   # 10% of birth rate
death_prob = 0.01  # low death probability

# %%
# 1) general setup of the problem and simulation
# ----------------------------------------------
# Describe the tutorial generation

# Load the transition model for the ground truths
truth_transition_model = CombinedLinearGaussianTransitionModel((ConstantVelocity(0.5),
                                                                ConstantVelocity(0.5)))

# Define the initial state of the target
initial_state = GaussianState([1,0,1,0],
                              np.diag([10,1,10,1]),
                              timestamp=start_time)

# initiate the groundtruth simulator
ground_truth_simulator = MultiTargetGroundTruthSimulator(
    transition_model= truth_transition_model,
    initial_state= initial_state,
    birth_rate= birth_rate,
    death_probability= death_prob,
    number_steps= number_of_steps,
    preexisting_states= [[-1,0,-1,0],[0,0,0,0]]
)

# define a clutter model
from stonesoup.models.clutter.clutter import ClutterModel

clutter_model = ClutterModel(
    clutter_rate=1.0,
    distribution=np.random.default_rng().uniform,
    dist_params=((-250,250), (-250,250))
)

# now create two radars - one bearing range and one with elevation maybe but before we can use the same
# and see how they behave
from stonesoup.sensor.radar.radar import RadarBearingRange
from stonesoup.sensor.radar.radar import RadarElevationBearingRange


# Let's assume that both radars have the same noise covariance for simplicity
radar_noise = CovarianceMatrix(np.diag([np.deg2rad(0.005), 0.05]))  # +/- 0.005 degrees and 0.05 meter range

# First radar
radar1 = RadarBearingRange(
    ndim_state= 4,
    position_mapping= (0, 2),
    noise_covar= radar_noise,
    clutter_model= clutter_model,
    max_range= 3000
)

# deep copy the first radar so changes in the first one does not influence the second
radar2 = deepcopy(radar1)

# Place the first sensor on each platform
sensor1_platform= FixedPlatform(
    states=GaussianState([-50,0,-50,0], np.diag([1,0,1,0])),
    position_mapping= (0, 2),
    sensors= [radar1]
)

# Place the second sensor in different parts
sensor2_platform= FixedPlatform(
states=GaussianState([50,0,50,0], np.diag([1,0,1,0])),
    position_mapping= (0, 2),
    sensors= [radar1]
)

# load the platform detection simulator
from stonesoup.simulator.platform import PlatformDetectionSimulator
radar_simulator = PlatformDetectionSimulator(
    groundtruth= ground_truth_simulator,
    platforms=[sensor1_platform, sensor2_platform]
)


# from stonesoup.plotter import Plotter, Plotterly
# #
# # Lists to hold the detections from each sensor and the path of the airborne radar
# s1_detections = []
# s2_detections = []
# #
# # # Extract the generator function from a copy of the simulator
# sim = deepcopy(radar_simulator)
# g = sim.detections_gen()
# #
# # Iterate over the time steps, extracting the detections, truths, and airborne sensor path
# for _ in range(number_of_steps):
#     s1_detections.append(next(g)[1])
#     s2_detections.append(next(g)[1])
# #    radar1_path.append(sim.platforms[0].position)
# truths = set(sim.groundtruth.groundtruth_paths)
#
# # Plot the truths and detections
# plotter = Plotterly()
# plotter.plot_ground_truths(truths, [0, 2])
# plotter.plot_measurements(s1_detections, [0, 2], marker= dict(color='yellow', symbol='305'))
# plotter.plot_measurements(s2_detections, [0, 2], marker= dict(color='orange', symbol='0'))
# plotter.fig.show()
# sys.exit()

# for now skip the plotting - the plots looks reasonable, even if in this case we are considering the
# multi-target case, maybe it is worth scale it down to 2/3 objects with known trajectory and merge
# and fuse the data accordingly

# Define the clutter area and spatial density for the tracker
clutter_area = np.prod(np.diff(clutter_model.dist_params))
clutter_spatial_density = clutter_model.clutter_rate/clutter_area  # define the clutter spatial density

# ok in the other case they use a probability approach, better see if I can do it with distance

# Distance approach hypothesiser
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment

from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.tracker.simple import MultiTargetTracker

# Define the Kalman updater
kalman_updater = ExtendedKalmanUpdater(measurement_model=None)

# Define the Kalman predictor
kalman_predictor = ExtendedKalmanPredictor(truth_transition_model)


# hypothesiser
hypothesiser = DistanceHypothesiser(
    predictor= kalman_predictor,
    updater= kalman_updater,
    #clutter_spatial_density=clutter_spatial_density,
    #prob_detect= 0.9,
    measure=Mahalanobis(),
    missed_distance=50
)

# This is a covariance based deleter
deleter = CovarianceBasedDeleter(covar_trace_thresh= 500)

data_associator= GNNWith2DAssignment(hypothesiser)

# Define a measurement initiator
initiator= MultiMeasurementInitiator(
    prior_state= initial_state,
    measurement_model= None,
    deleter=deleter,
    data_associator=data_associator,
    updater=kalman_updater,
    min_points=2
)

distance_tracker= MultiTargetTracker(
    initiator=initiator,
    deleter=deleter,
    detector=radar_simulator,
    data_associator=data_associator,
    updater=kalman_updater
)

groundtruth = set()
detections = set()
tracks = set()

for time, ctracks in distance_tracker:
    groundtruth.update(ground_truth_simulator.groundtruth_paths)
    detections.update(radar_simulator.detections)
    tracks.update(ctracks)

from stonesoup.plotter import Plotterly

plotter = Plotterly()

plotter.plot_ground_truths(groundtruth, mapping=[0, 2])
plotter.plot_measurements(detections, mapping=[0, 2])
#plotter.plot_tracks(tracks, mapping=[0, 2])
plotter.fig.show()
