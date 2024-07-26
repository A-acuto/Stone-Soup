#!/usr/bin/env python
# coding: utf-8


# %%
# General imports
# ^^^^^^^^^^^^^^^

import numpy as np
from datetime import datetime, timedelta
from copy import deepcopy
from time import perf_counter

# Load the pyehm plugins
from stonesoup.plugins.pyehm import JPDAWithEHM, JPDAWithEHM2

# %%
# Stone Soup Imports
#

from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity)

# %%
# Simulation parameters setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

np.random.seed(1908)  # set the random seed for the simulation
simulation_start_time = datetime.now().replace(microsecond=0)  # simulation start

# initial state of all targets
initial_state_mean = StateVector([0, 0, 0, 0])
initial_state_covariance = CovarianceMatrix(np.diag([5, 0.5, 5, 0.5]))
timestep_size = timedelta(seconds=1)
number_of_steps = 50 #   # number of time-steps (50 is default)
birth_rate = 0   # probability of new target to appear ( 0.25 is default)
death_probability = 0 #  probability of target to disappear (0.01 is default)

# setup the initial state of the simulation
initial_state = GaussianState(state_vector=initial_state_mean,
                              covar=initial_state_covariance,
                              timestamp=simulation_start_time)

# create the targets transition model
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.05), ConstantVelocity(0.05)])


Initial_preexisting_states=[[0, 0, 0, 0], [25, 0, 0, 0]]

# Put this all together in a multi-target simulator.
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
groundtruth_sim = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=initial_state,
    timestep=timestep_size,
    number_steps=number_of_steps,
    birth_rate=birth_rate, # 0 birth_rate
    death_probability=death_probability,
    preexisting_states=Initial_preexisting_states) #0 death_probability

# Load the measurement model
from stonesoup.models.measurement.linear import LinearGaussian

# initialise the measurement model
measurement_model_covariance = np.diag([1e-3, 1e-3])
measurement_model = LinearGaussian(4,
                                   [0, 2],
                                   measurement_model_covariance)

# probability of detection
probability_detection = 1.0 #0.99


# %%
# Generate clutter
# ^^^^^^^^^^^^^^^^

clutter_area = np.array([[-1, 1], [-1, 1]])*30
surveillance_area = ((clutter_area[0][1]-clutter_area[0][0])*
                     (clutter_area[1][1]-clutter_area[1][0]))

clutter_rate = 1e-10 # ( 1.2 is default)

clutter_spatial_density = clutter_rate/surveillance_area

# Instantiate the detection simulator
from stonesoup.simulator.simple import SimpleDetectionSimulator

detection_sim = SimpleDetectionSimulator(
    groundtruth=groundtruth_sim,
    measurement_model=measurement_model,
    detection_probability=probability_detection,
    meas_range=clutter_area,
    clutter_rate=clutter_rate)

# To make a 1 to 1 comparison between different trackers we have
# to feed the same detections to each trackers, so we have to
# duplicate the detection simulations.
from itertools import tee
detection, *detection_sims = tee(detection_sim, 6)  # Amir A

# %%
# 2) Prepare the trackers components with the different data associators;
# -----------------------------------------------------------------------
# We have set up the multi-target scenario, we instantiate all
# the relevant tracker components. We consider the
# :class:`~.UnscentedKalmanPredictor` and :class:`~.UnscentedKalmanUpdater`
# components for the tracker. Then, for the data association we
# use the :class:`~.JPDA` data associator implementation
# present in Stone Soup and the JPDA PyEHM implementation to
# gather relevant comparisons. Please note that we have to
# create multiple copies of the same detector simulator
# to provide each tracker with the same set of detections for
# a fairer comparison.

# %%
# Stone Soup tracker components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Load the Kalman predictor and updater
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.updater.kalman import UnscentedKalmanUpdater

# Instantiate the components
predictor = UnscentedKalmanPredictor(transition_model)
updater = UnscentedKalmanUpdater(measurement_model)

# Load the Initiator, Deleter and compose the trackers
from stonesoup.deleter.time import UpdateTimeStepsDeleter
deleter_tuned = 10  # 3 is the default value
deleter = UpdateTimeStepsDeleter(deleter_tuned) # Amir
# This one worked to give one state for one timestamp
# min_point_tune = 10 #5  # 2 is default value
# This one did not work and give multiple state for some timestamp
min_point_tune = 8

from stonesoup.initiator.simple import MultiMeasurementInitiator

# Load the probabilistic data associator and the tracker
from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA
from stonesoup.tracker.simple import MultiTargetMixtureTracker
from stonesoup.dataassociator.probability import JPDAwithLBP

# %%
# Design the trackers
# ^^^^^^^^^^^^^^^^^^^
factor_cov = 2

initial_states = GaussianState(np.array([0, 0, 0, 0]),
                               np.diag([5, 1, 5, 1]) ** factor_cov,
                               timestamp=simulation_start_time)

# Start with the standard JPDA
initiator = MultiMeasurementInitiator(
    prior_state=initial_states,
    measurement_model=None,
    deleter=deleter,
    data_associator=GlobalNearestNeighbour(PDAHypothesiser(predictor=predictor,
                                         updater=updater,
                                         clutter_spatial_density=clutter_spatial_density,
                                         prob_detect=probability_detection)),
    updater=updater,
    min_points=min_point_tune)

# Tracker
JPDA_tracker = MultiTargetMixtureTracker(
    initiator=initiator,
    deleter=deleter,
    detector=detection_sims[0],
    data_associator=JPDA(PDAHypothesiser(predictor=predictor,
                                         updater=updater,
                                         clutter_spatial_density=clutter_spatial_density,
                                         prob_detect=probability_detection)),
    updater=updater)

# Now we load the EHMJPDA, please note that the initiator is the same as the JPDA
EHM_initiator = MultiMeasurementInitiator(
    prior_state=initial_states,
    measurement_model=None,
    deleter=deleter,
    data_associator=GlobalNearestNeighbour(PDAHypothesiser(predictor=predictor,
                                         updater=updater,
                                         clutter_spatial_density=clutter_spatial_density,
                                         prob_detect=probability_detection,)),
    updater=updater,
    min_points=min_point_tune)

# In this tracker we use the JPDA with EHM
EHM1_tracker = MultiTargetMixtureTracker(
    initiator=EHM_initiator,
    deleter=deleter,
    detector=detection_sims[1],
    data_associator=JPDAWithEHM(PDAHypothesiser(predictor=predictor,
                                         updater=updater,
                                         clutter_spatial_density=clutter_spatial_density,
                                         prob_detect=probability_detection)),
    updater=updater)

# Copy the same initiator for EHM
EHM2_initiator = deepcopy(EHM_initiator)

# This tracker uses the the second implementation
# of EHM.
EHM2_tracker = MultiTargetMixtureTracker(
    initiator=EHM2_initiator,
    deleter=deleter,
    detector=detection_sims[2],
    data_associator=JPDAWithEHM2(PDAHypothesiser(predictor=predictor,
                                                updater=updater,
                                                clutter_spatial_density=clutter_spatial_density,
                                                prob_detect=probability_detection)),
    updater=updater)

# Now we load the LBPJPDA, please note that the initiator is the same as the JPDA (Amir)
LBP_initiator = MultiMeasurementInitiator(
    prior_state=initial_states,
    measurement_model=None,
    deleter=deleter,
    data_associator=GlobalNearestNeighbour(PDAHypothesiser(predictor=predictor,
                                         updater=updater,
                                         clutter_spatial_density=clutter_spatial_density,
                                         prob_detect=probability_detection,)),
    updater=updater,
    min_points=min_point_tune)

# In this tracker we use the JPDA with LBP (Amir)
LBP_tracker = MultiTargetMixtureTracker(
    initiator=LBP_initiator,
    deleter=deleter,
    detector=detection_sims[3],
    data_associator=JPDAwithLBP(hypothesiser=PDAHypothesiser(predictor=predictor,
                               updater=updater,
                               clutter_spatial_density=clutter_spatial_density,
                               prob_detect=probability_detection)),
    updater=updater)


# %%
# 3) Run the trackers to generate the tracks;
# -------------------------------------------
# We have instantiated the three versions of the
# trackers, one with the brute force JPDA hypothesis
# management, one with the EHM implementation [1]_ and
# one with the EHM2 implementation  [2]_.
# Now, we can run the trackers and gather the
# final tracks as well as the detections,
# clutter and define a metric plotter to evaluate
# the track accuracy using the metric manager.
# As the three methods will use the same hypothesis
# we will obtain the same tracks, we verify such
# claim by comparing the OSPA metric between
# each hyphotesiser.
# To measure the significant difference in
# computing time we measure the time while running the
# three different trackers.

# %%
# Stone Soup Metrics imports
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

# Instantiate the metrics tracker
from stonesoup.metricgenerator.basicmetrics import BasicMetrics

basic_JPDA = BasicMetrics(generator_name='basic_JPDA', tracks_key='JPDA_tracks',
                          truths_key='truths')
EHM1 = BasicMetrics(generator_name='EHM1', tracks_key='EHM1_tracks',
                    truths_key='truths')
EHM2 = BasicMetrics(generator_name='EHM2', tracks_key='EHM2_tracks',
                    truths_key='truths')
LBP = BasicMetrics(generator_name='LBP', tracks_key='LBP_tracks',
                    truths_key='truths') # Amir

# Compare the generated tracks to verify they obtain the same
# accuracy, we consider as truths tracks the EHM tracks
from stonesoup.metricgenerator.ospametric import OSPAMetric, NEESMetric, RMSEMetric, GOSPAMetric, KLMetric

# Instantiate the OSPA Metric
ospa_JPDA_EHM1 = OSPAMetric(c=40, p=1, generator_name='OSPA_JPDA-EHM1',
                            tracks_key='JPDA_tracks', truths_key='EHM1_tracks')
ospa_JPDA_EHM2 = OSPAMetric(c=40, p=1, generator_name='OSPA_JPDA-EHM2',
                           tracks_key='JPDA_tracks', truths_key='EHM2_tracks')
ospa_JPDA_LBP = OSPAMetric(c=40, p=1, generator_name='OSPA_JPDA-LBP',
                           tracks_key='JPDA_tracks', truths_key='LBP_tracks') # Amir

# Instantiate the NEES Metric
nees_JPDA_EHM1 = NEESMetric(c=None, p=None, generator_name='NEES_JPDA-EHM1',
                            tracks_key='JPDA_tracks', truths_key='EHM1_tracks')
nees_JPDA_EHM2 = NEESMetric(c=None, p=None, generator_name='NEES_JPDA-EHM2',
                            tracks_key='JPDA_tracks', truths_key='EHM2_tracks')
nees_JPDA_LBP = NEESMetric(c=None, p=None, generator_name='NEES_JPDA-LBP',
                           tracks_key='JPDA_tracks', truths_key='LBP_tracks')
#'''
# Instantiate the GOSPA Metric
gospa_JPDA_EHM1 = GOSPAMetric(c=40, p=1, generator_name='GOSPA_JPDA-EHM1',
                            tracks_key='JPDA_tracks', truths_key='EHM1_tracks')
gospa_JPDA_EHM2 = GOSPAMetric(c=40, p=1, generator_name='GOSPA_JPDA-EHM2',
                           tracks_key='JPDA_tracks', truths_key='EHM2_tracks')
gospa_JPDA_LBP = GOSPAMetric(c=40, p=1, generator_name='GOSPA_JPDA-LBP',
                           tracks_key='JPDA_tracks', truths_key='LBP_tracks') # Amir


# Instantiate the RMSE Metric
rmse_JPDA_EHM1 = RMSEMetric(c=None, p=None, generator_name='RMSE_JPDA-EHM1',
                            tracks_key='JPDA_tracks', truths_key='EHM1_tracks')
rmse_JPDA_EHM2 = RMSEMetric(c=None, p=None, generator_name='RMSE_JPDA-EHM2',
                            tracks_key='JPDA_tracks', truths_key='EHM2_tracks')
rmse_JPDA_LBP = RMSEMetric(c=None, p=None, generator_name='RMSE_JPDA-LBP',
                           tracks_key='JPDA_tracks', truths_key='LBP_tracks')

# Instantiate the KL Metric
kl_JPDA_EHM1 = KLMetric(c=None, p=None, generator_name='KL_JPDA-EHM1',
                        tracks_key='JPDA_tracks', truths_key='EHM1_tracks')
kl_JPDA_EHM2 = KLMetric(c=None, p=None, generator_name='KL_JPDA-EHM2',
                        tracks_key='JPDA_tracks', truths_key='EHM2_tracks')
kl_JPDA_LBP = KLMetric(c=None, p=None, generator_name='KL_JPDA-LBP',
                       tracks_key='JPDA_tracks', truths_key='LBP_tracks')
#'''
# Define the track data associator
from stonesoup.dataassociator.tracktotrack import TrackToTruth
associator = TrackToTruth(association_threshold=30)
# Set the backend to 'Agg' for non-interactive plotting
# import matplotlib
# matplotlib.use('Agg')


# Load the plotter
from stonesoup.metricgenerator.plotter import TwoDPlotter

plot_generator_JPDA = TwoDPlotter([0, 2], [0, 2], [0, 2], uncertainty=True, tracks_key='JPDA_tracks',
                                 truths_key='truths', detections_key='detections',
                                 generator_name='JPDA_plot')
plot_generator_EHM1 = TwoDPlotter([0, 2], [0, 2], [0, 2], uncertainty=True, tracks_key='EHM1_tracks',
                                truths_key='truths', detections_key='detections',
                                generator_name='EHM1_plot')

plot_generator_EHM2 = TwoDPlotter([0, 2], [0, 2], [0, 2], uncertainty=True, tracks_key='EHM2_tracks',
                                truths_key='truths', detections_key='detections',
                                generator_name='EHM2_plot')

plot_generator_LBP = TwoDPlotter([0, 2], [0, 2], [0, 2], uncertainty=True, tracks_key='LBP_tracks',
                                truths_key='truths', detections_key='detections',
                                generator_name='LBP_plot') # Amir


# Load the multi-manager
from stonesoup.metricgenerator.manager import MultiManager

# Load all the relevant components of the plots in the metric manager
metric_manager = MultiManager([basic_JPDA,
                               EHM1,
                               EHM2,
                               ospa_JPDA_EHM1,
                               ospa_JPDA_EHM2,
                               ospa_JPDA_LBP,
                               gospa_JPDA_EHM1, # Amir GOSPA
                               gospa_JPDA_EHM2, # Amir GOSPA
                               gospa_JPDA_LBP,  # Amir GOSPA
                               nees_JPDA_EHM1,  # Amir NEES
                               nees_JPDA_EHM2,  # Amir NEES
                               nees_JPDA_LBP,   # Amir NEES
                               rmse_JPDA_EHM1,  # Amir RMSE
                               rmse_JPDA_EHM2,  # Amir RMSE
                               rmse_JPDA_LBP,   # Amir RMSE
                               kl_JPDA_EHM1,    # Amir KL
                               kl_JPDA_EHM2,    # Amir KL
                               kl_JPDA_LBP,     # Amir KL
                               plot_generator_JPDA,
                               plot_generator_EHM1,
                               plot_generator_EHM2,
                               plot_generator_LBP  # Amir
                               ], associator)

# %%
# Run simulation
# ^^^^^^^^^^^^^^

# We  plot the various tracker results
JPDA_tracks = set()
EHM1_tracks = set()
EHM2_tracks = set()
LBP_tracks = set()     # Amir
groundtruths = set()
detections_set = set()

# We measure the computation time
start_time = perf_counter()
for time, ctracks in JPDA_tracker:
    JPDA_tracks.update(ctracks)
    detections_set.update(detection_sim.detections)

jpda_time = perf_counter() - start_time

groundtruths = groundtruth_sim.groundtruth_paths

start_time = perf_counter()
for time, etracks in EHM1_tracker:
    EHM1_tracks.update(etracks)

ehm1_time = perf_counter() - start_time


start_time = perf_counter()
for time, etracks in EHM2_tracker:
    EHM2_tracks.update(etracks)
ehm2_time = perf_counter() - start_time

## Added by Amir
start_time = perf_counter()
for time, etracks in LBP_tracker:
    LBP_tracks.update(etracks)
LBP_time = perf_counter() - start_time

# Add the various tracks to the metric manager
metric_manager.add_data({'truths': groundtruths,
                        'detections': detections_set}, overwrite=False)
metric_manager.add_data({'JPDA_tracks': JPDA_tracks}, overwrite=False)
metric_manager.add_data({'EHM1_tracks': EHM1_tracks}, overwrite=False)
metric_manager.add_data({'EHM2_tracks': EHM2_tracks}, overwrite=False)
metric_manager.add_data({'LBP_tracks': LBP_tracks}, overwrite=False) # Amir
# %%
# 4) Compare the trackers performances;
# -------------------------------------
# We have set up the trackers as well as
# the metric manager, to conclude this
# tutorial we show the results of the
# computing time needed for each tracker,
# the overall tracks generated and
# the differences between the tracks,
# if any. We start presenting the time
# performances of the different trackers along
# with the performance improvement obtained by the
# EHM data associators.

print('Comparisons between the trackers performances')
print(f'JPDA computing time: {jpda_time:.2f} seconds')
print(f'EHM1 computing time: {ehm1_time:.2f} seconds, {(jpda_time/ehm1_time-1)*100:.2f} % quicker than JPDA')
print(f'EHM2 computing time: {ehm2_time:.2f} seconds, {(jpda_time/ehm2_time-1)*100:.2f} % quicker than JPDA')
print(f'LBP computing time: {LBP_time:.2f} seconds, {(jpda_time/LBP_time-1)*100:.2f} % quicker than JPDA') # Amir

# Load the plotter package to plot the
# detections, tracks and detections.
from stonesoup.plotter import Plotterly, Plotter

# import os
# if not os.path.exists('results_scnario1'):
#     os.makedirs('results_scnario1')

'''
plotter = Plotterly()
plotter.plot_ground_truths(groundtruths, [0, 2])
plotter.plot_measurements(detections_set, [0, 2])
plotter.plot_tracks(JPDA_tracks, [0, 2], line= dict(color='orange'),
                    track_label='JPDA tracks')
plotter.plot_tracks(EHM1_tracks, [0, 2], line= dict(color='green', dash='dot'),
                    track_label='EHM1 tracks')
plotter.plot_tracks(EHM2_tracks, [0, 2], line= dict(color='red', dash='dot'),
                    track_label='EHM2 tracks')
plotter.plot_tracks(LBP_tracks, [0, 2], line= dict(color='blue', dash='dot'),
                    track_label='LBP tracks')
# plotter.fig
# plotter.fig.show()
'''

plotter = Plotterly()
plotter.plot_ground_truths(groundtruths, [0, 2])
plotter.plot_measurements(detections_set, [0, 2])
plotter.plot_tracks(JPDA_tracks, [0, 2],
                        track_label='JPDA')
#plotter = Plotter()
# plotter.plot_ground_truths(groundtruths, [0, 2])
# plotter.plot_measurements(detections_set, [0, 2])
plotter.plot_tracks(EHM1_tracks, [0, 2],
                        track_label='EHM1')
plotter.plot_tracks(EHM2_tracks, [0, 2],
                        track_label='EHM2')
plotter.plot_tracks(LBP_tracks, [0, 2],
                        track_label='LBP')
plotter.fig.show() #savefig('results_scnario1/Tracks_Scenario1.png')
# sys.exit()

# %%
# Show the metrics
# ^^^^^^^^^^^^^^^^
####################################################################################################
####################################### Small test of Amir Debugging ###############################
####################################################################################################
'''
# List of variables to keep
keep_vars = ['matplotlib', 'metrics', 'numpy','metric_manager']

# Delete all other variables
for var in list(globals().keys()):
    if var not in keep_vars and not var.startswith("_"):
        del globals()[var]
'''
import matplotlib.pyplot as plt
import numpy as np

metrics = metric_manager.generate_metrics()

MetricChar = ['NEES', 'OSPA',  'RMSE', 'KL']
tracker = ['EHM1', 'EHM2', 'LBP']
Metrics_id = ['NEES Metrics', 'OSPA distances', 'RMSE Metrics', 'KL Metrics']

for MetricChar_iter, Metrics_id_iter in zip(MetricChar, Metrics_id):
    for tracker_iter in tracker:
        # Access the corresponding metadata
        Test2 = metrics[MetricChar_iter + '_JPDA-' + tracker_iter][Metrics_id_iter].metadata
        Score_char = MetricChar_iter
        print(MetricChar_iter + ' ' +tracker_iter )

        # Prepare data
        timestamps = [item['timestamp'] for item in Test2]  # Replace with your actual timestamps
        Target_scores = [item['overall_' + MetricChar_iter].value for item in Test2]  # Target scores over time
        meas_state_vector = ([item['meas'] for item in Test2])  # Estimated state vectors
        truth_state_vector = ([item['truth'] for item in Test2])  # Estimated state vectors

        error_vector = [item['error'] for item in Test2]  # Error vectors
        Covar_vector = ([item['covariance'] for item in Test2])  # Covariance matrices

        # Plot 1: Metric Scores over Time
        plt.figure(figsize=(10, 6))
        # plt.plot(timestamps, Target_scores, label=[Score_char + ' Score'], color='blue')
        plt.plot(timestamps, Target_scores, label=f'{Score_char} Score', color='blue')
        plt.xlabel('Time')
        plt.ylabel(Score_char + ' Score')
        plt.title(Score_char + ' Score per Time (per timestamp) for ' + tracker_iter)
        plt.legend()
        plt.grid()
        # plt.show()
        # Save the plot to the 'results' directory with the filename 'sigma.png'
        # Ensure the directory exists
        plt.savefig(f'{MetricChar_iter}_curve_{tracker_iter}_Scenario1.png')
        plt.close()  # Close the plot to free up memory


        ''''
        # Plot 2: Errors over Time
        plt.figure(figsize=(12, 10))
        
        # Plot each error component
        for i in range(4):
            plt.subplot(4, 1, i+1)
            plt.plot(timestamps, error_vector[:, i], label=f'Error in State {i+1}', color='green')
            plt.xlabel('Time')
            plt.ylabel(f'Error {i+1}')
            plt.title(f'Error in State {i+1} Over Time')
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.show()

        # Plot 3: State Vectors with Variances
        plt.figure(figsize=(12, 10))

        # Plot each state component with variance
        for i in range(4):
            plt.subplot(4, 1, i+1)
            plt.plot(timestamps, meas_state_vector[:, i], label=f'Estimated State {i+1}', color='red')
            plt.plot(timestamps, truth_state_vector[:, i], label=f'Ground Truth State {i+1}', color='blue', linestyle='dashed')
            plt.xlabel('Time')
            plt.ylabel(f'State {i+1}')
            plt.title(f'Comparison of State {i+1} with Variance')
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.show()
        '''

####################################################################################################
####################################### End of Small test of Amir Debugging ########################
####################################################################################################


####################################################################################################
####################################### Plot Metrics ###############################################
####################################################################################################

# Load the metric plotter
from stonesoup.plotter import MetricPlotter
graph = MetricPlotter()

# Now we process the OSPA Metrics
graph.plot_metrics(metrics, generator_names=['OSPA_JPDA-EHM1',
                                             'OSPA_JPDA-EHM2',
                                             'OSPA_JPDA-LBP'],
                   color=['orange',  'blue', 'green'])

# update y-axis label and title, other subplots are displaying auto-generated title and labels
graph.axes[0].set(ylabel='OSPA metrics', title='OSPA distances over time between JPDA and EHMs tracks')
graph.fig.savefig('metrics_plot_OSPA_scenario1.png')

# Now we process the NEES Metrics
graph.plot_metrics(metrics, generator_names=['NEES_JPDA-EHM1',
                                             'NEES_JPDA-EHM2',
                                             'NEES_JPDA-LBP'],
                   color=['orange', 'blue', 'green'])
# Update y-axis label and title
graph.axes[0].set(ylabel='NEES metrics', title='NEES values over time between JPDA and EHMs tracks')
graph.fig.savefig('metrics_plot_nees_scenario1.png')

# Now we process the RMSE Metrics
graph.plot_metrics(metrics, generator_names=['RMSE_JPDA-EHM1',
                                             'RMSE_JPDA-EHM2',
                                             'RMSE_JPDA-LBP'],
                   color=['orange', 'blue', 'green'])


# Update y-axis label and title
graph.axes[0].set(ylabel='RMSE metrics', title='RMSE values over time between JPDA and EHMs tracks')
graph.fig.savefig('metrics_plot_rmse_scenario1.png')


# Now we process the GOSPA Metrics
graph.plot_metrics(metrics, generator_names=['GOSPA_JPDA-EHM1',
                                             'GOSPA_JPDA-EHM2',
                                             'GOSPA_JPDA-LBP'],
                   color=['orange', 'blue', 'green'])


# Update y-axis label and title
graph.axes[0].set(ylabel='GOSPA metrics', title='GOSPA values over time between JPDA and EHMs tracks')
graph.fig.savefig('metrics_plot_gospa_scenario1.png')

# Now we process the KL Metrics
graph.plot_metrics(metrics, generator_names=['KL_JPDA-EHM1',
                                             'KL_JPDA-EHM2',
                                             'KL_JPDA-LBP'],
                   color=['orange', 'blue', 'green'])


# Update y-axis label and title
graph.axes[0].set(ylabel='KL metrics', title='KL values over time between JPDA and EHMs tracks')
graph.fig.savefig('metrics_plot_kl_scenario1.png')





