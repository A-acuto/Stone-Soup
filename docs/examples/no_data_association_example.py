"""

Comparing various filters in no-data association scenarios
==========================================================

In this example we consider the complex case of running
tracker algorithms while on measurements involving no-
data associations, in particular coming from GPS or
navigation scenario. In this case, we consider the signal
is obtained by passive radars.
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

# We can consider as PassiveElevationBearing sensor

# %%
# 2) instantiate the various tracker components
# ---------------------------------------------
# We have out detections and measurements now
# we want to set up the various tracker components.
# We consider a class of :class:`~.ExtendedKalmanPredictor`
# :class:`~.UnscentedKalmanPredictor` and
# :class:`~.ParticlePredictor`

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

initiator_EKF = SimpleMeasurementInitiator(
    prior=GaussianState(np.array([0,0,0,0]),
                        np.diag([0.1, 0.1, 0.1, 0.1])),  # I will place it on top of the first measurements
    measurement_model= None,
    deleter= deleter,
    data_associator= None,  # will this work?
    updater= updater_EKF)

initiator_UKF = SimpleMeasurementInitiator(
    prior=GaussianState(np.array([0,0,0,0]),
                        np.diag([0.1, 0.1, 0.1, 0.1])),  # I will place it on top of the first measurements
    measurement_model= None,
    deleter= deleter,
    data_associator= None,  # will this work?
    updater= updater_UKF)

prior_state = GaussianState(np.array([0,0,0,0]),
                        np.diag([0.1, 0.1, 0.1, 0.1]))

initiator_part = SimpleMeasurementInitiator(
    prior_state=prior_state,
    measurement_model= None,
    skip_non_reversible= True
)

initiator_PF = GaussianParticleInitiator(number_particles=500,
                                         initator=initiator_part)

# Now we can create the various trackers. I assume in all cases it is a single target

from stonesoup.tracker.simple import SingleTargetTracker

tracker_UKF = SingleTargetTracker(
    initiator= initiator_UKF,
    updater= updater_UKF,
    detector= None, # or the data coming from outside
    deleter= deleter,
    data_association= None
)

tracker_EKF = SingleTargetTracker(
    initiator= initiator_EKF,
    updater= updater_EKF,
    detector= None, # or the data coming from outside
    deleter= deleter,
    data_association= None
)

tracker_PF = SingleTargetTracker(
    initiator= initiator_PF,
    updater= updater_PF,
    detector= None, # or the data coming from outside
    deleter= deleter,
    data_association= None
)

# %%
# 3) run the tracker on the data
# ------------------------------
# We have initialised the various trackers,
# using the various filters we want to compare.
# As explained earlier we don't consider the case
# of data associations.
# Before running the trackers on the data,
# let's prepare the metric manager to compute
# the various performances metrics.

# Load the metric manager
from stonesoup.metricgenerator.basicmetrics import BasicMetrics

basic_EKF = BasicMetrics(generator_name='Extended Kalman Filter', tracks_key='EKF_tracks', truths_key='truths')
basic_UKF = BasicMetrics(generator_name='Unscented Kalman Filter', tracks_key='UKF_tracks', truths_key='truths')
basic_PF = BasicMetrics(generator_name='Particle Filter', tracks_key='PF_tracks', truths_key='truths')

# Load the OSPA metric managers
from stonesoup.metricgenerator.ospametric import OSPAMetric
ospa_EKF_truth = OSPAMetric(c=40, p=1, generator_name='OSPA_EKF_truths',
                           tracks_key= 'EKF_tracks',  truths_key='truths')
ospa_UKF_truth = OSPAMetric(c=40, p=1, generator_name='OSPA_UKF_truths',
                           tracks_key= 'UKF_tracks',  truths_key='truths')
ospa_PF_truth = OSPAMetric(c=40, p=1, generator_name='OSPA_PF_truths',
                           tracks_key= 'PF_tracks',  truths_key='truths')

# load the SIAP metric (single integrated air picture)
from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics
from stonesoup.measures import Euclidean

siap_EKF_truth = SIAPMetrics(position_measure= Euclidean((0, 2)),
                             velocity_measure= Euclidean((1, 3)),
                             generator_name= 'SIAP_EKF-truth',
                             tracks_key= 'EKF_tracks',
                             truths_key= 'truths')

siap_UKF_truth = SIAPMetrics(position_measure= Euclidean((0, 2)),
                             velocity_measure= Euclidean((1, 3)),
                             generator_name= 'SIAP_UKF-truth',
                             tracks_key= 'UKF_tracks',
                             truths_key= 'truths')

siap_PF_truth = SIAPMetrics(position_measure= Euclidean((0, 2)),
                             velocity_measure= Euclidean((1, 3)),
                             generator_name= 'SIAP_PF-truth',
                             tracks_key= 'PF_tracks',
                             truths_key= 'truths')

# load the uncertainty metrics
from stonesoup.metricgenerator.uncertaintymetric import SumofCovarianceNormsMetric

sum_covariance_EKF = SumofCovarianceNormsMetric(tracks_key= 'EKF_tracks',
                                                generator_name= 'sum_covariance_EKF')

sum_covariance_UKF = SumofCovarianceNormsMetric(tracks_key= 'UKF_tracks',
                                                generator_name= 'sum_covariance_UKF')

sum_covariance_PF = SumofCovarianceNormsMetric(tracks_key= 'PF_tracks',
                                                generator_name= 'sum_covariance_PF')

# load the plotters
from stonesoup.metricgenerator.plotter import TwoDPlotter

plot_generator_EKF = TwoDPlotter([0, 2], [0, 2], [0, 2], uncertainty=True, tracks_key='EKF_tracks',
                                 truths_key='truths', detections_key='detections',
                                 generator_name='EKF_plot')

plot_generator_UKF = TwoDPlotter([0, 2], [0, 2], [0, 2], uncertainty=True, tracks_key='UKF_tracks',
                                 truths_key='truths', detections_key='detections',
                                 generator_name='UKF_plot')

plot_generator_PF = TwoDPlotter([0, 2], [0, 2], [0, 2], uncertainty=True, tracks_key='PF_tracks',
                                truths_key='truths', detections_key='detections',
                                generator_name='PF_plot')


# Define a data associator between the tracks and the truths
from stonesoup.dataassociator.tracktotrack import TrackToTruth
associator = TrackToTruth(association_threshold= 30)

# Place all the relevant metrics and
from stonesoup.metricgenerator.manager import MultiManager
metric_manager = MultiManager([basic_EKF,
                               basic_UKF,
                               basic_PF,
                               ospa_EKF_truth,
                               ospa_UKF_truth,
                               ospa_PF_truth,
                               siap_EKF_truth,
                               siap_UKF_truth,
                               siap_PF_truth,
                               sum_covariance_EKF,
                               sum_covariance_UKF,
                               sum_covariance_PF,
                               plot_generator_EKF,
                               plot_generator_UKF,
                               plot_generator_PF],
                              associator)

# Now we can run the various trackers and store the data

# missing a way to pass the detector detections
for step, (time, current_tracks) in enumerate(tracker_EKF, 1):
    metric_manager.add_data({'EKF_tracks': current_tracks}, overwrite=False)

for step, (time, current_tracks) in enumerate(tracker_UKF, 1):
    metric_manager.add_data({'UKF_tracks': current_tracks}, overwrite=False)

for step, (time, current_tracks) in enumerate(tracker_PF, 1):
    metric_manager.add_data({'PF_tracks': current_tracks}, overwrite=False)


# TO fill with the detections and the truths
metric_manager.add_data({'truths': truths,
                         'detections': detections}, overwrite=False)


# %%
# 4) visualise the tracker results
# --------------------------------
# We have run the trackers and we have
# gathered the relevant metrics to
# perform the evaluation. Now, we
# visualise firstly a visual comparison
# between the target track and the trackers
# performances and then we visualise the
# various metrics such as SIAP, OSPA and cumulative variance.

metrics = metric_manager.generate_metrics()

from stonesoup.plotter import MetricPlotter

graph = MetricPlotter()
graph.plot_metrics(metrics, generator_names=['OSPA_EKF-truth',
                                             'OSPA_PF-truth',
                                             'OSPA_EKF-PF',
                                             'SIAP_EKF-truth',
                                             'SIAP_PF-truth'],
                   # metric_names=['OSPA distances',
                   #               'SIAP Position Accuracy at times'],  # uncomment and run to see effect
                   color=['orange', 'green', 'blue'])

# update y-axis label and title, other subplots are displaying auto-generated title and labels
graph.axes[0].set(ylabel='OSPA metrics', title='OSPA distances over time')
graph.fig.show()


# %%
# We have presented an example showcasing how
# it is possible to use passive sensor data, e.g. GPS,
# with no data association algorithms inside Stone Soup
# and we have compared how different trackers perform
# in predicting the real target trajectory.



