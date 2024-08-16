import itertools

import datetime
import numpy as np
import pytest

# Import the proposals
from stonesoup.proposal.simple import PriorAsProposal, KFasProposal
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.types.particle import Particle
from stonesoup.types.prediction import ParticleStatePrediction
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.types.state import ParticleState, GaussianState
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.hypothesis import SingleHypothesis


def test_prior_proposal():
    # test that the proposal as prior and basic PF implementation
    # yield same results, since they are driven by the transition model
    np.random.seed(16549)

    # Initialise a transition model
    cv = ConstantVelocity(noise_diff_coeff=0)

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - timestamp

    num_particles = 9  # Number of particles

    # Define prior state
    prior_particles = [Particle(np.array([[i], [j]]), 1/num_particles)
                       for i, j in itertools.product([10, 20, 30], [10, 20, 30])]
    prior = ParticleState(None, particle_list=prior_particles, timestamp=timestamp)

    # predictors prior and standard stone soup
    predictor_prior = ParticlePredictor(cv,
                                        proposal=PriorAsProposal(cv))

    # Check that the predictor without prior specified works with the prior as
    # proposal
    predictor_base = ParticlePredictor(cv)

    # basic transition model evaluations
    eval_particles = [Particle(cv.matrix(timestamp=new_timestamp,
                                         time_interval=time_interval)
                               @ particle.state_vector,
                               1 / 9)
                      for particle in prior_particles]
    eval_mean = np.mean(np.hstack([i.state_vector for i in eval_particles]),
                        axis=1).reshape(2, 1)

    # construct the evaluation prediction
    eval_prediction = ParticleStatePrediction(None, new_timestamp, particle_list=eval_particles)

    prediction_base = predictor_base.predict(prior, timestamp=new_timestamp)
    prediction_prior = predictor_prior.predict(prior, timestamp=new_timestamp)

    assert np.all([eval_prediction.state_vector[:, i] ==
                   prediction_base.state_vector[:, i] for i in range(9)])
    assert np.all([prediction_base.weight[i] == 1 / 9 for i in range(9)])

    assert np.allclose(prediction_prior.mean, eval_mean)
    assert prediction_prior.timestamp == new_timestamp
    assert np.all([eval_prediction.state_vector[:, i] ==
                   prediction_prior.state_vector[:, i] for i in range(9)])
    assert np.all([prediction_prior.weight[i] == 1 / 9 for i in range(9)])


def test_kf_proposal():

    np.random.seed(16549)

    # Initialise a transition model
    cv = ConstantVelocity(noise_diff_coeff=1)

    # initialise the measurement model
    lg = LinearGaussian(ndim_state=2,
                        mapping=[0],
                        noise_covar=np.diag([1]))

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - timestamp

    num_particles = 9  # Number of particles

    # Define prior state
    prior_particles = [Particle(np.array([[i], [j]]), 1/num_particles)
                       for i, j in itertools.product([1, 2, 3], [1, 2, 3])]

    prior = ParticleState(None, particle_list=prior_particles, timestamp=timestamp)

    prior_kf = GaussianState(prior.mean, prior.covar, prior.timestamp)
    kf_predictor = KalmanPredictor(cv)
    kf_updater = KalmanUpdater(lg)

    # perform the kalman filter update
    prediction = kf_predictor.predict(prior_kf, timestamp=new_timestamp)

    new_state = GaussianState(state_vector=cv.function(prior_kf, noise=True,
                                                       time_interval=time_interval),
                              covar=np.diag([1, 1]),
                              timestamp=new_timestamp)

    detection = Detection(lg.function(new_state,
                                      noise=True),
                          timestamp=new_timestamp,
                          measurement_model=lg)

    predictions = [kf_predictor.predict(
        GaussianState(particle_sv, np.zeros_like(prior.covar), prior.timestamp),
        timestamp=timestamp)
        for particle_sv in prior.state_vector]

    updates = [kf_updater.update(SingleHypothesis(prediction, detection))
               for prediction in predictions]

#    eval_state = kf_updater.update(SingleHypothesis(prediction, detection))

    proposal = KFasProposal(KalmanPredictor(cv),
                            KalmanUpdater(lg))

    x = proposal.rvs(prior, measurement=detection, time_interval=time_interval)

    # pf_predictor = ParticlePredictor(cv,
    #                                  proposal=proposal)
    #
    # pf_updater = ParticleUpdater(lg)
    #
    # pred_state = pf_predictor.predict(prior, timestamp=new_timestamp, detection=detection)
    #

    #print()
    assert x.state_vector.shape == prior.state_vector.shape
#    assert np.allclose(pred_state.mean, eval_state.state_vector)
#    assert np.allclose(x.mean, eval_state.state_vector)
    assert pred_state.timestamp == new_timestamp