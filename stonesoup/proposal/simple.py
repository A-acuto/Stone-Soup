from typing import Union

import numpy as np
from enum import Enum
from scipy.stats import multivariate_normal as mvn

from stonesoup.base import Property
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition import TransitionModel
from stonesoup.proposal.base import Proposal
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.detection import Detection
from stonesoup.types.numeric import Probability
from stonesoup.types.state import State, GaussianState, SqrtGaussianState, ParticleState
from stonesoup.types.prediction import Prediction, ParticleStatePrediction
from stonesoup.updater.base import Updater
from stonesoup.predictor.base import Predictor
from stonesoup.predictor.kalman import SqrtKalmanPredictor
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.numeric import Probability
from stonesoup.types.particle import Particle


class PriorAsProposal(Proposal):
    """Proposal that uses the dynamics model as the importance density.
    This proposal uses the dynamics model to predict the next state.
    """
    transition_model: TransitionModel = Property(
        doc="The transition model used to make the prediction")

    def rvs(self, prior: State, measurement=None, time_interval=None, **kwargs):
        """Generate samples from the proposal.
        Parameters
        ----------
        state: :class:`~.State`
            The state to generate samples from.
        measurement: :class:`~.Detection`, optional
            The measurement used to calculate the time interval. If not provided, the time interval
            will be used to propagate the state.
        time_interval: :class:`datetime.timedelta`, optional
            The time interval between the prior and the proposed state. Only used if no detection
            is provided.

        Returns
        -------
        : 2-D array of shape (:attr:`ndim`, ``num_samples``)
            A set of Np samples, generated from the model's noise
            distribution.

        """

        if measurement is not None:
            timestamp = measurement.timestamp
            time_interval = measurement.timestamp - prior.timestamp
        else:
            timestamp = prior.timestamp + time_interval

        new_state_vector = self.transition_model.function(prior,
                                                          time_interval=time_interval,
                                                          **kwargs)
        return Prediction.from_state(prior,
                                     parent=prior,
                                     state_vector=new_state_vector,
                                     timestamp=timestamp,
                                     transition_model=self.transition_model,
                                     prior=prior)


class KalmanFilterProposalScheme(Enum):
    """Kalman Filter proposal scheme enumeration"""
    GLOBAL = 'global'  #: Global KF scheme
    LOCAL = 'local'  #: Local KF scheme


class KFasProposal(Proposal):
    """This proposal should inherit the Kalman properties
        to perform the various steps required
    """
    predictor: Predictor = Property(
        doc="predictor to use the various values")
    updater: Updater = Property(
        doc="Updater used for update the values")
    scheme: KalmanFilterProposalScheme = Property(
        default=KalmanFilterProposalScheme.GLOBAL,
        doc="The scheme for performing the Kalman Filter passage from a particle distribution. "
            "The difference between the two schemes comes from the approximation chosen: the "
            "``'global'`` scheme approximates the particles distribution with a Gaussian with the "
            "same mean and same covariance, with this state we apply the Kalman filtering and then"
            "we sample from the posterior distribution; ``'local'`` scheme instead treats each "
            "particle as a Gaussian state with zero covariance and propagate them using the "
            "Kalman iteration, at the end we sample from each posterior distribution and we "
            "compile the resulting final particle state. It is significantly slower than the "
            "global appoximation")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure kalman proposal scheme is a valid KalmanFilterProposalScheme
        self.scheme = KalmanFilterProposalScheme(self.scheme)

    def rvs(self, prior: State, measurement: Detection, time_interval=None, **kwargs):
        """Generate samples from the proposal.
            Use the kalman filter predictor
        Parameters
        ----------
        prior: :class:`~.State`
            The state to generate samples from.
        Returns
        -------
        : 2-D array of shape (:attr:`ndim`, ``num_samples``)
            A set of Np samples, generated from the model's noise
            distribution.
        """

        # get the number of particles
        num_particles = prior.state_vector.shape[1]

        # Get the time interval and timestamp
        if measurement is not None:
            timestamp = measurement.timestamp
            time_interval = measurement.timestamp - prior.timestamp
        else:
            timestamp = prior.timestamp + time_interval

        if time_interval.total_seconds() == 0:
            return Prediction.from_state(prior,
                                         parent=prior,
                                         state_vector=prior.state_vector,
                                         timestamp=timestamp,
                                         transition_model=self.predictor.transition_model,
                                         prior=prior)

        # Define the KF prior class
        prior_cls = GaussianState  # Default
        if isinstance(self.predictor, SqrtKalmanPredictor):
            prior_cls = SqrtGaussianState

        # Check the kalman filter proposal
        if self.scheme == KalmanFilterProposalScheme.GLOBAL:
            # Construct the Kalman filter prior
            kalman_prior = prior_cls(prior.mean, prior.covar, prior.timestamp)

            # Perform the Kalman filter prediction
            kalman_prediction = self.predictor.predict(kalman_prior, timestamp=timestamp)

            if measurement is not None:
                posterior_state = self.updater.update(
                    SingleHypothesis(kalman_prediction, measurement))
            else:
                # in case we don't have the measurement
                posterior_state = kalman_prediction  # keep the prediction

            # need to sample from the posterior now
            samples = mvn.rvs(posterior_state.state_vector.reshape(-1),
                              posterior_state.covar,
                              size=num_particles)

            # Compute the log of q(x_k|x_{k-1}, y_k)
            post_log_weights = mvn.logpdf(samples, np.array(posterior_state.mean).reshape(-1),
                                          posterior_state.covar)

        else:
            # Null covariance
            null_covar = np.zeros_like(prior.covar)

            predictions = [
                self.predictor.predict(
                    prior_cls(particle_sv, null_covar, prior.timestamp),
                    timestamp=timestamp,
                    noise=kwargs['noise'])
                for particle_sv in prior.state_vector
            ]

            if measurement is not None:
                updates = [self.updater.update(SingleHypothesis(prediction, measurement))
                           for prediction in predictions]
            else:
                # in case we don't have the measurement
                updates = predictions  # keep the prediction

            # Draw the samples
            samples = np.array([state.state_vector.reshape(-1) + mvn.rvs(cov=state.covar).T
                                for state in updates])

            # Compute the log of q(x_k|x_{k-1}, y_k)
            post_log_weights = np.array([mvn.logpdf(sample,
                                                    np.array(update.mean).reshape(-1),
                                                    update.covar)
                                         for sample, update in zip(samples, updates)])

        # Construct the prediction state
        pred_state = Prediction.from_state(prior,
                                           parent=prior,
                                           state_vector=StateVectors(samples.T),
                                           timestamp=timestamp,
                                           transition_model=self.predictor.transition_model,
                                           prior=prior)

        # Compute the log of p(x_k|x_{k-1})
        prior_log_weights = self.predictor.transition_model.logpdf(pred_state, prior,
                                                               time_interval=time_interval)

        # NOTE: The above may need to be different for 'global' and 'local' schemes. Currently,
        # it breaks the 'global' scheme. Spoke briefly to Paul H. about this, but he's on leave so
        # will chat again on Monday. To "fix" (may not be statistically correct), replace as
        # follows:
        # try:
        #     if self.scheme == KalmanFilterProposalScheme.GLOBAL:
        #         prior_log_weights = mvn.logpdf(pred_state.state_vector.T,
        #                                        np.array(kalman_prediction.mean).reshape(-1),
        #                                        kalman_prediction.covar)
        #     else:
        #         prior_log_weights = self.predictor.transition_model.logpdf(pred_state, prior,
        #                                                                    time_interval=time_interval)
        # except:
        #     a = 2

        # Compute the log weights w_{k|k-1} = w_{k-1} * p(x_k|x_{k-1}) / q(x_k|x_{k-1}, k_k)
        pred_state.log_weight = (prior.log_weight + prior_log_weights - post_log_weights)

        return pred_state
