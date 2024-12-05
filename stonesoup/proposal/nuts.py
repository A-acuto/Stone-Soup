import numpy as np
import numpy.random as npr
from scipy.stats import multivariate_normal as mvn
from scipy.stats import uniform
from copy import deepcopy, copy

from stonesoup.base import Property
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.state import ParticleState, State
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition import TransitionModel
from stonesoup.proposal.base import Proposal
from stonesoup.types.detection import Detection
from stonesoup.types.prediction import Prediction


class NUTS(Proposal):
    """No-U Turn Sampler"""

    transition_model: TransitionModel = Property(
        doc="The transition model used to make the prediction")  #  ~ used instead of target
    measurement_model: MeasurementModel = Property(
        doc="The measurement model used to evaluate the likelihood")
    step_size: float = Property(doc='Step size used in the LeapFrog calculation')
    mass_matrix: float = Property(doc='Mass matrix something something')
    mapping: tuple = Property(doc="Localisation mapping")
    v_mapping: tuple = Property(doc="Velocity mapping")
    num_dims: int = Property(doc='State dimension')
    num_samples: int = Property(doc='Number of samples')
    target_proposal_input: float = Property(
        doc='particle distribution',
        default=None)
    grad_target: float = Property(
        doc='Gradient of the particle distribution',
        default=None)
    max_tree_depth: int = Property(
        doc="Maximum tree depth NUTS can take to stop excessive tree growth.",
        default=10)
    delta_max: int = Property(
        doc='Rejection criteria threshold',
        default=100)

    # Initialise
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MM = np.tile(self.mass_matrix, (self.num_samples, 1, 1))  # mass matrix
        self.inv_MM = np.tile(np.linalg.pinv(self.mass_matrix), (self.num_samples, 1, 1))

        # Ensure step size is an array
        if np.isscalar(self.step_size):
            self.step_size = np.repeat(self.step_size, self.num_samples)

    def target_proposal(self, prior, state, detection,
                        time_interval):
        """Somewhat a target proposal that can be called for basic things"""
        tg_proposal = self.transition_model.logpdf(state, prior, time_interval=time_interval,
                                                   allow_singular=True) + \
                      self.measurement_model.logpdf(detection, state,
                                                    allow_singular=True)
        return tg_proposal

    def grad_target_proposal(self, prior, state, detection, time_interval, **kwargs):

        # grad log prior
        dx = state.state_vector - self.transition_model.function(prior,
                                                                 time_interval=time_interval,
                                                                 **kwargs)

        grad_log_prior = (
            np.linalg.pinv(self.transition_model.covar(time_interval=time_interval)) @ (-dx)
        ).T

        # Get Jacobians of measurements
        H = self.measurement_model.jacobian(state)

        # Get innov
        dy = detection.state_vector - self.measurement_model.function(state, **kwargs)

        # Compute the gradient H^T * inv(R) * innov
        if len(H.shape) < 3:
            # Single Jacobian matrix
            grad_log_pdf = (H.T @ np.linalg.pinv(self.measurement_model.covar()) @ dy).T
        else:
            # Jacobian matrix for each point
            HTinvR = H.transpose((0, 2, 1)) @ self.linalg.pinv(self.measurement_model.covar())
            grad_log_pdf = (HTinvR @ np.atleast_3d(dy))[:, :, 0]

        return grad_log_prior + grad_log_pdf

    def rvs(self, state, measurement: Detection = None, time_interval=None,
            **kwargs):

        # evaluate the measurement timestmaps, maybe no needed
        if measurement is not None:
            timestamp = measurement.timestamp
            time_interval = measurement.timestamp - state.timestamp
        else:
            timestamp = state.timestamp + time_interval

        if time_interval.total_seconds() == 0:
            return Prediction.from_state(state,
                                         parent=state,
                                         state_vector=state.state_vector,
                                         timestamp=state.timestamp,
                                         transition_model=self.transition_model,
                                         prior=state)

        # Copy the old state for the parent
        previous_state = copy(state)
        # Create a copy of the state vector to ensure the original is not modified
        previous_state.state_vector = deepcopy(state.state_vector)

        # state is the prior - propagate
        new_state = self.transition_model.function(state,
                                                   time_interval=time_interval,
                                                   **kwargs)

        new_state_pred = Prediction.from_state(state,
                                               parent=state,
                                               state_vector=new_state,
                                               timestamp=timestamp,
                                               transition_model=self.transition_model,
                                               prior=state)

        # evaluate the momentum
        v = mvn.rvs(mean=np.zeros(self.num_dims), cov=self.MM[0], size=self.num_samples)

        # calculate the gradient of the function
        #        grad_x = self.get_grad(state, time_interval)

        grad_x = self.grad_target_proposal(state, new_state_pred, measurement, time_interval)

        x_new, v_new, acceptance = self.generate_nuts_samples(state, new_state_pred,
                                                              v, grad_x, measurement,
                                                              time_interval)

        # pi(x_k)  # state is like prior
        pi_x_k = self.target_proposal(x_new, state, measurement, time_interval)

        # pi(x_k-1)
        pi_x_k1 = self.target_proposal(state, state, measurement, time_interval)

        # L-kernel
        #L = 1
        # q(x_k|x_k-1)
        #q = 1

        final = Prediction.from_state(previous_state,
                                      parent=previous_state,
                                      state_vector=x_new.state_vector,
                                      timestamp=timestamp,
                                      transition_model=self.transition_model,
                                      prior=state)

        final.log_weight += pi_x_k - pi_x_k1
        return final

    # looks like it is not used
    # def log_v_pdf(self, v):  # v is the particle momentum
    #     return mvn.logpdf(x=v, mean=np.zeros(self.num_samples), cov=self.MM[0])

    # standard pauls implementation
    # def get_grad(self, new_state, prior, time_interval):
    #     """get the functon gradient"""
    #     return self.grad_target(new_state.state_vector,
    #                             prior.state_vector,
    #                             time_interval)

    def get_grad(self, new_state, time_interval):
        """for now use the jacobian"""
        return self.transition_model.jacobian(new_state, time_interval=time_interval)

    def integrate_lf_vec(self, state, new_state_pred, v, grad_x, direction, h, time_interval,
                         measurement):
        # something on the integration  // removed the direction reshaped
        h = h.reshape(self.num_samples, 1)  # reshape the h?
        v = v + direction * (h / 2) * grad_x
        einsum = np.einsum('bij,bj->bi', self.inv_MM, v)
        state.state_vector = (state.state_vector.T + direction * h * einsum).T

        grad_x = self.grad_target_proposal(state, new_state_pred, measurement, time_interval)
        v = v + direction * (h / 2) * grad_x
        return state, v, grad_x

    def stop_criterion_vec(self, xminus, xplus, rminus, rplus):
        # Return True for particles we want to stop (NB opposite way round to s in Hoffman
        # and Gelman paper)
        dx = xplus.state_vector.T - xminus.state_vector.T
        left = (np.sum(dx * np.einsum('bij,bj->bi', self.inv_MM, rminus), axis=1) < 0)
        right = (np.sum(dx * np.einsum('bij,bj->bi', self.inv_MM, rplus), axis=1) < 0)
        return np.logical_or(left, right)

    def get_hamiltonian(self, v, logp):  # removed x since it is not used
        # Get Hamiltonian energy of system given log target weight logp
        return logp - 0.5 * np.sum(v * np.einsum('bij,bj->bi', self.inv_MM, v), axis=1).reshape(-1,
                                                                                                1)

    def merge_states_dir(self, xminus, vminus, grad_xminus, xplus, vplus, grad_xplus, direction):
        # Return xmerge = vectors of xminus where direction < 0 and xplus where direction > 0, and
        # similarly for v and grad_x
        mask = direction[:, 0] < 0
        mask = mask.reshape(len(mask), 1).astype(int)
        xmerge = mask * xminus.state_vector.T + (1 - mask) * xplus.state_vector.T
        vmerge = mask * vminus + (1 - mask) * vplus
        grad_xmerge = mask * grad_xminus + (1 - mask) * grad_xplus
        xminus.state_vector = xmerge.T
        return xminus, vmerge, grad_xmerge

    def generate_nuts_samples(self, x0, x1, v0, grad_x0, detection, time_interval):  # removed rngs
        # Sample energy: note that log(U(0,1)) has same distribution as -exponential(1)
        logp0 = self.target_proposal(x1, x0, detection, time_interval=time_interval).reshape(-1, 1)
        joint = self.get_hamiltonian(v0, logp0)  ##  removed x0 since it is not used
        logu = joint + np.log(uniform.rvs())  # np.log(rng.uniform())  ## original implementation

        # initialisation
        xminus = x0
        xplus = x0
        vminus = v0
        vplus = v0
        xprime = x0
        vprime = v0
        grad_xplus = grad_x0
        grad_xminus = grad_x0
        depth = 0

        # criterions
        stopped = np.zeros((self.num_samples, 1)).astype(bool)  # 0 if still running (opposite to MATLAB)
        numnodes = np.ones((self.num_samples, 1)).astype(int)

        # Used to compute acceptance rate
        alpha = np.zeros((self.num_samples, 1))
        nalpha = np.zeros((self.num_samples, 1)).astype(int)

        while np.any(stopped == 0):

            # Generate random direction in {-1, +1}
            direction = (2 * (uniform.rvs(0, 1, size=self.num_samples) < 0.5).astype(int) - 1).reshape(-1, 1)

            # Get new states from minus and plus depending on direction and build tree
            x_pm, v_pm, grad_x_pm = self.merge_states_dir(xminus, vminus, grad_xminus, xplus,
                                                          vplus, grad_xplus, direction)

            xminus2, vminus2, grad_xminus2, xplus2, vplus2, grad_xplus2, xprime2, vprime2, \
                numnodes2, stopped2, alpha2, nalpha2 = self.build_tree(x_pm, x1, v_pm, grad_x_pm,
                                                                       joint,
                                                                       logu, direction, stopped,
                                                                       depth,
                                                                       time_interval, detection)

            # Split the output back based on direction - keep the stopped samples the same
            idxminus = np.logical_and(np.logical_not(stopped), direction < 0).astype(int)
            xminus = State(state_vector=StateVectors((idxminus * xminus2.state_vector.T
                                                      + (1 - idxminus) * xminus.state_vector.T).T))
            vminus = idxminus * vminus2 + (1 - idxminus) * vminus
            grad_xminus = idxminus * grad_xminus2 + (1 - idxminus) * grad_xminus
            idxplus = np.logical_and(np.logical_not(stopped), direction > 0).reshape(self.num_samples,
                                                                                     1).astype(int)
            xplus = State(state_vector=StateVectors((idxplus * xplus2.state_vector.T +
                                                     (1 - idxplus) * xplus.state_vector.T).T))
            vplus = idxplus * vplus2 + (1 - idxplus) * vplus
            grad_xplus = idxplus * grad_xplus2 + (1 - idxplus) * grad_xplus

            # Update acceptance rate
            alpha = np.logical_not(stopped) * alpha2 + stopped * alpha
            nalpha = np.logical_not(stopped) * nalpha2 + stopped * nalpha

            # If no U-turn, choose new state
            samples = uniform.rvs(size=self.num_samples).reshape(-1, 1)
            u = numnodes.reshape(-1, 1) * samples < numnodes2.reshape(-1, 1)

            selectnew = np.logical_and(np.logical_not(stopped2), u).reshape(self.num_samples, 1).astype(int)

            xprime = State(state_vector=StateVectors((selectnew * xprime2.state_vector.T +
                                                      (1 - selectnew) * xprime.state_vector.T).T))
            vprime = selectnew * vprime2 + (1 - selectnew) * vprime

            # Update number of nodes and tree height
            numnodes = numnodes + numnodes2
            depth = depth + 1
            if depth > self.max_tree_depth:
                print("Max tree size in NUTS reached")
                break

            # Do U-turn test
            stopped = np.logical_or(stopped, stopped2)
            stopped = np.logical_or(stopped,
                                    self.stop_criterion_vec(xminus, xplus, vminus, vplus).reshape(
                                        -1, 1))

            acceptance = alpha / nalpha

        # we need to fix her ewith some stone soup states
        return xprime, vprime, acceptance

    def build_tree(self, x, x1, v, grad_x, joint, logu, direction, stopped, depth, time_interval,
                   detection):  # removed rng

        if depth == 0:

            # Base case
            # ---------

            not_stopped = np.logical_not(stopped)

            # Do leapfrog
            xprime2, vprime2, grad_xprime2 = self.integrate_lf_vec(x, x1, v, grad_x, direction,
                                                                   self.step_size,
                                                                   time_interval, detection)

            idx_notstopped = not_stopped.astype(int)

            xprime = State(state_vector=StateVectors(
                (idx_notstopped * xprime2.state_vector.T + (1 - idx_notstopped)
                 * x.state_vector.T).T))
            vprime = idx_notstopped * vprime2 + (1 - idx_notstopped) * v
            grad_xprime = idx_notstopped * grad_xprime2 + (1 - idx_notstopped) * grad_x

            # Get number of nodes
            logpprime = self.target_proposal(xprime, x, detection,
                                             time_interval=time_interval).reshape(-1, 1)
            jointprime = self.get_hamiltonian(vprime, logpprime)  # xprime
            numnodes = (logu <= jointprime).astype(int)

            # Update acceptance rate
            logalphaprime = np.where(jointprime > joint, 0.0, jointprime - joint)
            alphaprime = np.zeros((self.num_samples, 1))
            alphaprime[not_stopped] = np.exp(logalphaprime[not_stopped[:, 0], 0])
            alphaprime[np.isnan(alphaprime)] = 0.0
            nalphaprime = np.ones_like(alphaprime, dtype=int)

            # Stop bad samples
            stopped = np.logical_or(stopped, logu - self.delta_max >= jointprime)

            return xprime, vprime, grad_xprime, xprime, vprime, grad_xprime, xprime, vprime, \
                numnodes, stopped, alphaprime, nalphaprime

        else:

            # Recursive case
            # --------------

            # Build one subtree
            xminus, vminus, grad_xminus, xplus, vplus, grad_xplus, xprime, vprime, \
                numnodes, stopped, alpha, nalpha = self.build_tree(x, x1, v, grad_x, joint, logu,
                                                                   direction, stopped, depth - 1,
                                                                   time_interval,
                                                                   detection)  # removed rngs

            if np.any(stopped == 0):
                # Get new states from minus and plus depending on direction and build tree
                x_pm, v_pm, grad_x_pm = self.merge_states_dir(xminus, vminus, grad_xminus, xplus,
                                                              vplus, grad_xplus, direction)

                xminus2, vminus2, grad_xminus2, xplus2, vplus2, grad_xplus2, xprime2, vprime2, \
                    numnodes2, stopped2, alpha2, nalpha2 = self.build_tree(x_pm, x1, v_pm,
                                                                           grad_x_pm, joint,
                                                                           logu, direction,
                                                                           stopped, depth - 1,
                                                                           time_interval,
                                                                           detection)  # removed rngs

                # Split the output back based on direction - keep the stopped samples the same
                idxminus = np.logical_and(np.logical_not(stopped), direction < 0).astype(int)
                xminus = State(
                    state_vector=StateVectors((idxminus * xminus2.state_vector.T + (1 - idxminus)
                                               * xminus.state_vector.T).T))
                vminus = idxminus * vminus2 + (1 - idxminus) * vminus
                grad_xminus = idxminus * grad_xminus2 + (1 - idxminus) * grad_xminus
                idxplus = np.logical_and(np.logical_not(stopped), direction > 0).astype(int)
                xplus = State(
                    state_vector=StateVectors((idxplus * xplus2.state_vector.T + (1 - idxplus)
                                               * xplus.state_vector.T).T))
                vplus = idxplus * vplus2 + (1 - idxplus) * vplus
                grad_xplus = idxplus * grad_xplus2 + (1 - idxplus) * grad_xplus

                # Do new sampling
                u = numnodes.reshape(-1, 1) * uniform.rvs(size=self.num_samples).reshape(-1,
                                                                               1) < numnodes2.reshape(
                    -1, 1)

                selectnew = np.logical_and(np.logical_not(stopped2), u).reshape(self.num_samples, 1).astype(
                    int)
                xprime = State(state_vector=StateVectors((selectnew * xprime2.state_vector.T
                                                          + (
                                                                      1 - selectnew) * xprime.state_vector.T).T))
                vprime = selectnew * vprime2 + (1 - selectnew) * vprime

                # Do U-turn test
                stopped = np.logical_or(stopped, stopped2)
                stopped = np.logical_or(stopped, self.stop_criterion_vec(xminus, xplus, vminus,
                                                                         vplus).reshape(-1, 1))

                # Update number of nodes
                not_stopped = np.logical_not(stopped)
                numnodes = numnodes + numnodes2

                # Update acceptance rate
                alpha += not_stopped * alpha2
                nalpha += not_stopped * nalpha2
                #alpha[not_stopped] = alpha[not_stopped] + alpha2[not_stopped]
                #nalpha[not_stopped] = nalpha[not_stopped] + nalpha2[not_stopped]

            return xminus, vminus, grad_xminus, xplus, vplus, grad_xplus, xprime, \
                vprime, numnodes, stopped, alpha, nalpha

# ----------------------------------------------------------------------- #
## old stuff

# Set Maximum tree depth NUTS can take to stop excessive tree growth.
# self.max_tree_depth = 5  # ~ 10
# self.v_mapping = v_mapping
# self.mapping = mapping

# self.num_samples = N  # got it
# self.num_dims = D  # not needed
# self.target = p
# self.grad_target = grad_p  ## we need to pass it independently
#
# if np.isscalar(h):  ## OK
#     self.h = np.repeat(h, N)
# else:
#     self.h = h

# self.MM = np.tile(MM, (N, 1, 1))   # mass metric
# self.inv_MM = np.tile(np.linalg.inv(MM), (N, 1, 1))
# self.max_tree_depth = 10  ## ok
# self.delta_max = 100  ## ok


# def rvs(self, state, grad):
#     """ random variable state"""
#
#     return self.generate_NUTS_sample(state, self.mapping, self.v_mapping, grad)  # rng?

# def NUTSLeapfrog(self, x, v, grad_x, direction):
#     """
#     Performs a single Leapfrog step returning the final position, velocity and gradient.
#     """
#
#     v = np.add(v, (direction * self.h / 2) * grad_x)
#     x = np.add(x, direction * self.h * v)
#     grad_x = self.grad(x)
#     v = np.add(v, (direction * self.h / 2) * grad_x)
#
#     return x, v, grad_x

# def generate_NUTS_sample(self, state, mapping, v_mapping, grad_x):
#     """
#         Function that generates NUTS samples
#     """
#     # state or state/state vector
#     # joint lnp of x and momentum r
#     logp = self.transition_model.logpdf(state[self.mapping])  # maybe I should use the transition model? | target
#
#     self.H0 = logp - 0.5 * np.dot(state.state_vector[self.v_mapping],
#                                   state.state_vector[self.v_mapping].T)
#
#     logu = float(self.H0 - np.random.exponential(1))  ## rng changed
#
#     # INITIALISE THE TREE - Initialisation phase
#     # state, state minimal, state maximal
#     x, x_m, x_p = state.state_vector[self.mapping], state.state_vector[self.mapping], \
#                   state.state_vector[self.mapping]
#
#     # velocity
#     v, v_m, v_p = -state.state_vector[self.v_mapping], state.state_vector[self.v_mapping], \
#                   state.state_vector[self.v_mapping]
#
#     # gradients
#     gradminus, gradplus = grad_x, grad_x
#
#     # times
#     t, t_m, t_p = 0, 0, 0
#
#     depth = 0  # initial depth of the tree
#     n = 1  # Initially the only valid point is the initial point.
#     stop = 0  # Main loop: will keep going until stop == 1.
#
#     while stop == 0:  # loop my boy
#         # Choose a direction. -1 = backwards, 1 = forwards.
#         direction = int(2 * (np.random.uniform(0, 1) < 0.5) - 1)  #
#
#         if direction == -1:
#             x_m, v_m, gradminus, _, _, _, x_pp, v_pp, logpprime, nprime, stopprime, \
#             t_m, _, tprime = self.build_tree(x_m, v_m, gradminus, logu,
#                                              direction, depth, t_m, rng)  # rng?
#         else:
#             _, _, _, x_p, v_p, gradplus, x_pp, v_pp, logpprime, nprime, stopprime, \
#             _, t_p, tprime = self.build_tree(x_p, v_p, gradplus, logu,
#                                              direction, depth, t_p, rng)
#
#         # Use Metropolis-Hastings to decide whether to move to a point from the
#         # half-tree we just generated.
#         if stopprime == 0 and np.random.uniform() < min(1., float(nprime) / float(n)):
#             x = xprime
#             v = vprime
#             t = tprime
#
#         # Update number of valid points we've seen.
#         n += nprime
#
#         # Decide if it's time to stop.
#         stop = stopprime or self.stop_criterion(x_m, x_p, v_m, v_p)
#
#         # Increment depth.
#         depth += 1
#
#         if depth > self.max_tree_depth:
#             print("Max tree size in NUTS reached")
#             break
#
#     # maybe ?
#     final_state = np.zeros([state.shape])
#
#     final_state[self.mapping] = x
#     final_state[self.v_mapping] = v
#
#     return StateVector([final_state]), t  ## something

# def build_tree(self, x, v, grad_x, logu, direction, depth, t, rng):
#     """function to build the trees"""
#
#     if depth == 0:
#         xprime, vprime, gradprime = self.num_samplesUTSLeapfrog(x, v, grad_x, direction)
#         logpprime = self.transition_model.logpdf(xprime)
#         joint = logpprime - 0.5 * np.dot(vprime, vprime.T)
#         nprime = int(logu < joint)
#         stopprime = int((logu - 100.) >= joint)
#         xminus = xprime
#         xplus = xprime
#         vminus = vprime
#         vplus = vprime
#         gradminus = gradprime
#         gradplus = gradprime
#         tprime = t + self.h
#         tminus = tprime
#         tplus = tprime
#     else:
#         # Recursion: Implicitly build the height j-1 left and right subtrees.
#         xminus, vminus, gradminus, xplus, vplus, gradplus, xprime, vprime, logpprime, \
#         nprime, stopprime, tminus, tplus, tprime = self.build_tree(
#             x, v, grad_x, logu, direction, depth - 1, t, np.random)
#
#         # No need to keep going if the stopping criteria were met in the first subtree.
#         if stopprime == 0:
#             if direction == -1:
#                 xminus, vminus, gradminus, _, _, _, xprime2, vprime2, logpprime2, \
#                 nprime2, stopprime2, tminus, _, tprime2 = self.build_tree(xminus, vminus,
#                                                                           gradminus, logu, direction,
#                                                                           depth - 1, tminus, np.random)
#             else:
#                 _, _, _, xplus, vplus, gradplus, xprime2, vprime2, logpprime2, \
#                 nprime2, stopprime2, _, tplus, tprime2 = self.build_tree(xplus, vplus,
#                                                                          gradplus, logu, direction,
#                                                                          depth - 1, tplus, np.random)
#
#             if rng.uniform() < (float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.)):
#                 xprime = xprime2
#                 logpprime = logpprime2
#                 vprime = vprime2
#                 tprime = tprime2
#
#             # Update the number of valid points.
#             nprime = int(nprime) + int(nprime2)
#
#             # Update the stopping criterion.
#             stopprime = int(stopprime or stopprime2 or self.stop_criterion(xminus, xplus, vminus, vplus))
#
#     return xminus, vminus, gradminus, xplus, vplus, gradplus, xprime, vprime, \
#            logpprime, nprime, stopprime, tminus, tplus, tprime

# def stop_criterion(self, x_m, x_p, r_m, r_p):
#     """
#     Checks if a U-turn is present in the furthest nodes in the NUTS tree
#     """
#     return (np.dot((x_p - x_m), r_m.T) < 0) or \
#            (np.dot((x_p - x_m), r_p.T) < 0)
#
