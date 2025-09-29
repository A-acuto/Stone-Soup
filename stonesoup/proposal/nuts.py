import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import uniform
from scipy.special import logsumexp
from copy import deepcopy, copy

from stonesoup.base import Property
from stonesoup.types.array import StateVectors
from stonesoup.types.state import State
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.transition import TransitionModel
from stonesoup.proposal.base import Proposal
from stonesoup.types.detection import Detection
from stonesoup.types.prediction import Prediction

# optimiser
from stonesoup.proposal.base import Optimiser

class NUTSProposal(Proposal):
    """No-U Turn Sampler proposal

        This implementation follows the papers:
        [1] Varsi, A., Devlin, L., Horridge, P., & Maskell, S. (2024).
        A general-purpose fixed-lag no-u-turn sampler for nonlinear non-gaussian
        state space models. IEEE Transactions on Aerospace and Electronic Systems.

        [2] Devlin, L., Horridge, P., Green, P. L., & Maskell, S. (2021).
        The No-U-Turn sampler as a proposal distribution in a sequential Monte Carlo
        sampler with a near-optimal L-kernel.
        arXiv preprint arXiv:2108.02498.
    """

    transition_model: TransitionModel = Property(
        doc="The transition model used to make the prediction")
    measurement_model: MeasurementModel = Property(
        doc="The measurement model used to evaluate the likelihood")
    step_size: float = Property(doc='Step size used in the LeapFrog calculation')
    mass_matrix: float = Property(doc='Mass matrix needed for the Hamilitonian equation')
    mapping: tuple = Property(doc="Localisation mapping")
    v_mapping: tuple = Property(doc="Velocity mapping")
    num_dims: int = Property(doc='State dimension')
    num_samples: int = Property(doc='Number of samples')
    target_proposal_input: float = Property(
        doc='Particle distribution',
        default=None)
    grad_target: float = Property(
        doc='Gradient of the particle distribution',
        default=None)
    max_tree_depth: int = Property(
        doc="Maximum tree depth NUTS can take to stop excessive tree growth.",
        default=25)
    delta_max: int = Property(
        doc='Rejection criteria threshold',
        default=100)
    optimise: bool = Property(
        doc='Optimise the stepsize and mass matrix',
        default=True)
    num_iterations: int = Property(doc='Number of iterations for the NUTS proposal',
                                   default=1)

    # Initialise
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MM = np.tile(self.mass_matrix, (self.num_samples, 1, 1))  # mass matrix
        self.inv_MM = np.tile(np.linalg.pinv(self.mass_matrix), (self.num_samples, 1, 1))
    
        # Ensure step size is an array
        if np.isscalar(self.step_size):
            self.step_size = np.repeat(self.step_size, self.num_samples).reshape(1, -1)

        if self.optimise:
            self.optimiser = NUTS_optimiser(sampler=self)
            # in here we can also instantiate the optimiser with the parameters

    def target_proposal(self, prior_state, new_state_prediction, detection,
                        time_interval):
        """Target proposal"""

        tx_logpdf = self.transition_model.logpdf(new_state_prediction, prior_state, 
                                                 time_interval=time_interval,
                                                 allow_singular=True)
        mx_logpdf = self.measurement_model.logpdf(detection, new_state_prediction, 
                                                  allow_singular=True)
        tg_proposal = tx_logpdf + mx_logpdf

        return tg_proposal

    def grad_target_proposal(self, prior_state, new_state_prediction, 
                             detection, time_interval, **kwargs):
        
        """Gradient of the target proposal distribution"""

        dx = new_state_prediction.state_vector - self.transition_model.function(prior_state,
                                                                 time_interval=time_interval,
                                                                 **kwargs)

        
        tx_covar = self.transition_model.covar(time_interval=time_interval)
        mx_covar = self.measurement_model.covar()

        grad_log_prior = (np.linalg.pinv(tx_covar) @ (-dx)) #.T

        # temporary fix to make the jacobian work with particle state:
        # worth understanding if it is the better choice overall
    #    temp_x = State(state_vector=state.mean,
    #                   timestamp=state.timestamp)

        # Get Jacobians of measurements
        H = self.measurement_model.jacobian(new_state_prediction)

        # Get innov
        dy = detection.state_vector - self.measurement_model.function(new_state_prediction, 
                                                                      **kwargs)

        # Compute the gradient H^T * inv(R) * innov
        if len(H.shape) < 3:
            # Single Jacobian matrix
            grad_log_pdf = (H.T @ np.linalg.pinv(mx_covar) @ dy) # .T
        else:
            # Jacobian matrix for each point
            HTinvR = H.transpose((0, 2, 1)) @ self.linalg.pinv(mx_covar)
            grad_log_pdf = (HTinvR @ np.atleast_3d(dy))[:, :, 0]

        return grad_log_prior + grad_log_pdf

    def rvs(self, state, measurement: Detection = None, time_interval=None,
            **kwargs):

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

#        # evaluate the momentum
#        v = mvn.rvs(mean=np.zeros(self.num_dims), cov=self.MM[0], size=self.num_samples)

        if measurement is not None:

            # first instance
            iter_state = copy(state)
            
            # iterate multiple times the nuts proposal
            for i in range(self.num_iterations): 

                # evaluate the momentum
                v = mvn.rvs(mean=np.zeros(self.num_dims), cov=self.MM[0], size=self.num_samples).T

                # evalute the gradient
                grad_x = self.grad_target_proposal(iter_state, new_state_pred, measurement, time_interval)

                # needed to the optimiser  / or not
                # get_grad = self.target_proposal(state, new_state_pred, measurement, time_interval)
                # is this overall used?

<<<<<<< Updated upstream
            # in this case we might need to pass the optimiser
            # if self.optimise:
=======
            # we need the jacobian
            jk_minus = self.integrate_lf_vec(x_new, state, v, grad_x, 1, self.step_size, time_interval, measurement)
            jk_plus = self.integrate_lf_vec(x_new, state, v_new, grad_x, -1, self.step_size, time_interval, measurement)
>>>>>>> Stashed changes

            #     # when optimiser drop this, otherwise just use it
            #     print(self.step_size, self.step_size.shape, 'before')
            #     a = self.optimiser._run_smc_adaptive(state, new_state_pred, grad_x, v,
            #                                         time_interval, measurement)
            #     print(a, a.shape, 'after')
            #     sys.exit()
            # how can we pass the
#            else:
                x_new, v_new, acceptance = self.generate_nuts_samples(iter_state, new_state_pred,
                                                                    v, grad_x, measurement,
                                                                    time_interval)
                print(np.mean(acceptance))

<<<<<<< Updated upstream
                # print(np.isclose(x_new.state_vector, state.state_vector), 'after')
                if i == self.num_iterations:
                    x_new.timestamp = timestamp
                    # evalute the gradient
                    grad_x = self.grad_target_proposal(x_new, new_state_pred, 
                                                       measurement, time_interval)
                else:
                    iter_state = x_new
=======
            determinant_m = np.linalg.det(j_cab_m)
            determinant_p = np.linalg.det(j_cab_p)

            # qv I try with logpdf
            q_star_minus = mvn.logpdf(-v_new, mean=np.zeros(v.shape[1]), cov=self.mass_matrix)/determinant_m
            #qv (-v)/det(J)
            q_star_plus =mvn.logpdf(v, mean=np.zeros(v.shape[1]), cov=self.mass_matrix)/determinant_p
            #qv (v)/det(J)
            # print(np.mean(q_star_minus), np.mean(q_star_plus), '|||', np.mean(pi_x_k), np.mean(pi_x_k1))

            # maybe not needed these below
            # L-kernel
            #L = 1
            # q(x_k|x_k-1)
            #q = 1
>>>>>>> Stashed changes
        else:
            # No updates
            x_new = new_state_pred
            pi_x_k = 0
            pi_x_k1 = 0
            q_star_minus = 0
            q_star_plus = 0

        # pi(x_k)  # state is like prior
        pi_x_k = self.target_proposal(previous_state, x_new, measurement, time_interval)

        # pi(x_k-1)
        pi_x_k1 = self.target_proposal(previous_state, previous_state, measurement, time_interval)

        # we need to add the deteminat of the jacobian of the LF integrator
        # following eq 22 in Alessandro's papers
        # wt = wt-1 * (pi_x_k * qv(-v)/det(J))/(pi_x_k1 * p_x_xk1 * qv(v)/det(J))
        # 1/-1 id the direction

        jk_minus = self.integrate_lf_vec(previous_state, x_new, v, grad_x, 1, 
                                            self.step_size,
                                            time_interval, measurement)
        jk_plus = self.integrate_lf_vec(previous_state, x_new, v_new, grad_x, -1, 
                                        self.step_size,
                                        time_interval, measurement)

        j_cab_m = self.get_grad(jk_minus, time_interval)
        j_cab_p = self.get_grad(jk_plus, time_interval)

        determinant_m = np.linalg.det(j_cab_m)
        determinant_p = np.linalg.det(j_cab_p)

        # qv (-v)/det(J)
        q_star_minus = mvn.logpdf(-(v_new).T, mean=np.zeros(v.shape[0]),
                                    cov=self.mass_matrix)/determinant_m
        # qv (v)/det(J)
        q_star_plus = mvn.logpdf((v).T, mean=np.zeros(v.shape[0]),
                                    cov=self.mass_matrix)/determinant_p
                
                # iter_state = x_new
                # iter_state.timestamp = state.timestamp

                # iter_state = Prediction.from_state(previous_state,
                #                             parent=previous_state,
                #                             state_vector=x_new.state_vector,
                #                             timestamp=previous_state.timestamp,
                #                             transition_model=self.transition_model,
                #                             prior=previous_state)
                
#                iter_state.log_weight += pi_x_k - pi_x_k1 + q_star_minus - q_star_plus
                #print(iter_state.log_weight, 'log weight')


        final_state = Prediction.from_state(previous_state,
                                            parent=previous_state,
                                            state_vector=x_new.state_vector,
                                            timestamp=timestamp,
                                            transition_model=self.transition_model,
                                            prior=previous_state)

        final_state.log_weight += pi_x_k - pi_x_k1 + q_star_minus - q_star_plus

        return final_state

    def get_grad(self, new_state, time_interval):
        """Use the Jacobian of the model"""
        return self.transition_model.jacobian(new_state, time_interval=time_interval)

    def integrate_lf_vec(self, state, new_state_pred, v, grad_x, direction, h, time_interval,
                         measurement):
        """Leapfrog integration"""

        if len(h.shape) == 1:
            h = h.reshape(self.num_samples, 1)

        # maybe this can be propagated nicely
        # else:
        #     # this is a botch solution to avoid generation
        #     # of wrong dimension of the stepsize
        #     # h = h[0,:].reshape(self.num_samples, 1)
        #     continue
     
        # copy the state
        temp_state = copy(state)

        v = v + direction * (h / 2.) * grad_x
        einsum = np.einsum('bij,jb->ib', self.inv_MM, v)

        temp_state.state_vector = (temp_state.state_vector + direction * h * einsum)
        grad_x = self.grad_target_proposal(temp_state, new_state_pred, measurement, time_interval)
        v = v + direction * (h / 2.) * grad_x

        return temp_state, v, grad_x

    def stop_criterion_vec(self, xminus, xplus, rminus, rplus):
        """Stop Criterion"""

        # Return True for particles we want to stop (NB opposite way round to s in Hoffman
        # and Gelman paper)
        dx = xplus.state_vector - xminus.state_vector
        left = (np.sum(dx * np.einsum('bij,jb->ib', self.inv_MM, rminus), axis=0) < 0)
        right = (np.sum(dx * np.einsum('bij,jb->ib', self.inv_MM, rplus), axis=0) < 0)
        return np.logical_or(left, right)

    def get_hamiltonian(self, v, logp):
        """Hamiltonian calculation"""

        # Get Hamiltonian energy of system given log target weight logp
        return logp - 0.5 * np.sum(v * np.einsum('bij,jb->ib', self.inv_MM, v),
                                   axis=0) # .reshape(-1, 1)

    def merge_states_dir(self, xminus, vminus, grad_xminus, xplus, vplus, grad_xplus, direction):
        """ Auxiliary function to merge the states"""

        # Return xmerge = vectors of xminus where direction < 0 and xplus where direction > 0, and
        # similarly for v and grad_x
        merge_state = copy(xminus)
        mask = direction < 0
        mask = mask.reshape(1, mask.shape[1]).astype(int)
        xmerge = (mask * xminus.state_vector + (1 - mask) * xplus.state_vector)
        vmerge = (mask * vminus + (1 - mask) * vplus)
        grad_xmerge = (mask * grad_xminus + (1 - mask) * grad_xplus)
        merge_state.state_vector = xmerge
        return merge_state, vmerge, grad_xmerge

    def generate_nuts_samples(self, x0, x1, v0, grad_x0, detection, time_interval):
        """ Generate NUTS samples"""

        # Sample energy: note that log(U(0,1)) has same distribution as -exponential(1)
        logp0 = self.target_proposal(x0, x1, detection,
                                     time_interval=time_interval).reshape(1, -1)
        joint = self.get_hamiltonian(v0, logp0)
        logu = joint + np.log(uniform.rvs())

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

        # criteria
        stopped = np.zeros((1, self.num_samples)).astype(bool)
        numnodes = np.ones((1, self.num_samples)).astype(int)

        # Used to compute acceptance rate
        alpha = np.zeros((1, self.num_samples))
        nalpha = np.zeros((1, self.num_samples)).astype(int)
        cc = 0
        
        while np.any(stopped == 0):

            # Generate random direction in {-1, +1}
            direction = (2 * (uniform.rvs(0, 1, size=self.num_samples)
                              < 0.5).astype(int) - 1).reshape(1, -1)

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
            xminus = State(state_vector=StateVectors(idxminus * xminus2.state_vector
                                                      + (1 - idxminus) * xminus.state_vector),)
            # xminus and xplus should work
            vminus = idxminus * vminus2 + (1 - idxminus) * vminus
            grad_xminus = idxminus * grad_xminus2 + (1 - idxminus) * grad_xminus
            idxplus = np.logical_and(np.logical_not(stopped), direction > 0).astype(int)
            xplus = State(state_vector=StateVectors(idxplus * xplus2.state_vector +
                                                    (1 - idxplus) * xplus.state_vector),)
            vplus = idxplus * vplus2 + (1 - idxplus) * vplus
            grad_xplus = idxplus * grad_xplus2 + (1 - idxplus) * grad_xplus

            # make it easier
            not_stopped = np.logical_not(stopped)
            not_stopped2 = np.logical_not(stopped2)

            # Update acceptance rate
            alpha = not_stopped * alpha2 + stopped * alpha
            nalpha = not_stopped * nalpha2 + stopped * nalpha

            # If no U-turn, choose new state
            samples = uniform.rvs(size=self.num_samples).reshape(1, -1)
            u = numnodes * samples < numnodes2

            selectnew = np.logical_and(not_stopped2, u).reshape(1, self.num_samples).astype(int)

            # print(np.isclose(xprime2.state_vector, xprime.state_vector))
            xprime = State(state_vector=StateVectors(selectnew * xprime2.state_vector +
                                                     (1 - selectnew) * xprime.state_vector),)
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
                                    self.stop_criterion_vec(xminus, xplus, vminus, vplus))
           
            cc += 1
        print(f'couts {cc}')
        #sys.exit()
        acceptance = alpha / nalpha

        return xprime, vprime, acceptance

    def build_tree(self, prev_state, new_state, v, grad_x, joint, logu, 
                   direction, stopped, depth, time_interval,
                   detection):
        """Function to build the particle trees"""

        if depth == 0:

            # Base case
            # ---------

            # auxiliary bits
            not_stopped = np.logical_not(stopped)
            idx_notstopped = not_stopped.astype(int)

            # Do leapfrog
            xprime_temp2, vprime2, grad_xprime2 = self.integrate_lf_vec(prev_state, new_state, v, 
                                                                        grad_x, direction,
                                                                        self.step_size,
                                                                        time_interval, detection)

            # new xprime state
            xprime = State(state_vector=StateVectors(
                (idx_notstopped * xprime_temp2.state_vector + 
                 (1 - idx_notstopped)* prev_state.state_vector)),)
#                 timestamp=prev_state.timestamp)
            # new momentum
            vprime = idx_notstopped * vprime2 + (1 - idx_notstopped) * v
            # new gradient
            grad_xprime = idx_notstopped * grad_xprime2 + (1 - idx_notstopped) * grad_x

            # Get number of nodes
            logpprime = self.target_proposal(xprime, prev_state, detection,
                                             time_interval=time_interval).reshape(1, -1)
            jointprime = self.get_hamiltonian(vprime, logpprime)
            numnodes = (logu <= jointprime).astype(int)

            # Update acceptance rate
            logalphaprime = np.where(jointprime > joint, 0.0, jointprime - joint)
            alphaprime = np.zeros((1, self.num_samples))
            alphaprime[not_stopped] = np.exp(logalphaprime[not_stopped])
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
                numnodes, stopped, alpha, nalpha = self.build_tree(prev_state, new_state, v, 
                                                                   grad_x, joint, logu,
                                                                   direction, stopped, depth - 1,
                                                                   time_interval,
                                                                   detection)

            if np.any(stopped == 0):
                # Get new states from minus and plus depending on direction and build tree
                x_pm, v_pm, grad_x_pm = self.merge_states_dir(xminus, vminus, grad_xminus, xplus,
                                                              vplus, grad_xplus, direction)

                xminus2, vminus2, grad_xminus2, xplus2, vplus2, grad_xplus2, xprime2, vprime2, \
                    numnodes2, stopped2, alpha2, nalpha2 = self.build_tree(x_pm, new_state, v_pm,
                                                                           grad_x_pm, joint,
                                                                           logu, direction,
                                                                           stopped, depth - 1,
                                                                           time_interval,
                                                                           detection)

                not_stopped = np.logical_not(stopped)
                # Split the output back based on direction - keep the stopped samples the same
                idxminus = np.logical_and(not_stopped, direction < 0).astype(int)
                xminus = State(
                    state_vector=StateVectors(idxminus * xminus2.state_vector + (1 - idxminus)
                                            * xminus.state_vector))
                vminus = idxminus * vminus2 + (1 - idxminus) * vminus
                grad_xminus = idxminus * grad_xminus2 + (1 - idxminus) * grad_xminus
                idxplus = np.logical_and(not_stopped, direction > 0).astype(int)
                xplus = State(
                    state_vector=StateVectors((idxplus * xplus2.state_vector + (1 - idxplus)
                                               * xplus.state_vector)))
                vplus = idxplus * vplus2 + (1 - idxplus) * vplus
                grad_xplus = idxplus * grad_xplus2 + (1 - idxplus) * grad_xplus

                # Do new sampling
                samples = uniform.rvs(size=self.num_samples).reshape(1, -1)
                u = numnodes * samples < numnodes2

                selectnew = np.logical_and(np.logical_not(stopped2), u).reshape(1, 
                                                                                self.num_samples).astype(int)
                xprime = State(state_vector=StateVectors(selectnew * xprime2.state_vector
                                                         + (1 - selectnew) *
                                                         xprime.state_vector))
                vprime = selectnew * vprime2 + (1 - selectnew) * vprime

                # Do U-turn test
                stopped = np.logical_or(stopped, stopped2)
                stopped = np.logical_or(stopped, self.stop_criterion_vec(xminus, xplus, vminus,
                                                                         vplus))

                # Update number of nodes
                not_stopped = np.logical_not(stopped)
                numnodes = numnodes + numnodes2

                # Update acceptance rate
                alpha += not_stopped * alpha2
                nalpha += not_stopped * nalpha2

            return xminus, vminus, grad_xminus, xplus, vplus, grad_xplus, xprime, \
                vprime, numnodes, stopped, alpha, nalpha


# lets see if this work
class NUTS_optimiser(Optimiser):
    """
    Preprocessing to adapt the Stepsize and Mass matrix to optimal values.
    """

    # criteria
    t0: int = Property(doc='', default=10)
    gamma: float = Property(doc='', default=0.05)
    delta: float = Property(doc='', default=0.8)
    # min_acceptance: float = Property(doc='', default=0.4)  # oddly out
    # max_acceptance: float = Property(doc='', default=0.9)  # oddly out
    #num_dims: int = Property(doc='Number of dimension')
    #lf_integral: int = Property(doc='Leap frog integrator function, it is not an integer')
    sampler: Proposal = Property(doc='Sampler')  # this is where we are passing
    # In this case we have it already in the class state

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.sampler =

        self.num_dims = self.sampler.num_dims
        self.massMatrix = self.sampler.mass_matrix

    # to have it so it does not complain
    def rvs(self):
        print('helolo')

    def _run_smc_adaptive(self, init_states, samples_block, get_grad, v, time_interval, measurement):  # removed state_dim

        # samples block is the new state pred

        # use the sampler number of dimension  [maybe]
        # state_dim = self.sampler.num_dims  # not needed  # num dims does not work.

        # This SMC sampler runs multiple iterations and adaptively tunes the step size and mass matrix
        # Adapted to consider a single rolling window to hopefully speed things up

        # log weights can be accessed at any time

        # since states are composed we can get some info out
        logweights_prevk = init_states.log_weight
        logweights = samples_block.log_weight

        N = logweights.shape[0]  # This N?  # we can get it alreaduy

        # Set up step sizes (using x_{k-L:k} where x_{k-L-1} is in log_target)
        self.h = self._get_initial_step_size(init_states, samples_block, get_grad, measurement,
                                             time_interval=time_interval)  # I'd remove self rngs, sample block is a state

        # nowself.h looks ok
        mu = np.log(10 * self.h)
        Hbar = np.zeros_like(self.h)

        tolerances = [0.5, 0.4, 0.2, 0.1]  # tolerance of standard deviation of step sizes
        window_size = [5, 3, 3, 3]  # windows to evaluate step size std
        max_iterations = 50  # maximum number of iterations total

        # window size?
        max_window_size = max(window_size)
        tol_ptr = 0  # index of tolerance we are considering
        num_iterations = 0
        array_of_steps = [np.inf for _ in range(max_window_size)]

        while num_iterations < max_iterations and tol_ptr < len(tolerances):
            # here we are:

            # type particle, array, array, time delta
            init_states, samples_block, logweights, logweights_prevk, acceptance = self._iterate_static_smc(
                init_states, v, samples_block, get_grad, measurement, time_interval, logweights, logweights_prevk)
            logweightsum = logsumexp(logweights)
            neff = ess(logweights - logweightsum)

            # consider check
            # print(" - iteration ", num_iterations + 1, " mean acceptance ", np.mean(acceptance),
            #       'neff', neff)

            # Adapt step size and add to window
            Hbar = self._adapt_step_size(acceptance.reshape(-1), Hbar, mu, num_iterations)
            mean_stepsize = np.exp(logweights - logweightsum).T @ self.h
            array_of_steps = array_of_steps[1:] + [mean_stepsize]
            print(neff, N/2, mean_stepsize) # Ideally this does that. I'd skip for now

            # ignore resampling for the time being
            # Resample if necessary
            if neff <= N / 2:
                idx = self._resample_indices(logweights - logweightsum)  # pick one
                #print(idx)
               # print(logweights)
              #  print(logweights[idx])
              #  sys.exit()
                init_states = init_states[idx]

                # resample the indeces
                # logweights = logweightsum + np.zeros_like(logweights) - np.log(N)
                logweights_prevk = logweights_prevk[idx]
                Hbar = Hbar[idx]
                mu = mu[idx]
                self.h = self.h[idx]

#            print(mean_stepsize, array_of_steps, tolerances)
#            print(window_size, window_size[tol_ptr], array_of_steps[-window_size[tol_ptr]:])
            this_window = array_of_steps[-window_size[tol_ptr]:]
            relative_std = np.abs(np.std(this_window) / np.mean(this_window))
            print(relative_std < tolerances[tol_ptr])
#            print('arrr', array_of_steps )
#            sys.exit()
            # If enough iterations and effective sample size iss sufficiently high:
            num_iterations += 1
            if (num_iterations >= window_size[tol_ptr]) and (neff > N / 2):

                # Get std of step size samples normalised by mean
                this_window = array_of_steps[-window_size[tol_ptr]:]
                relative_std = np.abs(np.std(this_window) / np.mean(this_window))
                self._adapt_mass_matrix(samples_block, logweights - logweightsum)
                print(neff)
                sys.exit()
                # If relative std within current tolerance:
                if relative_std < tolerances[tol_ptr]:
                    # Move to next tolerance
                    tol_ptr += 1
                    if tol_ptr < len(tolerances):
                        # If not at end, advance tolerance pointer while we are still within the tolerance
                        self._adapt_mass_matrix(samples_block, logweights - logweightsum)
                        while tol_ptr < len(tolerances):
                            this_window = array_of_steps[-window_size[tol_ptr]:]
                            relative_std = np.abs(np.std(this_window) / np.mean(this_window))
                            if relative_std < tolerances[tol_ptr]:
                                tol_ptr += 1
                            else:
                                break
        sys.exit()
        return self.h #init_states, samples_block, logweights, logweights_prevk

    def _adapt_step_size(self, acceptance, Hbar, mu, k):
        """
        this works fine assuming the acceptance comes from

        Acceptance comes from nuts
        however if acceptances comes with NANS it influences
        the overall bits
        """

        # check acceptance
        acceptance = np.where(np.isnan(acceptance), 0., acceptance)

        eta = 1. / float(k + self.t0)
        Hbar = (1. - eta) * Hbar + eta * (self.delta - acceptance)
        self.h = np.exp(mu - np.sqrt(k) / self.gamma * Hbar)
        return Hbar

    # mass matrix
    def _adapt_mass_matrix(self, x, logw):
        """
        mass matrix adaptation
        """

        # Estimate mean and (diagonal) variance of samples
        _x = x.state_vector.copy().T   # T to fix the size
        wn = np.exp(logw) # - logsumexp(logw))  ## why logsumexp?
        x_shift = _x - (wn.T @ _x)
        var = wn.T @ np.square(x_shift)

        # (PRH: do we need this?) only take variance of samples
        # in space we are moving in i.e. no transformed parameters
        metric = 1. / var  # set metric to 1/var

        # Set metric parameters where needed, i.e. in forwards proposal and for weight update eq.s
        self.massMatrix = np.diag(metric)

    def _resample_indices(self, logweights):

        # Resample if necessary
        N = logweights.shape[0]
        i = np.linspace(0, N - 1, N, dtype=int)

        # PRH: Maybe add this to algorithms.random.RNG class later
        i_new = np.random.choice(i, N, p=np.exp(normalise(logweights)))  # p is the weight
        return i_new

    def _get_initial_step_size(self, init_state, current_state, get_grad, measurement,
                               time_interval=None):  # !!! X needs to be a state
        start_step = self.sampler.step_size  # ok this is the same one

        delta_H = self._check_step_size(init_state, current_state, get_grad, start_step,
                                       measurement, time_interval)

        init_direction = np.where(delta_H > np.log(0.5), 1, -1)
        direction = np.copy(init_direction)

        while (True):
            delta_H = self._check_step_size(init_state, current_state, get_grad, start_step,
                                            measurement, time_interval)

            # Deal with nans in delta_H
            direction = np.where(np.isnan(delta_H), -1, direction)
            delta_H = np.where(np.isnan(delta_H), -np.inf, delta_H)

            # Check particles to stop or which have changed direction
            done = np.logical_or(direction==0, #np.abs(direction) <= 1e-10,
                                 np.logical_or(np.logical_and(direction == -1, delta_H >= np.log(0.5)),
                                               np.logical_and(direction == 1, delta_H <= np.log(0.5))))
            direction = np.where(done, 0.0, direction)

            # Adjust step size - if not done, double or halve
            start_step = np.where(direction == 1, 2.0 * start_step, start_step)
            start_step = np.where(direction == -1, 0.5 * start_step, start_step)

            # If all done, we're finished
            if np.all(done):
                break

        # If done, halve the doubling ones back to their previous iteration
        start_step = np.where(init_direction == 1, 0.5 * start_step, start_step) # moved from init_direction

        return start_step

    def _iterate_static_smc(self, init_states, v, new_state_pred, get_grad, measurement,
                            time_interval, logweights, logweights_prevk):
        """
        Simulate a run of the SMC to evaluate the various parameters and
        make the correct assumptions on the stepsize and mass matrix
        """

        # removed because I don't use these
        # state_dim = self.num_dims    # self.model_list[0].ndim_state
        # block_dim = state_dim * len(self.model_list)   # mmm
        # num_samples = init_states.state_vector.shape[1]   # num samples

        # proposed using NUTS
        x_new, v_new, acceptance = self.sampler.generate_nuts_samples(init_states, new_state_pred,
                                                                      v, get_grad, measurement,
                                                                      time_interval)

        # update weights according to the optimiser  # pi_x_k
        log_target = self.sampler.target_proposal(x_new, init_states, measurement, time_interval)

        # maybe this is not needed here
        # pi(x_k-1)
        # pi_x_k1 = self.target_proposal(init_states, init_states, measurement, time_interval)

        # Update weights

        # evaluates the dterminants  - might not needed in the FL cases
        jk_minus = self.sampler.integrate_lf_vec(x_new, init_states, v, get_grad, 1, self.h,
                                         time_interval, measurement)
        jk_plus = self.sampler.integrate_lf_vec(x_new, init_states, v_new, get_grad, -1, self.h,
                                        time_interval, measurement)

        j_cab_m = self.sampler.get_grad(jk_minus, time_interval)
        j_cab_p = self.sampler.get_grad(jk_plus, time_interval)

        determinant_m = np.linalg.det(j_cab_m)
        determinant_p = np.linalg.det(j_cab_p)

        # this is the same as pi-x-k
        # log_target = self.sampler.target(new_samples_block)  # block_model.log_posterior(new_samples_block)
        #lkernel_logpdf = mvn(mean=np.zeros(block_dim), cov=self.massMatrix).logpdf(np.multiply(-1, new_v))
        lkernel_logpdf = mvn.logpdf(-v_new, mean=np.zeros(v.shape[1]), cov=self.massMatrix) / determinant_m
        #q_logpdf = mvn(mean=np.zeros(block_dim), cov=self.massMatrix).logpdf(v)
        q_logpdf = mvn.logpdf(v, mean=np.zeros(v.shape[1]), cov=self.massMatrix) / determinant_p

        # PRH: Removed normalisation for likelihood estimation
        # logweights = normalise(logweights + log_target - logweights_prevk + lkernel_logpdf - q_logpdf)
        logweights += log_target - logweights_prevk + lkernel_logpdf - q_logpdf
        logweights_prevk = log_target

        prediction = Prediction.from_state(init_states,
                                           parent = init_states,
                                           state_vector = x_new.state_vector,
                                           timestamp = new_state_pred.timestamp,
                                           transition_model = self.sampler.transition_model,
                                           prior = init_states)


        return init_states, prediction, logweights, logweights_prevk, acceptance

    # so this function needs to be called a few times because H1 changes and the
    # differece lies in the r_prime
    def _check_step_size(self, init_state, current_state, get_grad, step_size,
                         measurement, time_interval):  # X is a state removed rnf

        # samples a new momentum
        #r = np.transpose(mvn.rvs(mean=np.zeros(self.num_dims), cov=self.sampler.MM[0]))
        r = mvn.rvs(mean=np.zeros(self.num_dims), cov=self.sampler.MM[0],
                    size=self.sampler.num_samples)

        # Take a leapfrog step with momentum r
        # integrate has new state, state, r or velovituy, ones, stepsize, time interval, and measuremnet detection
        x_prime, r_prime, _ = self.sampler.integrate_lf_vec(current_state, init_state, r, get_grad,
                                                            np.ones((self.sampler.num_samples, 1)),  # direction
                                                            step_size,  # stepsize
                                                            time_interval,
                                                            measurement)

        # so this is the nasty bit. we got here the target proposal
        # X0/x1 prior and detection time interval
        # Calculate the (log) Hamiltonians at the end and start of the proposal

        common_value = self.sampler.target_proposal(current_state, init_state, measurement,
                                                    time_interval=time_interval)

        # removed the reshape (-1,1)
        H0 = common_value - (0.5 * np.sum(r * r, axis=1))
        H1 = common_value - (0.5 * np.sum(r_prime * r_prime, axis=1))

        # calculate the difference in the Hamiltonian
        delta_H = H1 - H0
        return delta_H

## functions extra
def ess(logweights):
    """Function to evaluate the effective sample size"""

    mask = np.invert(np.isneginf(logweights))

    inverse_neff = np.sum(np.exp(2*logweights)) #np.exp(logsumexp(2 * logweights[mask]))

    return 1 / inverse_neff


def normalise(logweights):
    """Function to normalise the weights"""

    mask = np.invert(np.isneginf(logweights))

    log_wsum = logsumexp(logweights[mask])

    return logweights - log_wsum



class NUTSProposalold(Proposal):
    """No-U Turn Sampler proposal
        This implementation follows the papers:
        [1] Varsi, A., Devlin, L., Horridge, P., & Maskell, S. (2024).
        A general-purpose fixed-lag no-u-turn sampler for nonlinear non-gaussian
        state space models. IEEE Transactions on Aerospace and Electronic Systems.
        [2] Devlin, L., Horridge, P., Green, P. L., & Maskell, S. (2021).
        The No-U-Turn sampler as a proposal distribution in a sequential Monte Carlo
        sampler with a near-optimal L-kernel.
        arXiv preprint arXiv:2108.02498.
    """

    transition_model: TransitionModel = Property(
        doc="The transition model used to make the prediction")
    measurement_model: MeasurementModel = Property(
        doc="The measurement model used to evaluate the likelihood")
    step_size: float = Property(doc='Step size used in the LeapFrog calculation')
    mass_matrix: float = Property(doc='Mass matrix needed for the Hamilitonian equation')
    mapping: tuple = Property(doc="Localisation mapping")
    v_mapping: tuple = Property(doc="Velocity mapping")
    num_dims: int = Property(doc='State dimension')
    num_samples: int = Property(doc='Number of samples')
    target_proposal_input: float = Property(
        doc='Particle distribution',
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
    num_iterations: int = Property(
        doc='Number of iterations to run the NUTS algorithm',
        default=10)

    # Initialise
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MM = np.tile(self.mass_matrix, (self.num_samples, 1, 1))  # mass matrix
        self.inv_MM = np.tile(np.linalg.pinv(self.mass_matrix), (self.num_samples, 1, 1))
        # self.num_iterations = num_iterations

        # Ensure step size is an array
        if np.isscalar(self.step_size):
            self.step_size = np.repeat(self.step_size, self.num_samples)

    def target_proposal(self, prior, state, detection,
                        time_interval):
        """Target proposal"""

        tx_logpdf = self.transition_model.logpdf(state, prior, time_interval=time_interval,
                                                 allow_singular=True)
        mx_logpdf = self.measurement_model.logpdf(detection, state, allow_singular=True)
        tg_proposal = tx_logpdf + mx_logpdf

        return tg_proposal

    def grad_target_proposal(self, prior, state, detection, time_interval, **kwargs):

        # grad log prior
        dx = state.state_vector - self.transition_model.function(prior,
                                                                 time_interval=time_interval,
                                                                 **kwargs)

        grad_log_prior = (
            np.linalg.pinv(self.transition_model.covar(time_interval=time_interval)) @ (-dx)
        ).T

        # temporary fix to make the jacobian work with particle state:
        # worth understanding if it is the better choice overall
        temp_x = State(state_vector=state.mean,
                       timestamp=state.timestamp)
        # Get Jacobians of measurements
        H = self.measurement_model.jacobian(temp_x)

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
        
        if measurement is not None:
            iter_state = state

            for i in range(self.num_iterations):

                # evaluate the momentum
                v = mvn.rvs(mean=np.zeros(self.num_dims), cov=self.MM[0], size=self.num_samples)

                # evaluate the gradient 
                grad_x = self.grad_target_proposal(iter_state, new_state_pred, measurement, time_interval)

                x_new, v_new, acceptance = self.generate_nuts_samples(iter_state, new_state_pred,
                                                                      v, grad_x, measurement, 
                                                                      time_interval)
                print(np.mean(acceptance))                                                        
                iter_state = x_new
                iter_state.timestamp = state.timestamp
        #        v = v_new   
                if i == self.num_iterations:
                    x_new.timestamp = new_state_pred.timestamp

            # pi(x_k)  # state is like prior
            pi_x_k = self.target_proposal(x_new, state, measurement, time_interval)

            # pi(x_k-1)
            pi_x_k1 = self.target_proposal(state, state, measurement, time_interval)

            # we need to add the deteminat of the jacobian of the LF integrator
            # following eq 22 in Alessandro's papers
            # wt = wt-1 * (pi_x_k * qv(-v)/det(J))/(pi_x_k1 * p_x_xk1 * qv(v)/det(J))
            # 1/-1 id the direction

            jk_minus = self.integrate_lf_vec(x_new, state, v, grad_x, 1, self.step_size,
                                             time_interval, measurement)
            jk_plus = self.integrate_lf_vec(x_new, state, v_new, grad_x, -1, self.step_size,
                                            time_interval, measurement)

            j_cab_m = self.get_grad(jk_minus, time_interval)
            j_cab_p = self.get_grad(jk_plus, time_interval)

            determinant_m = np.linalg.det(j_cab_m)
            determinant_p = np.linalg.det(j_cab_p)

            # qv (-v)/det(J)
            q_star_minus = mvn.logpdf(-v_new, mean=np.zeros(v.shape[1]),
                                      cov=self.mass_matrix)/determinant_m
            # qv (v)/det(J)
            q_star_plus = mvn.logpdf(v, mean=np.zeros(v.shape[1]),
                                     cov=self.mass_matrix)/determinant_p

        else:
            # No updates
            x_new = new_state_pred
            pi_x_k = 0
            pi_x_k1 = 0
            q_star_minus = 0
            q_star_plus = 0

        final_state = Prediction.from_state(previous_state,
                                            parent=previous_state,
                                            state_vector=x_new.state_vector,
                                            timestamp=timestamp,
                                            transition_model=self.transition_model,
                                            prior=state)

        final_state.log_weight += pi_x_k - pi_x_k1 + q_star_minus - q_star_plus

        return final_state

    def get_grad(self, new_state, time_interval):
        """Use the Jacobian of the model"""
        return self.transition_model.jacobian(new_state, time_interval=time_interval)

    def integrate_lf_vec(self, state, new_state_pred, v, grad_x, direction, h, time_interval,
                         measurement):
        """Leapfrog integration"""
        h = h.reshape(self.num_samples, 1)
        v = v + direction * (h / 2) * grad_x
        einsum = np.einsum('bij,bj->bi', self.inv_MM, v)
        state.state_vector = (state.state_vector.T + direction * h * einsum).T

        grad_x = self.grad_target_proposal(state, new_state_pred, measurement, time_interval)
        v = v + direction * (h / 2) * grad_x
        return state, v, grad_x

    def stop_criterion_vec(self, xminus, xplus, rminus, rplus):
        """Stop Criterion"""

        # Return True for particles we want to stop (NB opposite way round to s in Hoffman
        # and Gelman paper)
        dx = xplus.state_vector.T - xminus.state_vector.T
        left = (np.sum(dx * np.einsum('bij,bj->bi', self.inv_MM, rminus), axis=1) < 0)
        right = (np.sum(dx * np.einsum('bij,bj->bi', self.inv_MM, rplus), axis=1) < 0)
        return np.logical_or(left, right)

    def get_hamiltonian(self, v, logp):
        """Hamiltonian calculation"""

        # Get Hamiltonian energy of system given log target weight logp
        return logp - 0.5 * np.sum(v * np.einsum('bij,bj->bi', self.inv_MM, v),
                                   axis=1).reshape(-1, 1)

    def merge_states_dir(self, xminus, vminus, grad_xminus, xplus, vplus, grad_xplus, direction):
        """ Auxiliary function to merge the states"""

        # Return xmerge = vectors of xminus where direction < 0 and xplus where direction > 0, and
        # similarly for v and grad_x
        mask = direction[:, 0] < 0
        mask = mask.reshape(len(mask), 1).astype(int)
        xmerge = mask * xminus.state_vector.T + (1 - mask) * xplus.state_vector.T
        vmerge = mask * vminus + (1 - mask) * vplus
        grad_xmerge = mask * grad_xminus + (1 - mask) * grad_xplus
        xminus.state_vector = xmerge.T
        return xminus, vmerge, grad_xmerge

    def generate_nuts_samples(self, x0, x1, v0, grad_x0, detection, time_interval):
        """ Generate NUTS samples"""

        # Sample energy: note that log(U(0,1)) has same distribution as -exponential(1)
        logp0 = self.target_proposal(x1, x0, detection,
                                     time_interval=time_interval).reshape(-1, 1)
        joint = self.get_hamiltonian(v0, logp0)
        logu = joint + np.log(uniform.rvs())

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

        # criteria
        stopped = np.zeros((self.num_samples, 1)).astype(bool)
        numnodes = np.ones((self.num_samples, 1)).astype(int)

        # Used to compute acceptance rate
        alpha = np.zeros((self.num_samples, 1))
        nalpha = np.zeros((self.num_samples, 1)).astype(int)

        while np.any(stopped == 0):

            # Generate random direction in {-1, +1}
            direction = (2 * (uniform.rvs(0, 1, size=self.num_samples)
                              < 0.5).astype(int) - 1).reshape(-1, 1)

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
            idxplus = np.logical_and(np.logical_not(stopped), direction >
                                     0).reshape(self.num_samples, 1).astype(int)
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

            selectnew = np.logical_and(np.logical_not(stopped2), u).reshape(self.num_samples,
                                                                            1).astype(int)

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

        return xprime, vprime, acceptance

    def build_tree(self, x, x1, v, grad_x, joint, logu, direction, stopped, depth, time_interval,
                   detection):
        """Function to build the particle trees"""

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
                                                                   detection)

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
                                                                           detection)

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
                u = numnodes.reshape(-1, 1) * \
                    uniform.rvs(size=self.num_samples).reshape(-1, 1) < \
                    numnodes2.reshape(-1, 1)

                selectnew = np.logical_and(np.logical_not(stopped2), u).reshape(self.num_samples,
                                                                                1).astype(int)
                xprime = State(state_vector=StateVectors((selectnew * xprime2.state_vector.T
                                                          + (1 - selectnew) *
                                                          xprime.state_vector.T).T))
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

            return xminus, vminus, grad_xminus, xplus, vplus, grad_xplus, xprime, \
                vprime, numnodes, stopped, alpha, nalpha
<<<<<<< Updated upstream
        
=======

    # adaptive step size bit
    def single_iteration(self):
        """
            Function to iter over the value and check that the stepsize works ok.
        """

        return 'paul old magic trick'

    def adaptive_stepsize(self):
        """
            Function to optimise the stepsize and the mass matrix to improve the
            quality and avoid hand-tuning the various parameters
        """

        converged = False
        max_iterations =5
        iterations = 0

        #while (not converged and counter < max_iterations):
            # run it


        return 'pauls magic'

    def _adapt_step_size(self, acceptance, Hbar, mu, k):
        """Function to modify the stepsize"""
        self.t0 = 10
        self.delta = 0.8
        # acceptance comes from the nuts code
        eta = 1. / float(k + self.t0)
        Hbar = (1. - eta) * Hbar + eta * (self.delta - acceptance)
        # modify H
        self.h = np.exp(mu - np.sqrt(k) / self.gamma * Hbar)

        return Hbar


    def _check_step_size(self, x, grad_x, step_size, rng):
        """
            Function to check the step size
        """

        # samples a new momentum
        r = np.transpose(mvn.rvs(mean=np.zeros(self.num_dims), cov=self.MM[0]))
#        r = np.transpose(rng.multivariate_normal(mu=np.zeros(self.sampler.D), cov=self.sampler.MM[0]))

        #Take a leapfrog step with momentum r
        x_prime, r_prime, _ = self.Integrate_LF_vec(x, r, grad_x, np.ones((self.sampler.N)), step_size)

        # need to evaluate the target bits here
        # Calculate the (log) Hamiltonians at the end and start of the proposal
        H0 = self.sampler.target(x) - (0.5 * np.sum(r*r, axis=1))
        H1 = self.sampler.target(x_prime) - (0.5 * np.sum(r_prime*r_prime, axis=1))

        #calculate the difference in the Hamiltonian
        delta_H = H1 - H0

        return delta_H

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
>>>>>>> Stashed changes
