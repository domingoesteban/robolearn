import copy
import numpy as np

from robolearn.policies.policy_opt.config import POLICY_OPT_RANDOM
from robolearn.policies.policy_opt.policy_opt import PolicyOpt
from robolearn.policies.random_policy import RandomPolicy

class PolicyOptRandom(PolicyOpt):
    def __init__(self, hyperparams, dO, dU):
        config = copy.deepcopy(POLICY_OPT_RANDOM)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config, dO, dU)

        self.var = self._hyperparams['init_var'] * np.eye(dU)

        self.policy = RandomPolicy(dU, action_cov=self.var)

    def update(self, obs, tgt_mu, tft_prc, tgt_wt):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A random policy with updated parameters.
        """
        #TODO: Check if we can implemented
        return self.policy

    def prob(self, obs):
        """
        Run policy forward.
        Args:
            obs: Numpy array of observations that is N x T x dO.
        """
        #TODO: Check if this can be implemented
        dU = self._dU
        N, T = obs.shape[:2]

        output = np.zeros((N, T, dU))

        pol_sigma = np.tile(np.diag(self.var), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var), [N, T])

        return output, pol_sigma, pol_prec, pol_det_sigma
