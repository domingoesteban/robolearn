import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from robolearn.torch.core import PyTorchModule
from robolearn.core.serializable import Serializable
import robolearn.torch.pytorch_util as ptu
from robolearn.models import Transition
from robolearn.torch.ops.gauss_fit_joint_prior import gauss_fit_joint_prior


class TVLGDynamics(PyTorchModule, Transition):
    def __init__(self, horizon, obs_dim, action_dim):
        self._T = horizon
        Transition.__init__(self, obs_dim=obs_dim, action_dim=action_dim)

        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        super(TVLGDynamics, self).__init__()

        self.Fm = nn.Parameter(ptu.zeros(horizon, obs_dim, obs_dim+action_dim))
        self.fv = nn.Parameter(ptu.ones(horizon, obs_dim))
        self.dyn_covar = nn.Parameter(ptu.zeros(horizon, obs_dim, obs_dim))

        # Prior
        self._prior = None

    def get_next(self, observation, action):
        pass

    def forward(self, obs, act, time=None, stochastic=False):
        if time is None:
            raise NotImplementedError

        obs_and_act = torch.cat((obs, act), dim=-1)

        batch = obs.shape[:-1]

        mean = obs_and_act.mm(torch.t(self.Fm[time])) + self.fv[time]
        cov = self.dyn_covar[time]

        next_obs = mean

        return next_obs

    def get_prior(self):
        return self._prior

    def fit(self, X, U, regularization=1e-6):
        """ Fit dynamics. """
        N, T, dX = X.shape
        dU = U.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        it = slice(dX+dU)

        # Fit dynamics with least squares regression.
        dwts = (1.0 / N) * ptu.ones(N)


        for t in range(T - 1):
            Ys = torch.cat((X[:, t, :], U[:, t, :], X[:, t+1, :]), dim=-1)

            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.prior.eval(dX, dU, Ys)
            sig_reg = ptu.zeros((dX+dU+dX, dX+dU+dX))
            sig_reg[it, it] = regularization

            Fm, fv, dyn_covar = \
                gauss_fit_joint_prior(Ys, mu0, Phi, mm, n0,
                                      dwts, dX+dU, dX, sig_reg)
            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv
            self.dyn_covar[t, :, :] = dyn_covar

    def set_prior(self, prior):
        self._prior = prior
