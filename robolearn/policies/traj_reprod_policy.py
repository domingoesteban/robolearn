import numpy as np

from robolearn.policies.policy import Policy
from robolearn.utils.trajectory_reproducer import TrajectoryReproducer


class TrajectoryReproducerPolicy(Policy):
    """
    Trajectory Reproducer policy.
    """
    def __init__(self, traj_files, act_idx=None):
        Policy.__init__(self)

        self.traj_rep = TrajectoryReproducer(traj_files)

        # Assume K has the correct shape, and make sure others match.
        self.dU = self.traj_rep.dim

        if act_idx is None:
            self.act_idx = range(self.dU)
        else:
            self.act_idx = act_idx

    def act(self, x=None, obs=None, t=None, noise=None):
        return self.traj_rep.get_data(t)[self.act_idx] if noise is None else self.traj_rep.get_data(t)[self.act_idx] + noise

    def eval(self, x=None, obs=None, t=None, noise=None):
        return self.traj_rep.get_data(t)[self.act_idx] if noise is None else self.traj_rep.get_data(t)[self.act_idx] + noise
