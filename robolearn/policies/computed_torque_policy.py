import numpy as np
import rbdl

from robolearn.policies.policy import Policy

pd_tau_weights = np.array([0.80,  0.50,  0.80,  0.50,  0.50,  0.20,
                           0.80,  0.50,  0.50,  0.50,  0.50,  0.20,
                           0.50,  0.80,  0.50,
                           0.50,  0.80,  0.50,  0.50,  0.10,  0.20,   0.03,
                           0.03,  0.03,
                           0.50,  0.80,  0.50,  0.50,  0.10,  0.20,   0.03])
Kp_tau = 100 * pd_tau_weights
Kd_tau = 2 * pd_tau_weights


class ComputedTorquePolicy(Policy):
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.tau = np.zeros(robot_model.qdot_size)

    def eval(self, q_des, qdot_des, qddot_des, q, qdot):
        # Feedforward term
        rbdl.InverseDynamics(self.robot_model, q_des, qdot_des, qddot_des, self.tau)

        # PD term
        pd_tau = Kp_tau * (q_des - q) + \
                 Kd_tau * (qdot_des - qdot)

        self.tau += pd_tau
        return self.tau.copy()

    def get_params(self):
        pass
