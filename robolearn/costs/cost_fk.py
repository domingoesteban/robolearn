""" This file defines the forward kinematics cost function. """
import copy

import numpy as np

from robolearn.costs.config import COST_FK
from robolearn.costs.cost import Cost
from robolearn.costs.cost_utils import get_ramp_multiplier
from robolearn.utils.transformations import compute_cartesian_error


class CostFK(Cost):
    """
    Forward kinematics cost function. Used for costs involving the end
    effector position.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_FK)
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        """
        Evaluate forward kinematics (end-effector penalties) cost.
        Temporary note: This implements the 'joint' penalty type from
            the matlab code, with the velocity/velocity diff/etc.
            penalties removed. (use CostState instead)
        Args:
            sample: A single sample.
        """
        T = sample.T
        dX = sample.dX
        dU = sample.dU

        wpm = get_ramp_multiplier(
            self._hyperparams['ramp_option'], T,
            wp_final_multiplier=self._hyperparams['wp_final_multiplier']
        )
        wp = self._hyperparams['wp'] * np.expand_dims(wpm, axis=-1)

        # IK model terms
        robot_model = self._hyperparams['robot_model']
        end_effector_name = self._hyperparams['end_effector_name']
        end_effector_offset = self._hyperparams['end_effector_offset']
        joint_ids = self._hyperparams['joint_ids']

        # Initialize terms.
        l = np.zeros(T)
        lu = np.zeros((T, dU))
        lx = np.zeros((T, dX))
        luu = np.zeros((T, dU, dU))
        lxx = np.zeros((T, dX, dX))
        lux = np.zeros((T, dU, dX))

        # Choose target.
        tgt = self._hyperparams['target_end_effector'][[3, 4, 5, 6, 0, 1, 2]]
        x = sample.get_states('link_position')

        dist = np.zeros((T, 6))
        Jx = np.zeros((T, 6, len(self._hyperparams['state_idx'])))
        jtemp = np.zeros((6, robot_model.qdot_size))
        q = np.zeros(robot_model.q_size)

        temp_idx = np.ix_(range(6), self._hyperparams['state_idx'])
        for ii in range(T):
            q[joint_ids] = x[ii, :]
            dist[ii, :] = -compute_cartesian_error(tgt, robot_model.fk(end_effector_name,
                                                                       q=q,
                                                                       body_offset=end_effector_offset,
                                                                       update_kinematics=True,
                                                                       rotation_rep='quat'))
            robot_model.update_jacobian(jtemp, end_effector_name, q=q,
                                        body_offset=end_effector_offset, update_kinematics=True)

            Jx[ii, temp_idx[0], temp_idx[1]] = jtemp[:, joint_ids]

        # Evaluate penalty term. Use estimated Jacobians and no higher
        # order terms.
        jxx_zeros = np.zeros((T, dist.shape[1], Jx.shape[2], Jx.shape[2]))
        l, ls, lss = self._hyperparams['evalnorm'](
            wp, dist, Jx, jxx_zeros, self._hyperparams['l1'],
            self._hyperparams['l2'], self._hyperparams['alpha']
        )

        # Add to current terms.
        lx[:, self._hyperparams['state_idx']] = ls
        temp_idx = np.ix_(self._hyperparams['state_idx'], self._hyperparams['state_idx'])
        lxx[:, temp_idx[0], temp_idx[1]] = lss
        #sample.agent.pack_data_x(lx, ls, data_types=[JOINT_ANGLES])
        #sample.agent.pack_data_x(lxx, lss,
        #                         data_types=[JOINT_ANGLES, JOINT_ANGLES])

        return l, lx, lu, lxx, luu, lux
