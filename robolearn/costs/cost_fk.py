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

        wpm = get_ramp_multiplier(self._hyperparams['ramp_option'], T,
                                  wp_final_multiplier=self._hyperparams['wp_final_multiplier'])
        wp = self._hyperparams['wp'] * np.expand_dims(wpm, axis=-1)

        # IK model terms
        robot_model = self._hyperparams['robot_model']
        op_point_name = self._hyperparams['op_point_name']
        op_point_offset = self._hyperparams['op_point_offset']
        joint_ids = self._hyperparams['joint_ids']

        # Initialize terms.
        l = np.zeros(T)
        lu = np.zeros((T, dU))
        lx = np.zeros((T, dX))
        luu = np.zeros((T, dU, dU))
        lxx = np.zeros((T, dX, dX))
        lux = np.zeros((T, dU, dX))

        # Choose target.
        tgt = self._hyperparams['target_pose']
        if self._hyperparams['tgt_data_type'] == 'state':
            dist_measured = sample.get_states()[:, self._hyperparams['tgt_idx']]
        elif self._hyperparams['tgt_data_type'] == 'observation':
            dist_measured = sample.get_obs()[:, self._hyperparams['tgt_idx']]
        else:
            raise ValueError("Wrong 'tgt_data_type' hyperparameter. It is not neither state or observation")
        joints_sensed = sample.get_states()[:, self._hyperparams['joints_idx']]

        dist = np.zeros((T, 6))
        jtemp = np.zeros((6, robot_model.qdot_size))  # Joints Jacobian
        jtemp_dist = np.eye(6)  # Distance Jacobian
        q = np.zeros(robot_model.q_size)

        Jx = np.zeros((T, 6, len(self._hyperparams['joints_idx']) + len(self._hyperparams['tgt_idx'])))

        tgt_idx = np.ix_(range(6), range(-len(self._hyperparams['tgt_idx']), 0))
        #joints_idx = np.ix_(range(6), self._hyperparams['joints_idx'])
        joints_idx = np.ix_(range(6), range(len(self._hyperparams['joints_idx'])))

        for ii in range(T):
            q[joint_ids] = joints_sensed[ii, :]
            # dist[ii, :] = compute_cartesian_error(robot_model.fk(op_point_name,
            #                                                      q=q,
            #                                                      body_offset=op_point_offset,
            #                                                      update_kinematics=True,
            #                                                      rotation_rep='quat'),
            #                                       tgt)
            dist[ii, :] = dist_measured[ii, :] - tgt

            robot_model.update_jacobian(jtemp, op_point_name, q=q,
                                        body_offset=op_point_offset, update_kinematics=True)

            Jx[ii, joints_idx[0], joints_idx[1]] = jtemp[:, joint_ids]

            # TODO: Adding tgt Jacobian
            Jx[ii, tgt_idx[0], tgt_idx[1]] = jtemp_dist[:, :]

        # Evaluate penalty term. Use estimated Jacobians and no higher
        # order terms.
        jxx_zeros = np.zeros((T, dist.shape[1], Jx.shape[2], Jx.shape[2]))
        l, ls, lss = self._hyperparams['evalnorm'](wp, dist, Jx, jxx_zeros, self._hyperparams['l1'],
                                                   self._hyperparams['l2'], self._hyperparams['alpha'])
        # print('------')
        # print(l)
        # print('......')
        # import matplotlib.pyplot as plt
        # plt.plot(l)
        # plt.show()

        # Add to current terms.
        lx[:, self._hyperparams['joints_idx'] + self._hyperparams['tgt_idx']] = ls
        temp_idx = np.ix_(self._hyperparams['joints_idx'] + self._hyperparams['tgt_idx'],
                          self._hyperparams['joints_idx'] + self._hyperparams['tgt_idx'])
        lxx[:, temp_idx[0], temp_idx[1]] = lss
        #sample.agent.pack_data_x(lx, ls, data_types=[JOINT_ANGLES])
        #sample.agent.pack_data_x(lxx, lss,
        #                         data_types=[JOINT_ANGLES, JOINT_ANGLES])

        return l, lx, lu, lxx, luu, lux
