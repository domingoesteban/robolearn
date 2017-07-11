""" This file defines the forward kinematics cost function. """
import copy

import numpy as np

from robolearn.costs.config import COST_FK_RELATIVE
from robolearn.costs.cost import Cost
from robolearn.costs.cost_utils import get_ramp_multiplier, evall1l2term
from robolearn.utils.transformations import compute_cartesian_error, pose_transform


class CostFKRelative(Cost):
    """
    Forward kinematics cost function. Used for costs involving the end
    effector position.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_FK_RELATIVE)
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
        rel_pose = self._hyperparams['target_rel_pose']
        if self._hyperparams['rel_data_type'] == 'state':
            # data_pose = sample.get_states(self._hyperparams['rel_data_name'])
            data_pose = sample.get_states()[:, self._hyperparams['rel_idx']]
        elif self._hyperparams['rel_data_type'] == 'observation':
            # data_pose = sample.get_obs(self._hyperparams['rel_data_name'])
            data_pose = sample.get_obs()[:, self._hyperparams['rel_idx']]
        else:
            raise ValueError("Wrong 'rel_data_type' hyperparameter. It is not neither state or observation")
        #x = sample.get_states('link_position')
        x = sample.get_states()[:, self._hyperparams['data_idx']]

        dist = np.zeros((T, 6))
        Jx = np.zeros((T, 6, len(self._hyperparams['data_idx'])))
        jtemp = np.zeros((6, robot_model.qdot_size))
        q = np.zeros(robot_model.q_size)

        temp_idx = np.ix_(range(6), self._hyperparams['data_idx'])
        for ii in range(T):
            tgt = pose_transform(data_pose[ii, :], rel_pose)
            q[joint_ids] = x[ii, :]
            # dist[ii, :] = -compute_cartesian_error(tgt, robot_model.fk(end_effector_name,
            # dist[ii, :] = compute_cartesian_error(tgt, robot_model.fk(end_effector_name,
            #                                                           q=q,
            #                                                           body_offset=end_effector_offset,
            #                                                           update_kinematics=True,
            #                                                           rotation_rep='quat'))
            dist[ii, :] = compute_cartesian_error(robot_model.fk(end_effector_name,
                                                                 q=q,
                                                                 body_offset=end_effector_offset,
                                                                 update_kinematics=True,
                                                                 rotation_rep='quat'),
                                                  tgt)
            robot_model.update_jacobian(jtemp, end_effector_name, q=q,
                                        body_offset=end_effector_offset, update_kinematics=True)
            # if ii == 1:
            #     print('****')
            #     print(sample.get_states()[1, :])
            #     print(self._hyperparams['data_idx'])
            #     print(end_effector_name)
            #     print(x[ii, :])
            #     print(q[joint_ids])
            #     print(tgt)
            #     print(dist[ii, :])
            #     print('++++')
            #     raw_input('waaaaa')

            #Jx[ii, temp_idx[0], temp_idx[1]] = -jtemp[:, joint_ids]
            Jx[ii, temp_idx[0], temp_idx[1]] = jtemp[:, joint_ids]

        # Evaluate penalty term. Use estimated Jacobians and no higher
        # order terms.
        jxx_zeros = np.zeros((T, dist.shape[1], Jx.shape[2], Jx.shape[2]))
        #l, ls, lss = self._hyperparams['evalnorm'](
        l, ls, lss = evall1l2term(
            wp, dist, Jx, jxx_zeros, self._hyperparams['l1'],
            self._hyperparams['l2'], self._hyperparams['alpha']
        )

        # Add to current terms.
        lx[:, self._hyperparams['data_idx']] = ls
        temp_idx = np.ix_(self._hyperparams['data_idx'], self._hyperparams['data_idx'])
        lxx[:, temp_idx[0], temp_idx[1]] = lss
        #sample.agent.pack_data_x(lx, ls, data_types=[JOINT_ANGLES])
        #sample.agent.pack_data_x(lxx, lss,
        #                         data_types=[JOINT_ANGLES, JOINT_ANGLES])

        # print(l[1])
        # raw_input('ayachi')

        return l, lx, lu, lxx, luu, lux
