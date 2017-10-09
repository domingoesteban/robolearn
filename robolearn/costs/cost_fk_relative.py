""" This file defines the forward kinematics cost function. """
import copy

import numpy as np

from robolearn.costs.config import COST_FK_RELATIVE
from robolearn.costs.cost import Cost
from robolearn.costs.cost_utils import get_ramp_multiplier, evall1l2term
from robolearn.utils.transformations_utils import compute_cartesian_error, pose_transform, quaternion_inner


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
        jtemp = np.zeros((6, robot_model.qdot_size))
        q = np.zeros(robot_model.q_size)
        # Jx = np.zeros((T, 6, len(self._hyperparams['data_idx'])))

        # TODO: Adding rel_pose Jacobian
        Jx = np.zeros((T, 6, len(self._hyperparams['data_idx'])+len(self._hyperparams['rel_idx'])))
        temp_rel_jacobian = np.zeros((6, len(self._hyperparams['rel_idx'])))
        temp_rel_jacobian[0, 0] = temp_rel_jacobian[1, 1] = temp_rel_jacobian[2, 2] = -1
        temp_rel_jacobian[-1, -1] = temp_rel_jacobian[-2, -2] = temp_rel_jacobian[-3, -3] = -1
        rel_idx = np.ix_(range(6), range(-7, 0))

        data_idx = np.ix_(range(6), self._hyperparams['data_idx'])
        prev_op_point = np.array([0, 0, 0, 1, 0, 0, 0])
        for ii in range(T):
            tgt = pose_transform(data_pose[ii, :], rel_pose)
            q[joint_ids] = x[ii, :]
            # dist[ii, :] = -compute_cartesian_error(tgt, robot_model.fk(op_point_name,
            # dist[ii, :] = compute_cartesian_error(tgt, robot_model.fk(op_point_name,
            #                                                           q=q,
            #                                                           body_offset=op_point_offset,
            #                                                           update_kinematics=True,
            #                                                           rotation_rep='quat'))

            op_point = robot_model.fk(op_point_name, q=q, body_offset=op_point_offset, update_kinematics=True,
                                      rotation_rep='quat')

            if ii > 0:
                if quaternion_inner(op_point[:4], prev_op_point[:4]) < 0:
                    op_point[:4] *= -1

            prev_op_point[:] = op_point[:]

            dist[ii, :] = compute_cartesian_error(op_point, tgt)
            robot_model.update_jacobian(jtemp, op_point_name, q=q,
                                        body_offset=op_point_offset, update_kinematics=True)
            # if ii == 1:
            #     print('****')
            #     print(sample.get_states()[1, :])
            #     print(self._hyperparams['data_idx'])
            #     print(op_point_name)
            #     print(x[ii, :])
            #     print(q[joint_ids])
            #     print(tgt)
            #     print(dist[ii, :])
            #     print('++++')
            #     raw_input('waaaaa')

            #Jx[ii, data_idx[0], data_idx[1]] = -jtemp[:, joint_ids]
            Jx[ii, data_idx[0], data_idx[1]] = jtemp[:, joint_ids]

            # TODO: Adding rel_pose Jacobian
            Jx[ii, rel_idx[0], rel_idx[1]] = temp_rel_jacobian[:, :]


        # Evaluate penalty term. Use estimated Jacobians and no higher
        # order terms.
        jxx_zeros = np.zeros((T, dist.shape[1], Jx.shape[2], Jx.shape[2]))
        #l, ls, lss = self._hyperparams['evalnorm'](
        l, ls, lss = evall1l2term(
            wp, dist, Jx, jxx_zeros, self._hyperparams['l1'],
            self._hyperparams['l2'], self._hyperparams['alpha']
        )

        # Add to current terms.
        # lx[:, self._hyperparams['data_idx']] = ls
        lx[:, self._hyperparams['data_idx'] + self._hyperparams['rel_idx']] = ls
        # data_idx = np.ix_(self._hyperparams['data_idx'], self._hyperparams['data_idx'] + self._hyperparams['rel_idx'])
        data_idx = np.ix_(self._hyperparams['data_idx'] + self._hyperparams['rel_idx'],
                          self._hyperparams['data_idx'] + self._hyperparams['rel_idx'])
        lxx[:, data_idx[0], data_idx[1]] = lss
        #sample.agent.pack_data_x(lx, ls, data_types=[JOINT_ANGLES])
        #sample.agent.pack_data_x(lxx, lss,
        #                         data_types=[JOINT_ANGLES, JOINT_ANGLES])

        # print(l[1])
        # raw_input('ayachi')

        return l, lx, lu, lxx, luu, lux
