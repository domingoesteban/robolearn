import os
import numpy as np
import logging
from robolearn.envs.pybullet.pybullet_robot import PyBulletRobot


class Pusher3DofRobot(PyBulletRobot):
    def __init__(self, robot_name='reacher', self_collision=True):
        # Logger
        self.logger = logging.getLogger('pybullet')
        self.logger.setLevel(logging.INFO)

        self.logger.info('pbPUSHER | Creating new PusherURDF')

        pusher_urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'models/pusher3dof.urdf')

        self.act_dim = 3
        self.state_per_joint = 2

        self.obs_dim = self.act_dim*self.state_per_joint

        # self.power = 0.01  # Because each joint has power of 100
        self.power = 0.01# * 0.01

        joint_names = ['joint0', 'joint1', 'joint2']

        super(Pusher3DofRobot, self).__init__('urdf', pusher_urdf, 'base_link',
                                              self.act_dim, self.obs_dim,
                                              self_collision=self_collision,
                                              joint_names=joint_names)

        # Initial / default values
        self.initial_state = np.zeros(self.obs_dim)
        self.total_joints = self.action_dim

    def robot_reset(self, state=None):
        self.total_joints = len(self.ordered_joints)

        if state is not None:
            self.initial_state = state

        for jj, joint in enumerate(self.ordered_joints):
            joint.reset_position(self.initial_state[jj],
                                 self.initial_state[self.total_joints+jj])

        # Color
        color_list = [[0.0, 0.4, 0.6, 1]
                      for _ in range(self.get_total_bodies())]
        color_list[0] = [0.9, 0.4, 0.6, 1]
        color_list[-1] = [0.0, 0.8, 0.6, 1]
        self.set_body_colors(color_list)

    def robot_state(self):
        self.total_joints = len(self.ordered_joints)
        # self.initial_state = np.zeros(self.obs_dim)

        state = np.zeros(self.state_per_joint*len(self.ordered_joints))

        for jj, joint in enumerate(self.ordered_joints):
            state[[jj, self.total_joints+jj]] = joint.get_state()

        return state

    def get_state_info(self):
        state_info = list()  # Array of dict = {type, idx}

        state_info.append({'name': 'position',
                           'idx': list(range(0, self.total_joints))})
        state_info.append({'name': 'velocity',
                           'idx': list(range(self.total_joints,
                                        2*self.total_joints))})
        return state_info

    def apply_action(self, action):
        assert (np.isfinite(action).all())
        for n, joint in enumerate(self.ordered_joints):
            joint.set_motor_torque(self.power * joint.power_coef *
                                   float(np.clip(action[n], -1, +1)))

    def set_initial_state(self, state):
        if len(state) != len(self.initial_state):
            raise ValueError('Wrong robot initial_state size (%d != %d)'
                             % (len(state), len(self.initial_state)))
        self.initial_state = state


