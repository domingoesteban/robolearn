import os
import numpy as np
from robolearn.envs.pybullet.pybullet_robot import PyBulletRobot

BIGMAN_JOINT_NAMES = ['LHipLat',        # Joint 0
                      'LHipYaw',        # Joint 1
                      'LHipSag',        # Joint 2
                      'LKneeSag',       # Joint 3
                      'LAnkSag',        # Joint 4
                      'LAnkLat',        # Joint 5
                      'RHipLat',        # Joint 6
                      'RHipYaw',        # Joint 7
                      'RHipSag',        # Joint 8
                      'RKneeSag',       # Joint 9
                      'RAnkSag',        # Joint 10
                      'RAnkLat',        # Joint 11
                      'WaistLat',       # Joint 12
                      'WaistSag',       # Joint 13
                      'WaistYaw',       # Joint 14
                      'LShSag',         # Joint 15
                      'LShLat',         # Joint 16
                      'LShYaw',         # Joint 17
                      'LElbj',          # Joint 18
                      'LForearmPlate',  # Joint 19
                      'LWrj1',          # Joint 20
                      'LWrj2',          # Joint 21
                      'NeckYawj',       # Joint 22
                      'NeckPitchj',     # Joint 23
                      'RShSag',         # Joint 24
                      'RShLat',         # Joint 25
                      'RShYaw',         # Joint 26
                      'RElbj',          # Joint 27
                      'RForearmPlate',  # Joint 28
                      'RWrj1',          # Joint 29
                      'RWrj2']          # Joint 30


BIGMAN_INIT_CONFIG = [-0.05,  # Joint 0
                      0.0,  # Joint 1
                      -0.50,  # Joint 2
                      1.0,  # Joint 3
                      -0.45,  # Joint 4
                      0.05,  # Joint 5
                      0.05,  # Joint 6
                      0.0,  # Joint 7
                      -0.50,  # Joint 8
                      1.0,  # Joint 9
                      -0.45,  # Joint 10
                      -0.05,  # Joint 11
                      0.0,  # Joint 12
                      0.1,  # Joint 13
                      0.0,  # Joint 14
                      0.5,  # Joint 15
                      0.25,  # Joint 16
                      0.0,  # Joint 17
                      -1.0,  # Joint 18
                      0.0,  # Joint 19
                      0.0,  # Joint 20
                      0.0,  # Joint 21
                      0.0,  # Joint 22
                      0.0,  # Joint 23
                      0.5,  # Joint 24
                      -0.25,  # Joint 25
                      0.0,  # Joint 26
                      -1.0,  # Joint 27
                      0.0,  # Joint 28
                      0.0,  # Joint 29
                      0.0]  # Joint 30


class BigmanBulletRobot(PyBulletRobot):
    def __init__(self, robot_name='base_link', init_pos=None, self_collision=True, control_type='position'):

        urdf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'urdf/bigman.urdf')
        # urdf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'urdf/bigman_only_upper_body2.urdf')

        self.act_dim = 31
        self.obs_dim = 62

        self.power = 1

        super(BigmanBulletRobot, self).__init__('urdf', urdf_xml, robot_name, self.act_dim, self.obs_dim, init_pos=init_pos,
                                                joint_names=BIGMAN_JOINT_NAMES, self_collision=self_collision)

        if control_type not in ['position', 'velocity', 'torque']:
            raise ValueError('Wrong control type %s' % control_type)
        self.control_type = control_type

    def robot_reset(self):
        # Reorder the Joints
        if len(self.ordered_joints) != len(BIGMAN_JOINT_NAMES):
            raise AttributeError('Bigman ordered joints number (%d) is different than default %d' % (len(self.ordered_joints),
                                                                                                     len(BIGMAN_JOINT_NAMES)))

        for jj, joint_name in enumerate(BIGMAN_JOINT_NAMES):
            self.ordered_joints[jj] = self.jdict[joint_name]

        self.total_joints = len(self.ordered_joints)
        self.state_per_joint = 2
        self.initial_state = np.zeros(self.obs_dim)
        self.initial_state[:self.total_joints] = BIGMAN_INIT_CONFIG

        for jj, joint in enumerate(self.ordered_joints):
            joint.reset_position(self.initial_state[jj], self.initial_state[self.total_joints+jj])

        print('PARTS', self.parts.keys())
        print('body_name', self.robot_body.body_name)

    def robot_state(self):
        self.total_joints = len(self.ordered_joints)
        self.state_per_joint = 2

        state = np.zeros(self.state_per_joint*len(self.ordered_joints))

        for jj, joint in enumerate(self.ordered_joints):
            state[[jj, self.total_joints+jj]] = joint.current_position()

        return state

    def apply_action(self, action):
        assert (np.isfinite(action).all())
        for n, joint in enumerate(self.ordered_joints):
            # joint.set_motor_torque(self.power * joint.power_coef * float(np.clip(action[n], -1, +1)))
            # print(self.power * joint.power_coef * float(np.clip(action[n], -1, +1)), joint.power_coef, self.power)
            if self.control_type == 'position':
                joint.set_position(action[n])
            elif self.control_type == 'velocity':
                joint.set_velocity(action[n])
            elif self.control_type == 'torque':
                joint.set_motor_torque(self.power * joint.power_coef * float(np.clip(action[n], -1, +1)))


