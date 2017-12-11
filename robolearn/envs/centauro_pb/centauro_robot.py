import os
import numpy as np
from robolearn.envs.pybullet.pybullet_robot import PyBulletRobot

CENTAURO_JOINT_NAMES = ['torso_yaw',  # Joint 00

                        'j_arm2_1',  # Joint 01
                        'j_arm2_2',  # Joint 02
                        'j_arm2_3',  # Joint 03
                        'j_arm2_4',  # Joint 04
                        'j_arm2_5',  # Joint 05
                        'j_arm2_6',  # Joint 06
                        'j_arm2_7',  # Joint 07

                        'j_arm1_1',  # Joint 08
                        'j_arm1_2',  # Joint 09
                        'j_arm1_3',  # Joint 10
                        'j_arm1_4',  # Joint 11
                        'j_arm1_5',  # Joint 12
                        'j_arm1_6',  # Joint 13
                        'j_arm1_7',  # Joint 14

                        'hip_yaw_1',  # Joint 15
                        'hip_pitch_1',  # Joint 16
                        'knee_pitch_1',  # Joint 17

                        'hip_yaw_2',  # Joint 18
                        'hip_pitch_2',  # Joint 19
                        'knee_pitch_2',  # Joint 20

                        'hip_yaw_3',  # Joint 21
                        'hip_pitch_3',  # Joint 22
                        'knee_pitch_3',  # Joint 23

                        'hip_yaw_4',  # Joint 24
                        'hip_pitch_4',  # Joint 25
                        'knee_pitch_4',  # Joint 26

                        'ankle_pitch_1',  # Joint 27  #FL
                        'ankle_yaw_1',  # Joint 28
                        'j_wheel_1',  # Joint 29

                        'ankle_pitch_2',  # Joint 30  #FR
                        'ankle_yaw_2',  # Joint 31
                        'j_wheel_2',  # Joint 32


                        'ankle_pitch_3',  # Joint 33  #BL
                        'ankle_yaw_3',  # Joint 34
                        'j_wheel_3',  # Joint 35

                        'ankle_pitch_4',  # Joint 36  #BR
                        'ankle_yaw_4',  # Joint 37
                        'j_wheel_4',  # Joint 38

                        'neck_yaw',  # Joint 39
                        'neck_pitch',  # Joint 40
                        # 'j_ft_arm2',
                        # 'imu_joint',
                        ]


# TODO: WE ARE CHANGING ORDER IN LEG JOINTS
CENTAURO_INIT_CONFIG = [0.0,  # Joint 00

                        0.0,  # Joint 01
                        0.3,  # Joint 02
                        0.8,  # Joint 03
                        0.8,  # Joint 04
                        0.0,  # Joint 05
                        0.8,  # Joint 06
                        0.0,  # Joint 07

                        0.0,  # Joint 08
                        -0.3,  # Joint 09
                        -0.8,  # Joint 10
                        -0.8,  # Joint 11
                        0.0,  # Joint 12
                        -0.8,  # Joint 13
                        0.0,  # Joint 14

                        0.8,  # Joint 15
                        1.0,  # Joint 16
                        -1.0,  # Joint 17

                        0.8,  # Joint 18
                        1.0,  # Joint 19
                        -1.0,  # Joint 20

                        -0.8,  # Joint 21
                        1.0,  # Joint 22
                        -1.0,  # Joint 23

                        -0.8,  # Joint 24
                        1.0,  # Joint 25
                        -1.0,  # Joint 26

                        0.0,  # Joint 27
                        0.8,  # Joint 28
                        0.0,  # Joint 29

                        0.0,  # Joint 30
                        -0.8,  # Joint 31
                        0.0,  # Joint 32

                        0.0,  # Joint 33
                        -0.8,  # Joint 34
                        0.0,  # Joint 35

                        0.0,  # Joint 36
                        0.8,  # Joint 37
                        0.0,  # Joint 38

                        0.0,  # Joint 39
                        0.8,  # Joint 40
                        ]

# TODO: CHANGING ORDER IN LEGS
change_joint = [15, 16, 20, 21, 23, 25]
for ii in change_joint:
    CENTAURO_INIT_CONFIG[ii] *= -1

# CENTAURO_JOINT_NAMES = CENTAURO_JOINT_NAMES[:15]
# CENTAURO_INIT_CONFIG = CENTAURO_INIT_CONFIG[:15]

# remove_joints = [29, 32, 35, 38]
# CENTAURO_JOINT_NAMES = [v for i, v in enumerate(CENTAURO_JOINT_NAMES) if i not in remove_joints]
# CENTAURO_INIT_CONFIG = [v for i, v in enumerate(CENTAURO_INIT_CONFIG) if i not in remove_joints]

CENTAURO_BODY_PARTS = {'LA': range(1, 8),
                       'RA': range(8, 15),
                       'BA': range(1, 15),
                       'TO': [0],
                       'HE': range(39, 41),
                       'UB': list(range(1, 15))+list(range(39, 41)),
                       'FL': range(27, 30),
                       'FR': range(30, 33),
                       'BL': range(33, 36),
                       'BR': range(36, 39),
                       'LEGS': range(27, 39),
                       'WB': range(0, 41)}


class CentauroBulletRobot(PyBulletRobot):
    def __init__(self, robot_name='pelvis', init_pos=None, self_collision=True,
                 control_type='position', active_joints='WB'):

        # urdf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'urdf/centauro_full_with_imu.urdf')
        # urdf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'urdf/centauro_full.urdf')
        # urdf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'urdf/centauro_cvx_hull.urdf')
        urdf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'urdf/stick_hands3.urdf')

        self.act_dim = len(CENTAURO_JOINT_NAMES[active_joints])
        self.obs_dim = len(CENTAURO_JOINT_NAMES[active_joints])*2
        # self.act_dim = 27
        # self.obs_dim = 54
        # self.act_dim = 15
        # self.obs_dim = 30

        self.power = 1
        self.active_joints = active_joints

        super(CentauroBulletRobot, self).__init__('urdf', urdf_xml, robot_name,
                                                  self.act_dim, self.obs_dim,
                                                  init_pos=init_pos,
                                                  joint_names=CENTAURO_JOINT_NAMES,
                                                  self_collision=self_collision)

        if control_type not in ['position', 'velocity', 'torque']:
            raise ValueError('Wrong control type %s' % control_type)
        self.control_type = control_type

    def robot_reset(self):
        # Reorder the Joints
        if len(self.ordered_joints) != len(CENTAURO_JOINT_NAMES):
            raise AttributeError('Centauro ordered joints number (%d) is different than default %d' % (len(self.ordered_joints),
                                                                                                     len(CENTAURO_JOINT_NAMES)))

        for jj, joint_name in enumerate(CENTAURO_JOINT_NAMES):
            self.ordered_joints[jj] = self.jdict[joint_name]

        print('JOINT MAP')
        for jj, joint_name in enumerate(CENTAURO_JOINT_NAMES):
            print(jj, joint_name)

        self.total_joints = len(self.ordered_joints)
        self.state_per_joint = 2
        self.initial_state = np.zeros(self.obs_dim)
        self.initial_state[:self.total_joints] = CENTAURO_INIT_CONFIG

        for jj, joint in enumerate(self.ordered_joints):
            joint.reset_position(self.initial_state[jj], self.initial_state[self.total_joints+jj])
            print('JOINT', jj, joint.get_torque())

        print('PARTS', sorted(self.parts.keys()))
        print('body_name', self.robot_body.body_name)

        self._cameras = list()
        self._cameras.append(self.add_camera('kinect2_rgb_optical_frame'))

    def robot_state(self):
        self.total_joints = len(self.ordered_joints[self.active_joints])
        self.state_per_joint = 2

        state = np.zeros(self.state_per_joint*self.total_joints)

        for jj, joint in enumerate(self.ordered_joints[self.active_joints]):
            state[[jj, self.total_joints+jj]] = joint.current_position()

        return state

    def apply_action(self, action):
        assert (np.isfinite(action).all())
        for n, joint in enumerate(self.ordered_joints[self.active_joints]):
            # joint.set_motor_torque(self.power * joint.power_coef * float(np.clip(action[n], -1, +1)))
            # print(self.power * joint.power_coef * float(np.clip(action[n], -1, +1)), joint.power_coef, self.power)
            if self.control_type == 'position':
                joint.set_position(action[n])
            elif self.control_type == 'velocity':
                joint.set_velocity(action[n])
            elif self.control_type == 'torque':
                joint.set_motor_torque(self.power * joint.power_coef * float(np.clip(action[n], -1, +1)))

    def get_image(self, camera_id=-1):
        return self._cameras[camera_id].get_image()

