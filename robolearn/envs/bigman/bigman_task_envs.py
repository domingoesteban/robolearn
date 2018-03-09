import os

import numpy as np

from robolearn.envs.bigman.bigman_env import BigmanEnv
from robolearn.utils.iit.iit_robots_params import bigman_params
from robolearn.utils.robot_model import RobotModel
from robolearn.utils.tasks.bigman.lift_box_utils import Reset_condition_bigman_box_gazebo
from robolearn.utils.tasks.bigman.lift_box_utils import create_hand_relative_pose


class BigmanBoxEnv(BigmanEnv):
    def __init__(self, body_part_active='LA', body_part_sensed='LA', Ts=0.01):
        interface = 'ros'
        self.task_hyperparams = dict()
        # Robot configuration

        self.task_hyperparams['Ts'] = Ts
        self.task_hyperparams['body_part_active'] = body_part_active
        self.task_hyperparams['body_part_sensed'] = body_part_sensed
        self.task_hyperparams['command_type'] = 'effort'

        self.task_hyperparams['box_size'] = [0.4, 0.5, 0.3]
        #robot_urdf_file = os.environ["ROBOTOLOGY_ROOT"]+'/configs/ADVR_shared/bigman/urdf/bigman.urdf'
        robot_urdf_file = os.environ["ROBOTOLOGY_ROOT"]+'/robots/iit-bigman-ros-pkg/bigman_urdf/urdf/bigman.urdf'
        self.task_hyperparams['robot_model'] = RobotModel(robot_urdf_file)
        self.task_hyperparams['LH_name'] = 'LWrMot3'
        self.task_hyperparams['RH_name'] = 'RWrMot3'
        self.task_hyperparams['l_soft_hand_offset'] = np.array([0.000, -0.030, -0.210])
        self.task_hyperparams['r_soft_hand_offset'] = np.array([0.000, 0.030, -0.210])

        self.task_hyperparams['touching_box_config'] = \
            np.array([0.,  0.,  0.,  0.,  0.,  0.,
                      0.,  0.,  0.,  0.,  0.,  0.,
                      0.,  0.,  0.,
                      0.0568,  0.2386, -0.2337, -1.6803,  0.2226,  0.0107,  0.5633,
                      #0.,  0.,  0.,  -1.5708,  0.,  0., 0.,
                      0.,  0.,
                      0.0568,  -0.2386, 0.2337, -1.6803,  -0.2226,  0.0107,  -0.5633])
                      #0.,  0.,  0.,  -1.5708,  0.,  0., 0.])

        observation_active = [{'name': 'joint_state',
                               'type': 'joint_state',
                               'ros_topic': '/xbotcore/bigman/joint_states',
                               # 'fields': ['link_position', 'link_velocity', 'effort'],
                               'fields': ['link_position', 'link_velocity'],
                               # 'joints': bigman_params['joint_ids']['UB']},
                               'joints': bigman_params['joint_ids'][self.task_hyperparams['body_part_sensed']]},

                              {'name': 'prev_cmd',
                               'type': 'prev_cmd'},
                              ]

        state_active = [{'name': 'joint_state',
                         'type': 'joint_state',
                         'fields': ['link_position', 'link_velocity'],
                         'joints': bigman_params['joint_ids'][self.task_hyperparams['body_part_sensed']]},

                        {'name': 'prev_cmd',
                         'type': 'prev_cmd'},
                        ]

        if body_part_active.upper() in ['LA', 'BA']:
            self.task_hyperparams['left_hand_rel_pose'] = create_hand_relative_pose([0, 0, 0, 1, 0, 0, 0],
                                                           hand_x=0.0,
                                                           hand_y=self.task_hyperparams['box_size'][1]/2-0.02,
                                                           hand_z=0.0,
                                                           hand_yaw=0)

            distance_left_arm = {'name': 'distance_left_arm',
                                 'type': 'fk_pose',
                                 'body_name': self.task_hyperparams['LH_name'],
                                 'body_offset': self.task_hyperparams['l_soft_hand_offset'],
                                 'target_offset': self.task_hyperparams['left_hand_rel_pose'],
                                 'fields': ['orientation', 'position']}

            observation_active.append(distance_left_arm.copy())
            state_active.append(distance_left_arm.copy())

        if body_part_active.upper() in ['RA', 'BA']:
            self.task_hyperparams['right_hand_rel_pose'] = create_hand_relative_pose([0, 0, 0, 1, 0, 0, 0],
                                                            hand_x=0.0,
                                                            hand_y=-self.task_hyperparams['box_size'][1]/2+0.02,
                                                            hand_z=0.0,
                                                            hand_yaw=0)

            distance_right_arm = {'name': 'distance_right_arm',
                                  'type': 'fk_pose',
                                  'body_name': self.task_hyperparams['RH_name'],
                                  'body_offset': self.task_hyperparams['r_soft_hand_offset'],
                                  'target_offset': self.task_hyperparams['right_hand_rel_pose'],
                                  'fields': ['orientation', 'position']},

            observation_active.append(distance_right_arm.copy())
            state_active.append(distance_right_arm.copy())

        others = [# {'name': 'ft_left_arm',
                  #  'type': 'fk_vel',
                  #  'ros_topic': None,
                  #  'body_name': LH_name,
                  #  'body_offset': l_soft_hand_offset,
                  #  'fields': ['orientation', 'position']},

                  # {'name': 'ft_left_arm',
                  #  'type': 'ft_sensor',
                  #  'ros_topic': '/xbotcore/bigman/ft/l_arm_ft',
                  #  'fields': ['force', 'torque']},

                  # {'name': 'ft_right_arm',
                  #  'type': 'ft_sensor',
                  #  'ros_topic': '/xbotcore/bigman/ft/r_arm_ft',
                  #  'fields': ['force', 'torque']},

                  # {'name': 'ft_left_leg',
                  #  'type': 'ft_sensor',
                  #  'ros_topic': '/xbotcore/bigman/ft/l_leg_ft',
                  #  'fields': ['force', 'torque']},

                  # {'name': 'ft_right_leg',
                  #  'type': 'ft_sensor',
                  #  'ros_topic': '/xbotcore/bigman/ft/r_leg_ft',
                  #  'fields': ['force', 'torque']},

                  # {'name': 'imu1',
                  #  'type': 'imu',
                  #  'ros_topic': '/xbotcore/bigman/imu/imu_link',
                  #  'fields': ['orientation', 'angular_velocity', 'linear_acceleration']},

                  # {'name': 'optitrack',
                  #  'type': 'optitrack',
                  #  'ros_topic': '/optitrack/relative_poses',
                  #  'fields': ['orientation', 'position'],
                  #  'bodies': ['box']}
                  ]

        observation_active += others

        # Reset Function
        reset_condition_bigman_box_gazebo_fcn = Reset_condition_bigman_box_gazebo()

        # Create a BIGMAN ROS EnvInterface
        BigmanEnv.__init__(self, interface=interface, mode='simulation',
                                 body_part_active=self.task_hyperparams['body_part_active'],
                                 command_type=self.task_hyperparams['command_type'],
                                 observation_active=observation_active,
                                 state_active=state_active,
                                 cmd_freq=int(1/self.task_hyperparams['Ts']),
                                 robot_dyn_model=self.task_hyperparams['robot_model'],
                                 reset_simulation_fcn=reset_condition_bigman_box_gazebo_fcn)
        # reset_simulation_fcn=reset_condition_bigman_box_gazebo)

    def get_task_hyperparams(self):
        return self.task_hyperparams