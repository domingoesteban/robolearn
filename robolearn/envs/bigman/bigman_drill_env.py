import os
import numpy as np
from robolearn.envs.bigman.bigman_env import BigmanEnv
from robolearn.utils.reach_drill_utils import Reset_condition_bigman_drill_gazebo
from robolearn.utils.reach_drill_utils import create_hand_relative_pose
from robolearn.utils.reach_drill_utils import create_drill_relative_pose
from robolearn.utils.reach_drill_utils import create_bigman_drill_condition
from robolearn.utils.reach_drill_utils import spawn_drill_gazebo
from robolearn.utils.robot_model import RobotModel
from robolearn.utils.iit.iit_robots_params import bigman_params


class BigmanDrillEnv(BigmanEnv):

    def __init__(self, dt=0.01):

        interface = 'ros'

        drill_x = 0.70
        drill_y = 0.00
        drill_z = -0.1327
        drill_z = -0.1075
        drill_yaw = 0  # Degrees
        #drill_size = [0.1, 0.1, 0.3]
        drill_size = [0.11, 0.11, 0.3]  # Beer
        final_drill_height = 0.0
        drill_relative_pose = create_drill_relative_pose(drill_x=drill_x,
                                                         drill_y=drill_y,
                                                         drill_z=drill_z,
                                                         drill_yaw=drill_yaw)
        self.drill_relative_poses = list()

        robot_urdf_file = os.environ["ROBOTOLOGY_ROOT"]+'/robots/iit-bigman-ros-pkg/bigman_urdf/urdf/bigman_robot_fixed.urdf'
        robot_model = RobotModel(robot_urdf_file)
        LH_name = 'LWrMot3'
        RH_name = 'RWrMot3'
        l_soft_hand_offset = np.array([0.000, -0.030, -0.210])
        r_soft_hand_offset = np.array([0.000, 0.030, -0.210])

        touching_drill_config = np.array([0.,  0.,  0.,  0.,  0.,  0.,
                                          0.,  0.,  0.,  0.,  0.,  0.,
                                          0.,  0.,  0.,
                                          0.0568,  0.2386, -0.2337, -1.6803,  0.2226,  0.0107,  0.5633,
                                          0.,  0.,
                                          0.0568,  -0.2386, 0.2337, -1.6803,  -0.2226,  0.0107,  -0.5633])

        # Robot configuration
        interface = 'ros'
        body_part_active = 'RA'
        body_part_sensed = 'RA'
        command_type = 'effort'

        if body_part_active == 'RA':
            hand_y = -drill_size[1]/2-0.02
            hand_z = drill_size[2]/2+0.02
            hand_name = RH_name
            hand_offset = r_soft_hand_offset
        else:
            hand_y = drill_size[1]/2+0.02
            hand_z = drill_size[2]/2+0.02
            hand_name = LH_name
            hand_offset = l_soft_hand_offset

        hand_rel_pose = create_hand_relative_pose([0, 0, 0, 1, 0, 0, 0],
                                                  hand_x=0.0,
                                                  hand_y=hand_y,
                                                  hand_z=hand_z,
                                                  hand_yaw=0)


        object_name = 'drill'
        object_rel_pose = create_hand_relative_pose([0, 0, 0, 1, 0, 0, 0],
                                                    hand_x=0.0,
                                                    hand_y=hand_y,
                                                    hand_z=hand_z,
                                                    hand_yaw=0)

        self.reset_drill_fcn = Reset_condition_bigman_drill_gazebo()

        observation_active = [{'name': 'joint_state',
                               'type': 'joint_state',
                               'ros_topic': '/xbotcore/bigman/joint_states',
                               # 'fields': ['link_position', 'link_velocity', 'effort'],
                               'fields': ['link_position', 'link_velocity'],
                               # 'joints': bigman_params['joint_ids']['UB']},
                               'joints': bigman_params['joint_ids'][body_part_sensed]},

                              {'name': 'prev_cmd',
                               'type': 'prev_cmd'},

                              # {'name': 'ft_right_arm',
                              #  'type': 'ft_sensor',
                              #  'ros_topic': '/xbotcore/bigman/ft/r_arm_ft',
                              #  'fields': ['force', 'torque']},

                              {'name': 'distance_hand',
                               'type': 'fk_pose',
                               'body_name': hand_name,
                               'body_offset': hand_offset,
                               'target_offset': hand_rel_pose,
                               'fields': ['orientation', 'position']},

                              {'name': 'distance_object',
                               'type': 'object_pose',
                               'body_name': object_name,
                               'target_rel_pose': drill_relative_pose,
                               'fields': ['orientation', 'position']},
                              ]

        state_active = [{'name': 'joint_state',
                         'type': 'joint_state',
                         'fields': ['link_position', 'link_velocity'],
                         'joints': bigman_params['joint_ids'][body_part_sensed]},

                        {'name': 'prev_cmd',
                         'type': 'prev_cmd'},

                        {'name': 'distance_hand',
                         'type': 'fk_pose',
                         'body_name': hand_name,
                         'body_offset': hand_offset,
                         'target_offset': hand_rel_pose,
                         'fields': ['orientation', 'position']},

                        {'name': 'distance_object',
                         'type': 'object_pose',
                         'body_name': object_name,
                         'target_rel_pose': drill_relative_pose,
                         'fields': ['orientation', 'position']},
                        ]

        self.env_params = {
            'temp_object_name': object_name,
            'state_active': state_active,
            'observation_active': observation_active,
            'hand_name': hand_name,
            'hand_offset': hand_offset,
            'robot_model': robot_model,
            'joint_ids': bigman_params['joint_ids'][body_part_active],
            'body_part_active': 'RA',
            'body_part_sensed': 'RA',
            'command_type': 'effort',
        }

        # Spawn Box first because it is simulation
        spawn_drill_gazebo(drill_relative_pose, drill_size=drill_size)

        BigmanEnv.__init__(self, interface, mode='simulation',
                           body_part_active=body_part_active,
                           command_type=command_type,
                           observation_active=observation_active,
                           state_active=state_active,
                           cmd_freq=int(1/dt),
                           robot_dyn_model=robot_model,
                           optional_env_params=self.env_params,
                           reset_simulation_fcn=self.reset_drill_fcn)

    def add_init_cond(self, condition):

        joints_idx = \
            bigman_params['joint_ids'][self.env_params['body_part_sensed']]
        q = np.array(bigman_params['q0'][3])
        des_q = np.deg2rad(condition[:7])
        drill_x = condition[7]
        drill_y = condition[8]
        # drill_z = -0.1327
        drill_z = -0.1075
        drill_yaw = condition[9]
        q[joints_idx] = des_q

        drill_pose = create_drill_relative_pose(drill_x=drill_x,
                                                drill_y=drill_y,
                                                drill_z=drill_z,
                                                drill_yaw=drill_yaw)
        env_condition = create_bigman_drill_condition(q,
                                                      drill_pose,
                                                      self.get_state_info(),
                                                      joint_idxs=joints_idx)

        self.add_condition(env_condition)
        self.reset_drill_fcn.add_reset_poses(drill_pose)

        self.drill_relative_poses.append(drill_pose)

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, action):
        self.interface.send_action(action=action)
        return self.interface.get_state()

