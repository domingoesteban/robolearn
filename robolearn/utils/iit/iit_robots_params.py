# ########## #
# ALL ROBOTS #
# ########## #
import numpy as np

# XBOT Joint state fields
joint_state_fields = ['link_position',
                      'link_velocity',
                      'effort']

#joint_state_fields = ['link_position',   # 0
#                      'motor_position',  # 1
#                      'link_velocity',   # 2
#                      'motor_velocity',  # 3
#                      'effort',          # 4
#                      'temperature',     # 5
#                      'stiffness',       # 6
#                      'damping',         # 7
#                      'aux']             # 8

# XBOT FT sensor fields (geometry_msgs/WrenchStamped)
ft_sensor_fields = ['force', 'torque']

ft_sensor_dof = {'force': 3,  # x, y, z
                 'torque': 3}  # x, y, z

# XBOT IMU sensor fields (sensor_msgs/Imu)
imu_sensor_fields = ['orientation',  # x, y, z, w
                     'angular_velocity',  # x, y, z
                     'linear_acceleration']  # x, y, z

imu_sensor_dof = {'orientation': 4,  # x, y, z, w
                  'angular_velocity': 3,  # x, y, z
                  'linear_acceleration': 3}  # x, y, z

# ROS OPTITRACK (robolearn_gazebo_envs PACKAGE)
optitrack_fields = ['position', 'orientation']

optitrack_dof = {'position': 3,  # x, y, z
                 'orientation': 4}  # x, y, z, w

# Observation example
#observation_active = [{'name': 'joint_state',
#                       'type': 'joint_state',
#                       'ros_topic': '/xbotcore/centauro/joint_states',
#                       'fields': ['link_position', 'link_velocity', 'effort'],
#                       'joints': range(12, 27)},  # Value that can be gotten from robot_params['joints_names']['UB']
#
#                      {'name': 'ft_arm2',
#                       'type': 'ft_sensor',
#                       'ros_topic': '/xbotcore/centauro/ft/ft_arm2',
#                       'fields': ['force', 'torque']},
#
#                      {'name': 'imu',
#                       'type': 'imu',
#                       'ros_topic': '/xbotcore/centauro/imu/ft_arm2',
#                       'fields': ['orientation', 'angular_velocity', 'linear_acceleration']}]

# ###### #
# BIGMAN #
# ###### #

bigman_params = {}

bigman_params['joint_state_fields'] = joint_state_fields

bigman_params['joints_names'] = ['LHipLat',        # Joint 0
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

bigman_params['joints_limits'] = [(-0.697778, 0.87222),               # 'LHipLat'         # Joint 0
                                  (-0.872222, 1.57),                  # 'LHipYaw',        # Joint 1
                                  (-2.09333, 1.046667),               # 'LHipSag',        # Joint 2
                                  (0.0, 2.442222),                    # 'LKneeSag',       # Joint 3
                                  (-1.39556, 0.697778),               # 'LAnkSag',        # Joint 4
                                  (-0.785, 0.785),                    # 'LAnkLat',        # Joint 5
                                  (-0.87222, 0.697778),               # 'RHipLat',        # Joint 6
                                  (-1.57, 0.872222),                  # 'RHipYaw',        # Joint 7
                                  (-2.09333, 1.046667),               # 'RHipSag',        # Joint 8
                                  (0.0, 2.442222),                    # 'RKneeSag',       # Joint 9
                                  (-1.39556, 0.697778),               # 'RAnkSag',        # Joint 10
                                  (-0.785, 0.785),                    # 'RAnkLat',        # Joint 11
                                  (-0.610865238198, 0.610865238198),  # 'WaistLat',       # Joint 12
                                  (-0.349065850399, 1.3962634016),    # 'WaistSag',       # Joint 13
                                  (-2.84488668075, 2.84488668075),    # 'WaistYaw',       # Joint 14
                                  (-2.87979326579,2.79252680319),     # 'LShSag',         # Joint 15
                                  (0.0872664625997, 3.85717764691),   # 'LShLat',         # Joint 16
                                  (-2.84488668075, 2.84488668075),    # 'LShYaw',         # Joint 17
                                  (-2.96705972839, 0.593411945678),   # 'LElbj',          # Joint 18
                                  (-2.529, 2.529),                    # 'LForearmPlate',  # Joint 19
                                  (-1.48, 1.48),                      # 'LWrj1',          # Joint 20
                                  (-1.48, 1.48),                      # 'LWrj2',          # Joint 21
                                  (-0.610865238198, 0.610865238198),  # 'NeckYawj',       # Joint 22
                                  (-0.261799387799, 1.0471975512),    # 'NeckPitchj',     # Joint 23
                                  (-2.84488668075, 2.80998009571),    # 'RShSag',         # Joint 24
                                  (-3.85717764691, -0.12217304764),   # 'RShLat',         # Joint 25
                                  (-2.86233997327, 2.82743338823),    # 'RShYaw',         # Joint 26
                                  (-2.96705972839, 0.593411945678),   # 'RElbj',          # Joint 27
                                  (-2.529, 2.529),                    # 'RForearmPlate',  # Joint 28
                                  (-1.48, 1.48),                      # 'RWrj1',          # Joint 29
                                  (-1.48, 1.48)]                      # 'RWrj2']          # Joint 30

bigman_params['joint_ids'] = {'LA': range(15, 22),
                              'RA': range(24, 31),
                              'BA': range(15, 22) + range(24, 31),
                              'TO': range(12, 15),
                              'HE': range(22, 24),
                              'UB': range(12, 15) + range(15, 22) + range(24, 31) + range(22, 24),
                              'LL': range(0, 6),
                              'RL': range(6, 12),
                              'LB': range(0, 12),
                              'WB': range(0, 31)}


bigman_params['bodies_names'] = ['ROOT',
                                 'LHipMot',
                                 'LThighUpLeg',
                                 'LThighLowLeg',
                                 'LLowLeg',
                                 'LFootmot',
                                 'LFoot',
                                 'RHipMot',
                                 'RThighUpLeg',
                                 'RThighLowLeg',
                                 'RLowLeg',
                                 'RFootmot',
                                 'RFoot',
                                 'DWL',
                                 'DWS',
                                 'DWYTorso',
                                 'LShp',
                                 'LShr',
                                 'LShy',
                                 'LElb',
                                 'LForearm',
                                 'LWrMot2',
                                 'LWrMot3',
                                 'NeckYaw',
                                 'NeckPitch',
                                 'RShp',
                                 'RShr',
                                 'RShy',
                                 'RElb',
                                 'RForearm',
                                 'RWrMot2',
                                 'RWrMot3']


bigman_params['q0'] = []   # A list of initial configurations
# Config 0
bigman_params['q0'].append([0,  # 'LHipLat'
                            0,  # 'LHipYaw'
                            0,  # 'LHipSag'
                            0,  # 'LKneeSag'
                            0,  # 'LAnkSag'
                            0,  # 'LAnkLat'
                            0,  # 'RHipLat'
                            0,  # 'RHipYaw'
                            0,  # 'RHipSag'
                            0,  # 'RKneeSag'
                            0,  # 'RAnkSag'
                            0,  # 'RAnkLat'
                            0,  # 'WaistLat'
                            0,  # 'WaistSag'
                            0,  # 'WaistYaw'
                            0,  # 'LShSag'
                            0,  # 'LShLat'
                            0,  # 'LShYaw'
                            0,  # 'LElbj'
                            0,  # 'LForearmPlate'
                            0,  # 'LWrj1'
                            0,  # 'LWrj2'
                            0,  # 'NeckYawj'
                            0,  # 'NeckPitchj'
                            0,  # 'RShSag'
                            0,  # 'RShLat'
                            0,  # 'RShYaw'
                            0,  # 'RElbj'
                            0,  # 'RForearmPlate'
                            0,  # 'RWrj1'
                            0])  # 'RWrj2'

# Conf1
bigman_params['q0'].append([0,  # 'LHipLat'       #0
                            0,  # 'LHipYaw'       #1
                            0,  # 'LHipSag'       #2
                            0,  # 'LKneeSag'      #3
                            0,  # 'LAnkSag'       #4
                            0,  # 'LAnkLat'       #5
                            0,  # 'RHipLat'       #6
                            0,  # 'RHipYaw'       #7
                            0,  # 'RHipSag'       #8
                            0,  # 'RKneeSag'      #9
                            0,  # 'RAnkSag'       #10
                            0,  # 'RAnkLat'       #11
                            0,  # 'WaistLat'      #12
                            0,  # 'WaistSag'      #13
                            0,  # 'WaistYaw'      #14
                            0,  # 'LShSag'        #15
                            np.array(50),  # 'LShLat'        #16
                            0,  # 'LShYaw'        #17
                            0,  # 'LElbj'         #18
                            0,  # 'LForearmPlate' #19
                            0,  # 'LWrj1'         #20
                            0,  # 'LWrj2'         #21
                            0,  # 'NeckYawj'      #22
                            0,  # 'NeckPitchj'    #23
                            0,  # 'RShSag'        #24
                            np.array(-50),  # 'RShLat'        #25
                            0,  # 'RShYaw'        #26
                            0,  # 'RElbj'         #27
                            0,  # 'RForearmPlate' #28
                            0,  # 'RWrj1'         #29
                            0])  # 'RWrj2'        #30

# Conf3: 'B' pose
bigman_params['q0'].append([0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0,
                            0, 1.5708, 0, -1.5708, 0, 0, 0,
                            0, 0,
                            0, -1.5708, 0, -1.5708, 0, 0, 0])

bigman_params['observation_active'] = [{'name': 'joint_state',
                                        'type': 'joint_state',
                                        'ros_topic': '/xbotcore/centauro/joint_states',
                                        'fields': ['link_position', 'link_velocity', 'effort'],
                                        'joints': range(0, 31)},  # Value that can be gotten from robot_params['joints_names']['UB']

                                       {'name': 'ft_left_arm',
                                        'type': 'ft_sensor',
                                        'ros_topic': '/xbotcore/bigman/ft/l_arm_ft',
                                        'fields': ['force', 'torque']},

                                       {'name': 'ft_right_arm',
                                        'type': 'ft_sensor',
                                        'ros_topic': '/xbotcore/bigman/ft/r_arm_ft',
                                        'fields': ['force', 'torque']},

                                       {'name': 'ft_left_leg',
                                        'type': 'ft_sensor',
                                        'ros_topic': '/xbotcore/bigman/ft/l_leg_ft',
                                        'fields': ['force', 'torque']},

                                       {'name': 'ft_right_leg',
                                        'type': 'ft_sensor',
                                        'ros_topic': '/xbotcore/bigman/ft/r_leg_ft',
                                        'fields': ['force', 'torque']},

                                       {'name': 'imu',
                                        'type': 'imu',
                                        'ros_topic': '/xbotcore/bigman/imu/imu_link',
                                        'fields': ['orientation', 'angular_velocity', 'linear_acceleration']}]

bigman_params['state_active'] = [{'name': 'joint_state',
                                  'type': 'joint_state',
                                  'fields': ['link_position', 'link_velocity'],
                                  'joints': range(0, 31)}]  # Value that can be gotten from robot_params['joints_names']['UB']

# ######## #
# CENTAURO #
# ######## #

centauro_params = {}

#centauro_params['joint_state_fields'] = joint_state_fields
#centauro_params['ft_sensor_fields'] = ft_sensor_fields

centauro_params['joints_names'] = ['hip_yaw_1',     # Joint 0
                                   'hip_pitch_1',   # Joint 1
                                   'knee_pitch_1',  # Joint 2
                                   'hip_yaw_2',     # Joint 3
                                   'hip_pitch_2',   # Joint 4
                                   'knee_pitch_2',  # Joint 5
                                   'hip_yaw_3',     # Joint 6
                                   'hip_pitch_3',   # Joint 7
                                   'knee_pitch_3',  # Joint 8
                                   'hip_yaw_4',     # Joint 9
                                   'hip_pitch_4',   # Joint 10
                                   'knee_pitch_4',  # Joint 11
                                   'torso_yaw',     # Joint 12
                                   'j_arm1_1',      # Joint 13
                                   'j_arm1_2',      # Joint 14
                                   'j_arm1_3',      # Joint 15
                                   'j_arm1_4',      # Joint 16
                                   'j_arm1_5',      # Joint 17
                                   'j_arm1_6',      # Joint 18
                                   'j_arm1_7',      # Joint 19
                                   'j_arm2_1',      # Joint 20
                                   'j_arm2_2',      # Joint 21
                                   'j_arm2_3',      # Joint 22
                                   'j_arm2_4',      # Joint 23
                                   'j_arm2_5',      # Joint 24
                                   'j_arm2_6',      # Joint 25
                                   'j_arm2_7']      # Joint 26

centauro_params['joint_ids'] = {'LA': range(13, 20),
                                'RA': range(20, 27),
                                'BA': range(13, 27),
                                'TO': [12],
                                'UB': range(12, 27),
                                'L1': range(0, 3),
                                'L2': range(3, 6),
                                'L3': range(6, 9),
                                'L4': range(9, 12),
                                'LEGS': range(0, 12),
                                'WB': range(0, 27)}

centauro_params['q0'] = []   # A list of initial configurations
centauro_params['q0'].append([0,  # 'hip_yaw_1',     # Joint 0
                              0,  # 'hip_pitch_1',   # Joint 1
                              0,  # 'knee_pitch_1',  # Joint 2
                              0,  # 'hip_yaw_2',     # Joint 3
                              0,  # 'hip_pitch_2',   # Joint 4
                              0,  # 'knee_pitch_2',  # Joint 5
                              0,  # 'hip_yaw_3',     # Joint 6
                              0,  # 'hip_pitch_3',   # Joint 7
                              0,  # 'knee_pitch_3',  # Joint 8
                              0,  # 'hip_yaw_4',     # Joint 9
                              0,  # 'hip_pitch_4',   # Joint 10
                              0,  # 'knee_pitch_4',  # Joint 11
                              0,  # 'torso_yaw',     # Joint 12
                              0,  # 'j_arm1_1',      # Joint 13
                              0,  # 'j_arm1_2',      # Joint 14
                              0,  # 'j_arm1_3',      # Joint 15
                              0,  # 'j_arm1_4',      # Joint 16
                              0,  # 'j_arm1_5',      # Joint 17
                              0,  # 'j_arm1_6',      # Joint 18
                              0,  # 'j_arm1_7',      # Joint 19
                              0,  # 'j_arm2_1',      # Joint 20
                              0,  # 'j_arm2_2',      # Joint 21
                              0,  # 'j_arm2_3',      # Joint 22
                              0,  # 'j_arm2_4',      # Joint 23
                              0,  # 'j_arm2_5',      # Joint 24
                              0,  # 'j_arm2_6',      # Joint 25
                              0])  # 'j_arm2_7']      # Joint 26


centauro_params['observation_active'] = [{'name': 'joint_state',
                       'type': 'joint_state',
                       'ros_topic': '/xbotcore/centauro/joint_states',
                       'fields': ['link_position', 'link_velocity', 'effort'],
                       'joints': range(12, 27)},  # Value that can be gotten from robot_params['joints_names']['UB']

                      {'name': 'ft_arm2',
                       'type': 'ft_sensor',
                       'ros_topic': '/xbotcore/centauro/ft/ft_arm2',
                       'fields': ['force', 'torque']}]

centauro_params['state_active'] = [{'name': 'joint_state',
                                    'type': 'joint_state',
                                    'fields': ['link_position', 'link_velocity'],
                                    'joints': range(12, 27)}]  # Value that can be gotten from robot_params['joints_names']['UB']
