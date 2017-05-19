# ########## #
# ALL ROBOTS #
# ########## #

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

bigman_params['q0'] = []   # A list of initial configurations
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
