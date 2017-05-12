# ########## #
# ALL ROBOTS #
# ########## #

# XBOT Joint state fields
joint_state_fields = ['link_position',
                      'motor_position',
                      'link_velocity',
                      'motor_velocity',
                      'effort',
                      'temperature',
                      'stiffness',
                      'damping',
                      'aux']

# XBOT FT sensor fields (geometry_msgs/WrenchStamped)
ft_sensor_fields = ['force',  # x, y, z
                    'torque']  # x, y, z

# XBOT IMU sensor fields (sensor_msgs/Imu)
imu_sensor_fields = ['orientation',  # x, y, z, w
                     'angular_velocity',  # x, y, z
                     'linear_acceleration']  # x, y, z


# ###### #
# BIGMAN #
# ###### #

bigman_params = {}

bigman_params['joint_state_fields'] = joint_state_fields

bigman_params['joints_names'] = ['LHipLat',  # Joint 0
                                 'LHipYaw',  # Joint 1
                                 'LHipSag',  # Joint 2
                                 'LKneeSag',  # Joint 3
                                 'LAnkSag',  # Joint 4
                                 'LAnkLat',  # Joint 5
                                 'RHipLat',  # Joint 6
                                 'RHipYaw',  # Joint 7
                                 'RHipSag',  # Joint 8
                                 'RKneeSag',  # Joint 9
                                 'RAnkSag',  # Joint 10
                                 'RAnkLat',  # Joint 11
                                 'WaistLat',  # Joint 12
                                 'WaistSag',  # Joint 13
                                 'WaistYaw',  # Joint 14
                                 'LShSag',  # Joint 15
                                 'LShLat',  # Joint 16
                                 'LShYaw',  # Joint 17
                                 'LElbj',  # Joint 18
                                 'LForearmPlate',  # Joint 19
                                 'LWrj1',  # Joint 20
                                 'LWrj2',  # Joint 21
                                 'NeckYawj',  # Joint 22
                                 'NeckPitchj',  # Joint 23
                                 'RShSag',  # Joint 24
                                 'RShLat',  # Joint 25
                                 'RShYaw',  # Joint 26
                                 'RElbj',  # Joint 27
                                 'RForearmPlate',  # Joint 28
                                 'RWrj1',  # Joint 29
                                 'RWrj2']  # Joint 30

bigman_params['body_parts'] = {'LA': range(15, 22),
                               'RA': range(24, 31),
                               'BA': range(15, 22) + range(24, 31),
                               'TO': range(12, 15),
                               'HE': range(22, 24),
                               'UB': range(12, 15) + range(15, 22) + range(24, 31) + range(22, 24),
                               'LL': range(0, 6),
                               'RL': range(6, 12),
                               'LB': range(0, 12),
                               'WB': range(0, 31)}


# ######## #
# CENTAURO #
# ######## #

centauro_params = {}

centauro_params['joint_state_fields'] = joint_state_fields

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

centauro_params['body_parts'] = {'LA': range(13, 20),
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

