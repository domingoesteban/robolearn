# ########## #
# ALL ROBOTS #
# ########## #
import numpy as np

# XBOT Joint state fields
xbot_joint_state_fields = ['link_position',
                           'link_velocity',
                           'effort']

# joint_state_fields = ['link_position',   # 0
#                       'motor_position',  # 1
#                       'link_velocity',   # 2
#                       'motor_velocity',  # 3
#                       'effort',          # 4
#                       'temperature',     # 5
#                       'stiffness',       # 6
#                       'damping',         # 7
#                       'aux']             # 8

# Observation example
# observation_active = [{'name': 'joint_state',
#                        'type': 'joint_state',
#                        'ros_topic': '/xbotcore/centauro/joint_states',
#                        'fields': ['link_position', 'link_velocity', 'effort'],
#                        'joints': range(12, 27)},  # Value that can be gotten from robot_params['joints_names']['UB']
#
#                       {'name': 'ft_arm2',
#                        'type': 'ft_sensor',
#                        'ros_topic': '/xbotcore/centauro/ft/ft_arm2',
#                        'fields': ['force', 'torque']},
#
#                       {'name': 'imu',
#                        'type': 'imu',
#                       'ros_topic': '/xbotcore/centauro/imu/ft_arm2',
#                       'fields': ['orientation', 'angular_velocity', 'linear_acceleration']}]

# ###### #
# BIGMAN #
# ###### #

bigman_params = dict()

bigman_params['joint_state_fields'] = xbot_joint_state_fields

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
                                  (-2.87979326579, 2.79252680319),    # 'LShSag',         # Joint 15
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

bigman_params['effort_limits'] = [400,  # 'LHipLat'         # Joint 0
                                  140,  # 'LHipYaw',        # Joint 1
                                  400,  # 'LHipSag',        # Joint 2
                                  400,  # 'LKneeSag',       # Joint 3
                                  330,  # 'LAnkSag',        # Joint 4
                                  210,  # 'LAnkLat',        # Joint 5
                                  400,  # 'RHipLat',        # Joint 6
                                  140,  # 'RHipYaw',        # Joint 7
                                  400,  # 'RHipSag',        # Joint 8
                                  400,  # 'RKneeSag',       # Joint 9
                                  330,  # 'RAnkSag',        # Joint 10
                                  210,  # 'RAnkLat',        # Joint 11
                                  120,  # 'WaistLat',       # Joint 12
                                  220,  # 'WaistSag',       # Joint 13
                                  120,  # 'WaistYaw',       # Joint 14
                                  120,  # 'LShSag',         # Joint 15
                                  120,  # 'LShLat',         # Joint 16
                                  120,  # 'LShYaw',         # Joint 17
                                  120,  # 'LElbj',          # Joint 18
                                  60,   # 'LForearmPlate',  # Joint 19
                                  60,   # 'LWrj1',          # Joint 20
                                  60,   # 'LWrj2',          # Joint 21
                                  120,  # 'NeckYawj',       # Joint 22
                                  120,  # 'NeckPitchj',     # Joint 23
                                  120,  # 'RShSag',         # Joint 24
                                  120,  # 'RShLat',         # Joint 25
                                  120,  # 'RShYaw',         # Joint 26
                                  120,  # 'RElbj',          # Joint 27
                                  60,   # 'RForearmPlate',  # Joint 28
                                  60,   # 'RWrj1',          # Joint 29
                                  60]   # 'RWrj2']          # Joint 30

bigman_params['joint_ids'] = {'LA': range(15, 22),
                              'RA': range(24, 31),
                              'BA': range(15, 22) + range(24, 31),
                              'TO': range(12, 15),
                              'HE': range(22, 24),
                              'UB': range(12, 15) + range(15, 22) + range(24, 31),
                              'UBH': range(12, 15) + range(15, 22) + range(24, 31) + range(22, 24),
                              'LL': range(0, 6),
                              'RL': range(6, 12),
                              'LB': range(0, 12),
                              'WB': range(0, 31)}


bigman_params['bodies_names'] = ['ROOT',            # 0
                                 'LHipMot',         # 1
                                 'LThighUpLeg',     # 2
                                 'LThighLowLeg',    # 3
                                 'LLowLeg',         # 4
                                 'LFootmot',        # 5
                                 'LFoot',           # 6
                                 'RHipMot',         # 7
                                 'RThighUpLeg',     # 8
                                 'RThighLowLeg',    # 9
                                 'RLowLeg',         # 10
                                 'RFootmot',        # 11
                                 'RFoot',           # 12
                                 'DWL',             # 13
                                 'DWS',             # 14
                                 'DWYTorso',        # 15
                                 'LShp',            # 16
                                 'LShr',            # 17
                                 'LShy',            # 18
                                 'LElb',            # 19
                                 'LForearm',        # 20
                                 'LWrMot2',         # 21
                                 'LWrMot3',         # 22
                                 'NeckYaw',         # 23
                                 'NeckPitch',       # 24
                                 'RShp',            # 25
                                 'RShr',            # 26
                                 'RShy',            # 27
                                 'RElb',            # 28
                                 'RForearm',        # 29
                                 'RWrMot2',         # 30
                                 'RWrMot3']         # 31


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
                            np.deg2rad(50),  # 'LShLat'        #16
                            0,  # 'LShYaw'        #17
                            0,  # 'LElbj'         #18
                            0,  # 'LForearmPlate' #19
                            0,  # 'LWrj1'         #20
                            0,  # 'LWrj2'         #21
                            0,  # 'NeckYawj'      #22
                            0,  # 'NeckPitchj'    #23
                            0,  # 'RShSag'        #24
                            np.deg2rad(-50),  # 'RShLat'        #25
                            0,  # 'RShYaw'        #26
                            0,  # 'RElbj'         #27
                            0,  # 'RForearmPlate' #28
                            0,  # 'RWrj1'         #29
                            0])  # 'RWrj2'        #30

# Conf2: 'B' pose
bigman_params['q0'].append([0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0,
                            0, 1.5708, 0, -1.5708, 0, 0, 0,
                            0, 0,
                            0, -1.5708, 0, -1.5708, 0, 0, 0])

# Conf3: 'Homing'
bigman_params['q0'].append([-0.06, 0.0, -0.45, 0.9, -0.45, 0.06,
                            0.06, 0.0, -0.45, 0.9, -0.45, -0.06,
                            0, 0, 0,
                            1.1, 0.2, -0.3, -2.0, 0.0, -0.0, -0.0,
                            0, 0.6,
                            1.1, -0.2, 0.3, -2.0, 0.0, -0.0, -0.0])

bigman_params['stiffness_gains'] = np.array([8000.,  5000.,  8000.,  5000.,  5000.,  2000.,
                                             8000.,  5000.,  5000.,  5000.,  5000.,  2000.,
                                             5000.,  8000.,  5000.,
                                             5000.,  8000.,  5000.,  5000.,   300.,  2000.,   300.,
                                             300.,   300.,
                                             5000.,  8000.,  5000.,  5000.,   300.,  2000.,   300.])
bigman_params['damping_gains'] = np.array([30.,  50.,  30.,  30.,  30.,   5.,
                                           30.,  50.,  30.,  30.,  30.,   5.,
                                           30.,  50.,  30.,
                                           30.,  50.,  30.,  30.,   1.,   5.,   1.,
                                           1.,   1.,
                                           30.,  50.,  30.,  30.,   1.,   5.,   1.])

bigman_params['observation_active'] = [{'name': 'joint_state',
                                        'type': 'joint_state',
                                        'ros_topic': '/xbotcore/centauro/joint_states',
                                        'fields': ['link_position', 'link_velocity', 'effort'],
                                        'joints': range(0, 31)},

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
                                  'joints': range(0, 31)}]  # It can be gotten from robot_params['joints_names']['UB']

# ######## #
# CENTAURO #
# ######## #

centauro_params = dict()

centauro_params['joints_names'] = ['hip_yaw_1',      # Joint 0
                                   'hip_pitch_1',    # Joint 1
                                   'knee_pitch_1',   # Joint 2
                                   'ankle_pitch_1',  # Joint 3
                                   'ankle_yaw_1',    # Joint 4
                                   'j_wheel_1',      # Joint 5
                                   'hip_yaw_2',      # Joint 6
                                   'hip_pitch_2',    # Joint 7
                                   'knee_pitch_2',   # Joint 8
                                   'ankle_pitch_2',  # Joint 9
                                   'ankle_yaw_2',    # Joint 10
                                   'j_wheel_2',      # Joint 11
                                   'hip_yaw_3',      # Joint 12
                                   'hip_pitch_3',    # Joint 13
                                   'knee_pitch_3',   # Joint 14
                                   'ankle_pitch_3',  # Joint 15
                                   'ankle_yaw_3',    # Joint 16
                                   'j_wheel_3',      # Joint 17
                                   'hip_yaw_4',      # Joint 18
                                   'hip_pitch_4',    # Joint 19
                                   'knee_pitch_4',   # Joint 20
                                   'ankle_pitch_4',  # Joint 21
                                   'ankle_yaw_4',    # Joint 22
                                   'j_wheel_4',      # Joint 23
                                   'torso_yaw',      # Joint 24
                                   'j_arm1_1',       # Joint 25
                                   'j_arm1_2',       # Joint 26
                                   'j_arm1_3',       # Joint 27
                                   'j_arm1_4',       # Joint 28
                                   'j_arm1_5',       # Joint 29
                                   'j_arm1_6',       # Joint 30
                                   'j_arm1_7',       # Joint 31
                                   'j_ft_1',         # Joint 31
                                   'j_arm1_8',         # Joint 31
                                   'j_arm2_1',       # Joint 32
                                   'j_arm2_2',       # Joint 33
                                   'j_arm2_3',       # Joint 34
                                   'j_arm2_4',       # Joint 35
                                   'j_arm2_5',       # Joint 36
                                   'j_arm2_6',       # Joint 37
                                   'j_arm2_7'      # Joint 38
                                   'j_ft_2',         # Joint 31
                                   'j_arm2_8',         # Joint 31
                                   'neck_velodyne',
                                   'neck_yaw',
                                   'neck_pitch',
                                   ]

centauro_params['joint_ids'] = {'LA': list(range(25, 32)),
                                'RA': list(range(32, 39)),
                                'BA': list(range(25, 39)),
                                'TO': [24],
                                'UB': list(range(24, 39)),
                                'L1': list(range(0, 6)),
                                'L2': list(range(6, 12)),
                                'L3': list(range(12, 18)),
                                'L4': list(range(18, 24)),
                                'LEGS': list(range(0, 12)),
                                'WB': list(range(0, 27))}

centauro_params['q0'] = []   # A list of initial configurations
centauro_params['q0'].append([0,  # 'hip_yaw_1',      # Joint 0
                              0,  # 'hip_pitch_1',    # Joint 1
                              0,  # 'knee_pitch_1',   # Joint 2
                              0,  # 'ankle_pitch_1',  # Joint 3
                              0,  # 'ankle_yaw_1',    # Joint 4
                              0,  # 'j_wheel_1',      # Joint 5
                              0,  # 'hip_yaw_2',      # Joint 6
                              0,  # 'hip_pitch_2',    # Joint 7
                              0,  # 'knee_pitch_2',   # Joint 8
                              0,  # 'ankle_pitch_2',  # Joint 9
                              0,  # 'ankle_yaw_2',    # Joint 10
                              0,  # 'j_wheel_2',      # Joint 11
                              0,  # 'hip_yaw_3',      # Joint 12
                              0,  # 'hip_pitch_3',    # Joint 13
                              0,  # 'knee_pitch_3',   # Joint 14
                              0,  # 'ankle_pitch_3',  # Joint 15
                              0,  # 'ankle_yaw_3',    # Joint 16
                              0,  # 'j_wheel_3',      # Joint 17
                              0,  # 'hip_yaw_4',      # Joint 18
                              0,  # 'hip_pitch_4',    # Joint 19
                              0,  # 'knee_pitch_4',   # Joint 20
                              0,  # 'ankle_pitch_4',  # Joint 21
                              0,  # 'ankle_yaw_4',    # Joint 22
                              0,  # 'j_wheel_4',      # Joint 23
                              0,  # 'torso_yaw',      # Joint 24
                              0,  # 'j_arm1_1',       # Joint 25
                              0,  # 'j_arm1_2',       # Joint 26
                              0,  # 'j_arm1_3',       # Joint 27
                              0,  # 'j_arm1_4',       # Joint 28
                              0,  # 'j_arm1_5',       # Joint 29
                              0,  # 'j_arm1_6',       # Joint 30
                              0,  # 'j_arm1_7',       # Joint 31
                              0,  # 'j_arm2_1',       # Joint 32
                              0,  # 'j_arm2_2',       # Joint 33
                              0,  # 'j_arm2_3',       # Joint 34
                              0,  # 'j_arm2_4',       # Joint 35
                              0,  # 'j_arm2_5',       # Joint 36
                              0,  # 'j_arm2_6',       # Joint 37
                              0])  # 'j_arm2_7']      # Joint 38


centauro_upper_body_params = dict()

centauro_upper_body_params['joints_names'] = [
    'torso_yaw',      # Joint 0
    'j_arm1_1',       # Joint 1
    'j_arm1_2',       # Joint 2
    'j_arm1_3',       # Joint 3
    'j_arm1_4',       # Joint 4
    'j_arm1_5',       # Joint 5
    'j_arm1_6',       # Joint 6
    'j_arm1_7',       # Joint 7
    'j_arm2_1',       # Joint 8
    'j_arm2_2',       # Joint 9
    'j_arm2_3',       # Joint 10
    'j_arm2_4',       # Joint 11
    'j_arm2_5',       # Joint 12
    'j_arm2_6',       # Joint 13
    'j_arm2_7'      # Joint 14
]
centauro_upper_body_params['q0'] = []
centauro_upper_body_params['q0'].append([
    0,
    0, -0.3, -0.8, -1.2, 0, -0.8, 0,
    0, 0.3, 0.8, 1.2, 0, 0.8, 0,
])

centauro_upper_body_params['joint_ids'] = {'LA': list(range(1, 8)),
                                            'RA': list(range(8, 15)),
                                            'BA': list(range(1, 15)),
                                            'TO': [0],
                                            'UB': list(range(15)),
                                           }


centauro_params['observation_active'] = [{'name': 'joint_state',
                                          'type': 'joint_state',
                                          'ros_topic': '/xbotcore/centauro/joint_states',
                                          'fields': ['link_position', 'link_velocity', 'effort'],
                                          'joints': range(12, 27)},

                                         {'name': 'ft_arm2',
                                          'type': 'ft_sensor',
                                          'ros_topic': '/xbotcore/centauro/ft/ft_arm2',
                                          'fields': ['force', 'torque']}]

centauro_params['state_active'] = [{'name': 'joint_state',
                                    'type': 'joint_state',
                                    'fields': ['link_position', 'link_velocity'],
                                    'joints': range(12, 27)}]
