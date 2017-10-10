from robolearn.envs.gazebo_ros_env import GazeboROSEnv

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState


class Manipulator2dEnv(GazeboROSEnv):

    def __init__(self, host='localhost', roscore_port=None, gzserver_port=None):

        ros_actions = list()
        for ii in range(3):
            ros_actions.append({'name': 'joint_effort'+str(ii),
                                'type': 'joint_effort',
                                'freq': 100,
                                'dof': 1,
                                'ros_class': Float64,
                                'ros_topic': '/manipulator2d/joint'+str(ii)+'_position_controller/command',
                                })

        ros_observations = [{'name': 'joint_state',
                             'type': 'joint_state',
                             'ros_class': JointState,
                             'fields': ['position', 'velocity'],
                             'joints': [0, 1, 2],  # Joint IDs
                             'ros_topic': '/manipulator2d/joint_states',
                             'active': True,
                             }
                            ]

        state_active = [{'name': 'joint_state',
                         'type': 'joint_state',
                         'fields': ['position', 'velocity'],
                         'joints': [0, 1, 2],  # Joint IDs
                         }
                        ]

        ros_commands = [['roslaunch',
                         'manipulator2d_gazebo',
                         'manipulator2d_world.launch']
                        ]

        super(Manipulator2dEnv, self).__init__(ros_actions=ros_actions,
                                               ros_observations=ros_observations,
                                               state_active=state_active,
                                               host=host, roscore_port=roscore_port, gzserver_port=gzserver_port,
                                               ros_commands=ros_commands)
