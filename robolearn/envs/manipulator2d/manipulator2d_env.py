from robolearn.envs.gazebo_ros_env import GazeboROSEnv

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState


class Manipulator2dEnv(GazeboROSEnv):

    def __init__(self, host='localhost', roscore_port=None, gzserver_port=None):

        super(Manipulator2dEnv, self).__init__(host=host, roscore_port=roscore_port, gzserver_port=gzserver_port)

        self.action_types = list()
        self.action_topic_infos = list()
        self.observation_active = list()
        self.state_active = list()
        for ii in range(3):
            self.action_types.append({'name': 'joint_effort',
                                      'dof': 1})
            self.action_topic_infos.append({'name': '/manipulator2d/joint'+str(ii)+'_position_controller/command',
                                            'type': Float64,
                                            'freq': 100})
        self.observation_active.append({'name': 'joint_state',
                                        'type': 'joint_state',
                                        'ros_class': JointState,
                                        'fields': ['position', 'velocity'],
                                        'joints': [0, 1, 2],  # Joint IDs
                                        'ros_topic': '/manipulator2d/joint_states',
                                        })
        self.state_active.append({'name': 'joint_state',
                                  'type': 'joint_state',
                                  'fields': ['position', 'velocity'],
                                  'joints': [0, 1, 2],  # Joint IDs
                                  })

