import os
import numpy as np
from robolearn.envs.pybullet.pybullet_robot import PyBulletRobot


class ReacherMJCRobot(PyBulletRobot):
    def __init__(self, robot_name='reacher', self_collision=True):

        mjc_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/reacher_robot.xml')

        self.act_dim = 2
        self.obs_dim = 4

        self.power = 0.01

        super(ReacherMJCRobot, self).__init__('mjcf', mjc_xml, robot_name, self.act_dim, self.obs_dim,
                                              self_collision=self_collision)

    def robot_reset(self):
        self.total_joints = len(self.ordered_joints)
        self.state_per_joint = 2
        self.initial_state = np.zeros(self.obs_dim)

        for jj, joint in enumerate(self.ordered_joints):
            joint.reset_position(self.initial_state[jj], self.initial_state[self.total_joints+jj])

        # Color
        color_list = [[0.0, 0.4, 0.6, 1] for _ in range(self.get_total_bodies())]
        color_list[0] = [0.9, 0.4, 0.6, 1]
        color_list[-1] = [0.0, 0.8, 0.6, 1]
        self.set_body_colors(color_list)

    def robot_state(self):
        self.total_joints = len(self.ordered_joints)
        self.state_per_joint = 2
        self.initial_state = np.zeros(self.obs_dim)

        state = np.zeros(self.state_per_joint*len(self.ordered_joints))

        for jj, joint in enumerate(self.ordered_joints):
            state[[jj, self.total_joints+jj]] = joint.current_position()

        return state

    def apply_action(self, action):
        assert (np.isfinite(action).all())
        for n, joint in enumerate(self.ordered_joints):
            joint.set_motor_torque(self.power * joint.power_coef * float(np.clip(action[n], -1, +1)))
            # print(self.power * joint.power_coef * float(np.clip(action[n], -1, +1)), joint.power_coef, self.power)


