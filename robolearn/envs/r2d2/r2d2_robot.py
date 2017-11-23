import os
import numpy as np
from robolearn.envs.pybullet.pybullet_robot import PyBulletRobot


class R2D2Robot(PyBulletRobot):
    def __init__(self, robot_name='r2d2', init_pos=None, self_collision=True):

        urdf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'urdf/r2d2.urdf')

        self.act_dim = 15
        self.obs_dim = 30

        self.power = 0.1

        super(R2D2Robot, self).__init__('urdf', urdf_xml, robot_name, self.act_dim, self.obs_dim, init_pos=init_pos,
                                        self_collision=self_collision)


    def robot_reset(self):
        self.total_joints = len(self.ordered_joints)
        self.state_per_joint = 2
        self.initial_state = np.zeros(self.obs_dim)

        for jj, joint in enumerate(self.ordered_joints):
            joint.reset_position(self.initial_state[jj], self.initial_state[self.total_joints+jj])

    def robot_state(self):
        self.total_joints = len(self.ordered_joints)
        self.state_per_joint = 2
        self.initial_state = np.zeros(self.obs_dim)

        state = np.zeros(self.state_per_joint*len(self.ordered_joints))

        for jj, joint in enumerate(self.ordered_joints):
            state[[jj, self.total_joints+jj]] = joint.current_position()

        return state

    def appy_action(self, action):
        assert (np.isfinite(action).all())
        for n, joint in enumerate(self.ordered_joints):
            joint.set_motor_torque(self.power * joint.power_coef * float(np.clip(action[n], -1, +1)))
            print(self.power * joint.power_coef * float(np.clip(action[n], -1, +1)), joint.power_coef, self.power)


