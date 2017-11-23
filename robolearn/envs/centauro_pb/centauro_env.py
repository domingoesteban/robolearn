import os
import numpy as np
import pybullet as pb
import gym
from gym import spaces
from gym.utils import seeding
from robolearn.envs.centauro_pb.centauro_robot import CentauroBulletRobot

from robolearn.envs.pybullet.bullet_env import BulletEnv


class CentauroBulletEnv(BulletEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, render=False):

        # Environment/Scene
        self.done = False
        gravity = 9.8
        sim_timestep = 1./1000.
        frame_skip = 10

        # Robot
        # init_pos = [0, 0, 1.12]
        # init_pos = [0, 0, 1.05]
        # init_pos = [0, 0, 1.041]
        init_pos = [0, 0, 0.7975]
        self._robot = CentauroBulletRobot(init_pos=init_pos, control_type='velocity', self_collision=False)

        self.action_bounds = [(-np.pi, np.pi) for _ in range(self._robot.action_dim)]
        self._act_dim = self._robot.action_dim + 0
        self._obs_dim = self._robot.observation_dim + 0

        super(CentauroBulletEnv, self).__init__(gravity=gravity, sim_timestep=sim_timestep, frameskip=frame_skip,
                                                render=render)

        # Environment settings
        self.max_time = 10  # s
        self.time_counter = 0
        self._observation = np.zeros(self.observation_space.shape)

        # Sensors
        self._cameras = list()

    def reset_model(self):
        self.done = False
        self.time_counter = 0

        # xml_env = '/home/domingo/robotlearning-workspace/learning-modules/robolearn/robolearn/envs/pybullet/mjcf/reacher_world.xml'
        # pb.loadMJCF(xml_env)

        # plane_urdf = '/home/domingo/robotlearning-workspace/learning-modules/robolearn/robolearn/envs/pybullet/plane/plane.urdf'
        plane_urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/plane/plane.urdf')
        plane_uid = pb.loadURDF(plane_urdf)

        # table_sdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/big_support/model.sdf')
        # self.table_uid = pb.loadSDF(table_sdf)
        # pb.changeVisualShape(self.table_uid[-1], -1, rgbaColor=[0.38, 0.4, 0.42, 1])
        # pb.resetBasePositionAndOrientation(self.table_uid[-1], (1.2, 0, 0.001), (0, 0, 0, 1))
        table_urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/big_support/model.urdf')
        self.table_uid = pb.loadURDF(table_urdf)
        pb.changeVisualShape(self.table_uid, -1, rgbaColor=[0.38, 0.4, 0.42, 1])
        # pos, ori = pb.getBasePositionAndOrientation(ii)
        pb.resetBasePositionAndOrientation(self.table_uid, (1.2, 0, 0.001), (0, 0, 0, 1))

        drill_sdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/cordless_drill/model.sdf')
        self.drill_uid = pb.loadSDF(drill_sdf)
        pb.changeVisualShape(self.drill_uid[-1], -1, rgbaColor=[0.0, 0.0, 0.5, 1])
        # pos, ori = pb.getBasePositionAndOrientation(ii)
        # pb.resetBasePositionAndOrientation(self.drill_uid[-1], (1.2, 0, 1.157), (0, 0, 0, 1))
        # pb.resetBasePositionAndOrientation(self.drill_uid[-1], (1.2, 0, 1.148), (0, 0, 0, 1))
        pb.resetBasePositionAndOrientation(self.drill_uid[-1], (1.2, 0, 0.883), (0, 0, 0, 1))


        # import os
        # import pybullet_data
        # planeName = os.path.join(pybullet_data.getDataPath(),"mjcf/ground_plane.xml")
        # self.ground_plane_mjcf = pb.loadMJCF(planeName)
        # for i in self.ground_plane_mjcf:
        #     pb.changeVisualShape(i, -1, rgbaColor=[0, 0, 0, 0])

        robot_state = self._robot.reset()
        self._observation = self.get_env_obs(robot_state)

        return np.array(self._observation)

    def _step(self, action):
        self._robot.apply_action(action)

        self.do_simulation()

        robot_state = self._robot.robot_state()  # also calculates self.joints_at_limit
        self._observation = self.get_env_obs(robot_state)

        reward = self.calc_reward()

        self.time_counter += self.dt
        self.done = self._termination()
        return np.array(self._observation), reward, self.done, {}

    def get_env_obs(self, robot_observation):
        # Extends a vector with more data
        # rgb = pb.getCameraImage(width=self._width,
        #                         height=self._height,
        #                         viewMatrix=self._view_mat,
        #                         projectionMatrix=self._proj_matrix)[2]
        # np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
        #
        # # Only image
        # observation = np_img_arr
        extension = np.array([])
        self._observation = np.concatenate((robot_observation, extension))

        return self._observation

    def _termination(self):
        if self.time_counter >= self.max_time:
            return True
        else:
            return False

    def calc_reward(self):
        reward = 0
        return reward

