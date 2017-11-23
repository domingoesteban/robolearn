import os
import numpy as np
import pybullet as pb
import gym
from gym import spaces
from gym.utils import seeding
from robolearn.envs.pybullet.bullet_env import BulletEnv
from robolearn.envs.reacher.reacher_robot import ReacherMJCRobot
from robolearn.envs.reacher.reacher_target import ReacherTarget


class ReacherBulletEnv(BulletEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, render=False, obs_with_img=False):
        self.np_random = None
        self._obs_with_img = obs_with_img

        # Environment/Scene
        self.done = False
        gravity = 9.8
        sim_timestep = 1./100.
        frame_skip = 1

        # Robot
        self._robot = ReacherMJCRobot()

        # Target
        self.tgt_pos = [(0, 0.1)]
        self.tgts = [ReacherTarget() for _ in self.tgt_pos]

        self.action_bounds = [(-np.pi, np.pi) for _ in range(self._robot.action_dim)]
        self._act_dim = self._robot.action_dim + 0
        self._obs_dim = self._robot.observation_dim + 0

        super(ReacherBulletEnv, self).__init__(gravity=gravity, sim_timestep=sim_timestep, frameskip=frame_skip,
                                               render=render)

        self.set_visualizer_data(distance=0.5, yaw=-30, pitch=-90, target_pos=None)

        # Environment settings
        self.max_time = 5  # s
        self.time_counter = 0
        self.img_width = 320
        self.img_height = 320
        if self._obs_with_img:
            self.set_render_data(distance=0.490, yaw=-00, pitch=-90, target_pos=None,
                                 width=self.img_width, height=self.img_height)
        self._observation = np.zeros(self.observation_space.shape)

    def reset_model(self):
        self.done = False
        self.time_counter = 0

        # Ground
        mjcf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mjcf/reacher_ground.xml')
        pb_bodies = pb.loadMJCF(mjcf_xml)
        if issubclass(type(pb_bodies), int):
            pb_bodies = [pb_bodies]
        for ii in pb_bodies:
            pb.changeVisualShape(ii, -1, rgbaColor=[0.0, 0.0, 0.0, 1])

        # Border
        mjcf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mjcf/reacher_border.xml')
        pb_bodies = pb.loadMJCF(mjcf_xml)
        if issubclass(type(pb_bodies), int):
            pb_bodies = [pb_bodies]
        for ii in pb_bodies:
            pb.changeVisualShape(ii, -1, rgbaColor=[0.9, 0.4, 0.6, 1])

        # Target 0
        tgt_state = self.tgts[0].reset()
        des_pos = np.zeros(3)
        des_pos[:2] = self.tgt_pos[0]
        self.tgts[0].parts['target'].reset_pose(des_pos, [0, 0, 0, 1])

        color_list = [(0.9, 0.2, 0.2, 1) for _ in range(self.tgts[0].get_total_bodies())]
        self.tgts[0].set_body_colors(color_list)

        # Camera
        if self._obs_with_img:
            self.set_render_data(width=self.img_width, height=self.img_height)

        # Robot
        robot_state = self._robot.reset()
        self._observation = self.get_env_obs(robot_state)

        return np.array(self._observation)

    def _step(self, action):

        if not hasattr(self, 'dummy_pos'):
            self.dummy_pos = self.tgts[0].parts['target'].current_position()[:2]
        self.dummy_pos = [0.1, -0.1]
        self.tgts[0].apply_action(self.dummy_pos)

        self._robot.apply_action(action)

        self.do_simulation()

        robot_state = self._robot.robot_state()  # also calculates self.joints_at_limit
        self._observation = self.get_env_obs(robot_state)

        reward = self.calc_reward(action)

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
        fingertip_pos = self._robot.parts['fingertip'].current_position()
        target_pos = self.tgts[0].parts['target'].current_position()
        vec = fingertip_pos - target_pos

        if self._obs_with_img:
            np_img_arr = self.render(mode='rgb_array').flatten()
            self.temporal_compa = np_img_arr.copy()
        else:
            np_img_arr = np.array([])

        self._observation = np.concatenate((robot_observation, vec, np_img_arr))

        return self._observation

    def _termination(self):
        if self.time_counter >= self.max_time:
            return True
        else:
            return False

    def calc_reward(self, a):
        fingertip_pos = self._robot.parts['fingertip'].current_position()
        target_pos = self.tgts[0].parts['target'].current_position()
        vec = fingertip_pos - target_pos
        reward_dist = -np.linalg.norm(vec)
        print('reward -->', target_pos, fingertip_pos, vec)

        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        return reward

    def change_img_size(self, height=None, width=None):
        if width is not None:
            self.img_width = width
        if height is not None:
            self.img_height = height

