import os
import numpy as np
import pybullet as pb
import gym
from gym import spaces
from gym.utils import seeding
from robolearn.envs.pybullet.bullet_env import BulletEnv
from robolearn.envs.reacher.reacher_robot import ReacherMJCRobot
from robolearn.envs.reacher.reacher_target import ReacherTarget


TGT_DEF_COLORS = ['yellow', 'red', 'green', 'white', 'blue']

class ReacherBulletEnv(BulletEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, render=False, obs_with_img=False, obs_mjc_gym=False, ntargets=1, rdn_tgt_pos=True):
        self.np_random = None
        self._obs_with_img = obs_with_img
        self._obs_mjc_gym = obs_mjc_gym

        # Environment/Scene
        self.done = False
        gravity = 9.8
        sim_timestep = 1./100.
        frame_skip = 1
        self._pos_dof = 2  # 2 or 3

        # Robot
        self._robot = ReacherMJCRobot()

        # Target(s)
        self.tgt_pos_is_rdn = rdn_tgt_pos
        self.tgt_pos = [(0, 0.1) for _ in range(ntargets)]
        self.tgt_colors = [TGT_DEF_COLORS[cc] for cc in range(ntargets)]
        self.tgt_cost_weights = [1. for _ in range(ntargets)]
        self.tgts = [ReacherTarget() for _ in range(ntargets)]

        # OBS/ACT
        self.action_bounds = [(-np.pi, np.pi) for _ in range(self._robot.action_dim)]
        self._act_dim = self._robot.action_dim + 0
        self.img_width = 320
        self.img_height = 320

        self._obs_dim = self._robot.observation_dim + self._pos_dof*ntargets
        if self._obs_with_img:
            self._obs_dim += self.img_width*self.img_height*3
        if self._obs_mjc_gym:
            self._obs_dim += self._robot.action_dim + 2*ntargets

        super(ReacherBulletEnv, self).__init__(gravity=gravity, sim_timestep=sim_timestep, frameskip=frame_skip,
                                               render=render)

        self.set_visualizer_data(distance=0.5, yaw=-30, pitch=-90, target_pos=None)

        # Environment settings
        self.max_time = 5  # s
        self.time_counter = 0
        # self.set_render_data(distance=1.490, yaw=-00, pitch=-90, target_pos=None,
        self.set_render_data(distance=0.490, yaw=-00, pitch=-90, target_pos=None,
                             width=self.img_width, height=self.img_height)
        self._observation = np.zeros(self.observation_space.shape)

    def reset_model(self):
        self.done = False
        self.time_counter = 0

        # Ground
        # mjcf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/reacher_ground.xml')
        # pb_bodies = pb.loadMJCF(mjcf_xml)
        ground_urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/reacher_ground.urdf')
        pb_bodies = pb.loadURDF(ground_urdf)
        if issubclass(type(pb_bodies), int):
            pb_bodies = [pb_bodies]
        for ii in pb_bodies:
            pb.changeVisualShape(ii, -1, rgbaColor=[0.0, 0.0, 0.0, 1])

        # Border
        mjcf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/reacher_border.xml')
        pb_bodies = pb.loadMJCF(mjcf_xml)
        if issubclass(type(pb_bodies), int):
            pb_bodies = [pb_bodies]
        for ii in pb_bodies:
            pb.changeVisualShape(ii, -1, rgbaColor=[0.9, 0.4, 0.6, 1])

        # Target 0
        for tt, tgt in enumerate(self.tgts):
            tgt_state = tgt.reset()
            if self.tgt_pos_is_rdn:
                while True:
                    self.tgt_pos[tt] = self.np_random.uniform(low=-.2, high=.2, size=2)
                    if np.linalg.norm(self.tgt_pos[tt]) < 2:
                        break
            tgt.set_color(self.tgt_colors[tt])
            # des_pos = np.zeros(3)
            # des_pos[:2] = self.tgt_pos[tt]
            # tgt.set_robot_pose(des_pos, [0, 0, 0, 1])
            # tgt.apply_action(des_pos[:2])
            tgt.set_state(self.tgt_pos[tt], np.zeros_like(self.tgt_pos[tt]))

        # Camera
        self.set_render_data(width=self.img_width, height=self.img_height)

        # Robot
        robot_state = self._robot.reset()
        self._observation = self.get_env_obs(robot_state)

        return np.array(self._observation)

    def _step(self, action):
        # Move target at any step
        # if not hasattr(self, 'dummy_pos'):
        #     self.dummy_pos = self.tgts[0].parts['target'].current_position()[:2]
        # self.dummy_pos = [0.1, -0.1]
        # self.tgts[0].apply_action(self.dummy_pos)

        self._robot.apply_action(action)

        self.do_simulation()

        robot_state = self._robot.robot_state()  # also calculates self.joints_at_limit
        self._observation = self.get_env_obs(robot_state)

        reward, rw_dist, rw_ctrl = self.calc_reward(action)

        self.time_counter += self.dt
        self.done = self._termination()
        return np.array(self._observation), reward, self.done, dict(reward_dist=rw_dist, reward_ctrl=rw_ctrl)

    def get_env_obs(self, robot_observation):
        fingertip_pos = self._robot.parts['fingertip'].current_position()

        pos_dof = self._pos_dof
        vec = np.zeros(pos_dof*len(self.tgts))
        for tt, tgt in enumerate(self.tgts):
            target_pos = tgt.parts['target'].current_position()
            vec[tt*pos_dof:(tt+1)*pos_dof] = fingertip_pos[:pos_dof] - target_pos[:pos_dof]

        if self._obs_with_img:
            np_img_arr = self.render(mode='rgb_array').flatten()
        else:
            np_img_arr = np.array([])

        if self._obs_mjc_gym:
            theta = robot_observation[:2]
            tgt_state = np.array([tgt.get_state()[:2] for tgt in self.tgts]).flatten()
            robot_observation = np.concatenate((np.cos(theta), np.sin(theta), tgt_state,
                                                robot_observation[2:]))

        self._observation = np.concatenate((robot_observation, vec, np_img_arr))

        return self._observation

    def _termination(self):
        if self.time_counter >= self.max_time:
            return True
        else:
            return False

    def calc_reward(self, a):
        fingertip_pos = self._robot.parts['fingertip'].current_position()

        reward_dist = np.zeros(len(self.tgts))
        for tt, target in enumerate(self.tgts):
            target_pos = self.tgts[tt].get_position()
            vec = fingertip_pos - target_pos
            reward_dist[tt] = -np.linalg.norm(vec) * self.tgt_cost_weights[tt]
        # print('reward -->', target_pos, fingertip_pos, vec)

        reward_ctrl = - np.square(a).sum()
        reward = reward_dist.sum() + reward_ctrl
        return reward, reward_dist, reward_ctrl

    def change_img_size(self, height=None, width=None):
        prev_width = self.img_width
        prev_height = self.img_height
        if width is not None:
            self.img_width = width
        if height is not None:
            self.img_height = height

        # Recalculate state dimension
        if self._obs_with_img:
            self._obs_dim += (self.img_width*self.img_height - prev_width*prev_height)*3
            self.update_obs_space()

    def set_tgt_cost_weights(self, tgt_cost_weights):
        if len(tgt_cost_weights) != len(self.tgt_cost_weights):
            raise ValueError('tgt weights do not match number of targets!!')

        self.tgt_cost_weights = np.array(tgt_cost_weights).copy()

    def set_tgt_pos(self, tgt_pos):
        if len(tgt_pos) != len(self.tgt_pos):
            raise ValueError('tgt positions do not match number of targets!!')

        self.tgt_pos = tgt_pos

