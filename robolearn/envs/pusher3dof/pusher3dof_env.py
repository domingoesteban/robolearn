import os
import numpy as np
import logging
import pybullet as pb
import gym
from gym import spaces
from gym.utils import seeding
from robolearn.envs.pybullet.bullet_env import BulletEnv
from robolearn.envs.pusher3dof.pusher3dof_robot import Pusher3DofRobot
from robolearn.envs.pusher3dof.pusher_target import PusherTarget
from robolearn.envs.pybullet.pybullet_object import PyBulletObject
from robolearn.utils.transformations_utils import compute_cartesian_error
from robolearn.utils.transformations_utils import euler_from_quat
from robolearn.utils.transformations_utils import create_quat
from robolearn.utils.transformations_utils import normalize_angle
from pybullet import getEulerFromQuaternion, getQuaternionFromEuler


TGT_DEF_COLORS = ['yellow', 'red', 'green', 'white', 'blue']


class Pusher3DofBulletEnv(BulletEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, render=False, obs_with_img=False, obs_mjc_gym=False,
                 ntargets=1, rdn_tgt_pos=True, sim_timestep=0.001,
                 frame_skip=10, tgt_types=None):
        self.np_random = None
        self._obs_with_img = obs_with_img
        self._obs_mjc_gym = obs_mjc_gym

        # Environment/Scene
        self.done = False
        gravity = 9.8
        # sim_timestep = 1./100.
        frame_skip = frame_skip  # Like the control rate (multiplies sim_ts)
        self._pose_dof = 3   # [x, y, theta]

        # Robot
        self._robot = Pusher3DofRobot()
        self._ee_offset = [0.06]

        # Target(s)
        self.tgt_pos_is_rdn = rdn_tgt_pos
        self.tgt_pos = [(0, 0.1, 0) for _ in range(ntargets)]
        self.tgt_colors = [TGT_DEF_COLORS[cc] for cc in range(ntargets)]
        self.tgt_cost_weights = [1. for _ in range(ntargets)]
        if tgt_types is None:
            tgt_types = ['C' for _ in range(ntargets)]
        self.tgts = [PusherTarget(type=tgt_type) for tgt_type in tgt_types]

        # OBS/ACT
        self.action_bounds = [(-np.pi, np.pi) for _ in range(self._robot.action_dim)]
        self._act_dim = self._robot.action_dim + 0
        self.img_width = 320
        self.img_height = 320

        self._obs_dim = self._robot.observation_dim + self._pose_dof + \
                        self._pose_dof*ntargets
        if self._obs_with_img:
            self._obs_dim += self.img_width*self.img_height*3
        if self._obs_mjc_gym:
            self._obs_dim += self._robot.action_dim + 2*ntargets

        super(Pusher3DofBulletEnv, self).__init__(gravity=gravity,
                                                  sim_timestep=sim_timestep,
                                                  frameskip=frame_skip,
                                                  render=render)

        self.set_visualizer_data(distance=1.1, yaw=-30, pitch=-90,
                                 target_pos=None)

        # State is the same than Obs
        self._state_dim = self._obs_dim
        self.get_state_info = self.get_obs_info
        self.get_state = self.get_observation


        # Environment settings
        self.max_time = 5  # s
        self.time_counter = 0
        self.set_render_data(distance=1.8, yaw=-00, pitch=-90,
                             target_pos=None, width=self.img_width,
                             height=self.img_height)
        self._observation = np.zeros(self.observation_space.shape)

        # Reset environment so we can get info from pybullet
        self.set_rendering(False)
        self.reset()
        self.obs_types = self.get_obs_types()
        self.set_rendering(render)

        # Initial conditions
        self.init_cond = None

        # LOGGER
        self.logger = logging.getLogger('robolearn_env')
        self.logger.setLevel(logging.WARNING)

    def reset_model(self, condition=None):
        self.done = False
        self.time_counter = 0

        # Ground
        # mjcf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/reacher_ground.xml')
        # pb_bodies = pb.loadMJCF(mjcf_xml)
        ground_urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'models/pusher_ground.urdf')
        pb_bodies = pb.loadURDF(ground_urdf)
        if issubclass(type(pb_bodies), int):
            pb_bodies = [pb_bodies]
        for ii in pb_bodies:
            pb.changeVisualShape(ii, -1, rgbaColor=[0.0, 0.0, 0.0, 1])


        # target_urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                            'models/pusher_target.urdf')
        # target1 = PyBulletObject('urdf', target_urdf, 'target',
        #                          init_pos=(0., -0.5, 0.05))
        # target1.reset()
        # target1.set_pose((0., 0.5, 0.05))


        # Border
        mjcf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'models/pusher_border.xml')
        pb_bodies = pb.loadMJCF(mjcf_xml)
        if issubclass(type(pb_bodies), int):
            pb_bodies = [pb_bodies]
        for ii in pb_bodies:
            pb.changeVisualShape(ii, -1, rgbaColor=[0.9, 0.4, 0.6, 1])

        # Targets
        for tt, tgt in enumerate(self.tgts):
            tgt_pose = tgt.reset()
            if self.tgt_pos_is_rdn:
                while True:
                    self.tgt_pos[tt][:2] = self.np_random.uniform(low=-.2,
                                                                  high=.2,
                                                                  size=2)
                    if np.linalg.norm(self.tgt_pos[tt]) < 2:
                        break
                self.tgt_pos[tt][2] = self.np_random.uniform(low=-np.pi,
                                                             high=np.pi,
                                                             size=1)
            tgt.set_color(self.tgt_colors[tt])

            des_pos = np.zeros(3)
            des_pos[:2] = self.tgt_pos[tt][:2]
            des_pos[2] = 0.055
            tgt.set_pose(des_pos, create_quat(rot_yaw=self.tgt_pos[tt][2]))
            # tgt.apply_action(des_pos[:2])
            # tgt.set_state(self.tgt_pos[tt], np.zeros_like(self.tgt_pos[tt]))

        # Camera
        self.set_render_data(width=self.img_width, height=self.img_height)

        # Robot
        robot_state = self._robot.reset()
        self._observation = self.get_env_obs(robot_state)

        return np.array(self._observation)

    def _step(self, action):
        self._robot.apply_action(action)

        self.do_simulation()

        robot_state = self._robot.robot_state()  # also calculates self.joints_at_limit
        self._observation = self.get_env_obs(robot_state)

        reward, rw_dist, rw_ctrl = self.calc_reward(action)

        self.time_counter += self.dt
        self.done = self._termination()
        return (np.array(self._observation), reward, self.done,
               dict(reward_dist=rw_dist, reward_ctrl=rw_ctrl))

    def get_observation(self):
        return np.array(self._observation)

    def get_env_obs(self, robot_observation):
        gripper_pos = self._robot.parts['gripper_center'].get_pose()[:3]

        pose_dof = self._pose_dof
        vec = np.zeros(pose_dof + pose_dof*len(self.tgts))
        gripper_pose = self._robot.parts['gripper_center'].get_pose()
        xy = gripper_pose[:2]
        ori = getEulerFromQuaternion(gripper_pose[3:])[2]
        # ori = normalize_angle(ori, range='pi')
        vec[:pose_dof] = [xy[0], xy[1], ori]
        last_idx = pose_dof
        for tgt in self.tgts:
            # target_pos = tgt.parts['target'].current_position()
            # target_pos = tgt.get_pose()[:3]
            # vec[tt*pose_dof:(tt+1)*pose_dof] = \
            #     gripper_pos[:pose_dof] - target_pos[:pose_dof]

            tgt_pose = tgt.get_pose()
            xy = tgt_pose[:2]
            # ori = euler_from_quat(tgt_pose[3:], order='xyzw')[2]
            ori = getEulerFromQuaternion(tgt_pose[3:])[2]
            # ori = normalize_angle(ori, range='pi')

            # cartesian_error = \
            #     compute_cartesian_error(tgt_pose, gripper_pose, first='pos')
            # vec[tt*pose_dof:(tt+1)*pose_dof] = \
            #     cartesian_error[err_idx]
            vec[last_idx:last_idx+pose_dof] = [xy[0], xy[1], ori]
            last_idx += pose_dof

        if self._obs_with_img:
            np_img_arr = self.render(mode='rgb_array').flatten()
        else:
            np_img_arr = np.array([])

        if self._obs_mjc_gym:
            theta = robot_observation[:2]
            # tgt_state = np.array([tgt.get_state()[:2] for tgt ineuler_from_quat self.tgts]).flatten()
            tgt_state = np.array([tgt.get_pose()[:2] for tgt in self.tgts]).flatten()
            robot_observation = np.concatenate((np.cos(theta), np.sin(theta), tgt_state,
                                                robot_observation[2:]))

        self._observation = np.concatenate((robot_observation, vec, np_img_arr))

        return self._observation

    def get_obs_types(self):
        obs_types = list(self._robot.get_state_info())

        # End-effector
        new_idx = obs_types[-1]['idx'][-1] + 1
        obs_types.append({'name': 'ee',
                          'idx': list(range(new_idx,
                                            new_idx+self._pose_dof))})

        for tt in range(len(self.tgts)):
            new_idx = obs_types[-1]['idx'][-1] + 1
            obs_types.append({'name': 'tgt'+str(tt),
                              'idx': list(range(new_idx,
                                                new_idx+self._pose_dof))})

        return obs_types

    def _termination(self):
        if self.time_counter >= self.max_time:
            return True
        else:
            return False

    def calc_reward(self, a):
        gripper_pos = self._robot.parts['gripper_center'].current_position()

        reward_dist = np.zeros(len(self.tgts))
        for tt, target in enumerate(self.tgts):
            target_pos = self.tgts[tt].get_pose()[:3]
            vec = gripper_pos - target_pos
            reward_dist[tt] = -np.linalg.norm(vec) * self.tgt_cost_weights[tt]
        # print('reward -->', target_pos, gripper_pos, vec)

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

    def get_obs_info(self, name=None):
        """
        Return Observation info dictionary.
        :param name: Name of the observation. If not specified, returns for all the observations.
        :return: obs_info dictionary with keys: names, dimensions and idx.
        """
        if name is None:
            obs_info = {'names': [obs['name'] for obs in self.obs_types],
                        'dimensions': [len(obs['idx']) for obs in self.obs_types],
                        'idx': [obs['idx'] for obs in self.obs_types]}
        else:
            obs_idx = self.get_obs_idx(name=name)
            obs_info = {'names': self.obs_types[obs_idx]['name'],
                        'dimensions': len(self.obs_types[obs_idx]['idx']),
                        'idx': self.obs_types[obs_idx]['idx']}
        return obs_info

    def get_obs_idx(self, name=None, obs_type=None):
        """
        Return the index of the observation that match the specified name or obs_type
        :param name:
        :type name: str
        :param obs_type:
        :type obs_type: str
        :return: Index of the observation
        :rtype: int
        """
        if name is not None:
            for ii, obs in enumerate(self.obs_types):
                if obs['name'] == name:
                    return ii
            raise ValueError("There is not observation with name %s" % name)

        if obs_type is not None:
            for ii, obs in enumerate(self.obs_types):
                if obs['type'] == obs_type:
                    return ii
            raise ValueError("There is not observation with type %s" % obs_type)

        if name is None and obs_type is None:
            raise AttributeError("No name or obs_type specified")

    def get_env_info(self):
        """
        Return Observation and State info dictionary.
        :return: Dictionary with obs_info and state_info dictionaries. Each one
                 with keys: names, dimensions and idx.
        """
        env_info = {'obs': self.get_obs_info(),
                    'state': self.get_state_info()}
        return env_info

    def add_init_cond(self, cond):
        if len(cond) != self._obs_dim:
            raise ValueError('Wrong initial condition size. (%d != %d)'
                             % (len(cond), self._obs_dim))

        if self.init_cond is None:
            self.init_cond = list()

        self.init_cond.append(cond)

    def get_conditions(self, cond=None):
        if cond is None:
            return list(self.init_cond)
        else:
            return self.init_cond[cond]

    def reset(self, condition=None):
        if condition is not None:
            # Robot State
            init_state = self.init_cond[condition][:self._robot.observation_dim]
            self._robot.set_initial_state(init_state)
            # Targets
            no_target_last_idx = self._robot.observation_dim + self._pose_dof
            tgt_pos = self.init_cond[condition][no_target_last_idx:]
            self.set_tgt_pos([tgt_pos[:self._pose_dof],
                              tgt_pos[-self._pose_dof:]])

        super(Pusher3DofBulletEnv, self).reset()

        # Replace init_cond with current state
        if condition is not None:
            self.init_cond[condition] = np.copy(self._observation)

    def get_total_joints(self):
        return self._robot.total_joints

    def stop(self):
        pass
