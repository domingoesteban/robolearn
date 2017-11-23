import numpy as np
import pybullet as pb
import gym
from gym import spaces
from gym.utils import seeding
from robolearn.envs.r2d2.r2d2_robot import R2D2Robot
from robolearn.envs.pybullet.bullet_scenes import SingleRobotScene


class R2D2BulletEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, render=False):
        self._p = pb
        self._is_rendering = render

        self.physicsClientId = -1

        # Rendering
        self._cam_dist = 0.6  # 3
        self._cam_yaw = 0  # 0
        self._cam_pitch = -90  # -30
        self._render_width = 320
        self._render_height = 320  # 240

        self._seed()
        self.np_random = None

        # Environment/Scene
        self.done = False
        self._scene = None
        self.gravity = 9.8
        self.timestep = 1./100.
        self.frame_skip = 1
        self.max_time = 2/self.timestep
        self.dummy_counter = 0

        # Robot
        init_pos = [0, 0, 0.475]
        self._robot = R2D2Robot(init_pos=init_pos)

        # Action Space
        high = np.ones([self._robot.action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        # self.action_space = spaces.Box(self._robot.action_space.low, self._robot.action_space.high)

        # Observation Space
        # self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 4))
        high = np.inf * np.ones([self._robot.obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)
        # self.observation_space = spaces.Box(self._robot.observation_space.low, self._robot.observation_space.high)

        self._observation = np.zeros(self.observation_space.shape)

    def __del__(self):
        if self.physicsClientId >= 0:
            pb.disconnect()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        # self._robot.np_random = self.np_random # use the same np_randomizer for robot as for env
        return [seed]

    def _reset(self):
        if self.physicsClientId < 0:
            self.physicsClientId = pb.connect(pb.SHARED_MEMORY)
            if self.physicsClientId < 0:
                if self._is_rendering:
                    self.physicsClientId = pb.connect(pb.GUI)
                # pb.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
                else:
                    self.physicsClientId = pb.connect(pb.DIRECT)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)

        if self._is_rendering:
            pb.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=-30, cameraPitch=-90,
                                          cameraTargetPosition=[0, 0, 0])

        self.done = False
        self.dummy_counter = 0

        if self._scene is None:
            self._scene = SingleRobotScene(self.gravity, self.timestep, self.frame_skip)
        # Restart Scene
        if not self._scene.multiplayer:
            self._scene.episode_restart()
        # self.robot.scene = self.scene

        # xml_env = '/home/domingo/robotlearning-workspace/learning-modules/robolearn/robolearn/envs/pybullet/mjcf/reacher_world.xml'
        # pb.loadMJCF(xml_env)

        temp_urdf = '/home/domingo/robotlearning-workspace/learning-modules/robolearn/robolearn/envs/pybullet/plane/plane.urdf'
        pb.loadURDF(temp_urdf)

        # import os
        # import pybullet_data
        # planeName = os.path.join(pybullet_data.getDataPath(),"mjcf/ground_plane.xml")
        # self.ground_plane_mjcf = pb.loadMJCF(planeName)
        # for i in self.ground_plane_mjcf:
        #     pb.changeVisualShape(i, -1, rgbaColor=[0, 0, 0, 0])

        robot_state = self._robot.reset()
        self._observation = self.get_env_obs(robot_state)

        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        return np.array(self._observation)

    def _step(self, action):
        self._robot.appy_action(action)
        self._scene.global_step()

        robot_state = self._robot.robot_state()  # also calculates self.joints_at_limit
        self._observation = self.get_env_obs(robot_state)

        reward = self._reward()

        self.dummy_counter += 1
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
        if self.dummy_counter >= self.max_time:
            return True
        else:
            return False

    def _reward(self):
        reward = 0
        return reward

    def _render(self, mode='rgb_array', close=False):
        if mode == 'human':
            self._is_rendering = True
        if mode != 'rgb_array':
            return np.array([])

        base_pos = [0, 0, 0]
        if hasattr(self, '_robot'):
            if hasattr(self._robot, 'body_xyz'):
                base_pos = self._robot.body_xyz

        view_matrix = pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)

        proj_matrix = pb.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width)/self._render_height,
            nearVal=0.1, farVal=100.0)

        (_, _, px, _, _) = pb.getCameraImage(
            width=self._render_width, height=self._render_height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL
        )

        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _close(self):
        if self.physicsClientId >= 0:
            pb.disconnect(self.physicsClientId)
            self.physicsClientId = -1

