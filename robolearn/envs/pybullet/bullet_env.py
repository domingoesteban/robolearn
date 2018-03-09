import numpy as np
import pybullet as pb
import gym
from gym import spaces
from gym.utils import seeding
from robolearn.envs.pybullet.bullet_scenes import SingleRobotScene


class BulletEnv(gym.Env):

    def __init__(self, gravity=9.8, sim_timestep=0.001, frameskip=1,
                 render=False):
        self._p = pb
        self.physicsClientId = -1

        # Environment/Scene
        self._scene = None
        self.gravity = gravity
        self.sim_timestep = sim_timestep
        self.frame_skip = frameskip

        # Rendering RGB
        self._render_data = dict()
        self._render_data['distance'] = 3
        self._render_data['yaw'] = 0  # 0
        self._render_data['pitch'] = -30  # -30
        self._render_data['width'] = 320
        self._render_data['height'] = 320  # 240
        self._render_data['target_pos'] = [0, 0, 0]

        # Rendering Human (Visualizer)
        self._vis_data = dict()
        self._vis_data['distance'] = 3.5
        self._vis_data['yaw'] = 40
        self._vis_data['pitch'] = -40
        self._vis_data['target_pos'] = [0, 0, 0]
        self.viewer = None
        self._is_rendering = render

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        # Action Space
        low = np.array([bound[0] for bound in self.action_bounds])
        high = np.array([bound[1] for bound in self.action_bounds])
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

        # Observation Space
        high = np.inf * np.ones([self._obs_dim])
        low = -high
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        self.seed()

    def __del__(self):
        if hasattr(self, 'physicsClientId'):
            if self.physicsClientId >= 0:
                pb.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        # use the same np_randomizer for robot as for env
        # self._robot.np_random = self.np_random
        return [seed]

    def reset_model(self):
        # It should return the model full observation
        raise NotImplementedError

    def viewer_setup(self):
        raise NotImplementedError

    def set_visualizer_data(self, distance=None, yaw=None, pitch=None,
                            target_pos=None):
        if distance is not None:
            self._vis_data['distance'] = distance
        if yaw is not None:
            self._vis_data['yaw'] = yaw
        if pitch is not None:
            self._vis_data['pitch'] = pitch
        if target_pos is not None:
            self._vis_data['target_pos'] = target_pos

    def set_render_data(self, distance=None, yaw=None, pitch=None,
                        target_pos=None, width=None, height=None):
        if distance is not None:
            self._render_data['distance'] = distance
        if yaw is not None:
            self._render_data['yaw'] = yaw
        if pitch is not None:
            self._render_data['pitch'] = pitch
        if target_pos is not None:
            self._render_data['target_pos'] = target_pos
        if width is not None:
            self._render_data['width'] = width
        if height is not None:
            self._render_data['height'] = height

    def get_render_data(self):
        return pb.getDebugVisualizerCamera()

    def _reset(self, **kwargs):
        if self.physicsClientId < 0:
            self.physicsClientId = pb.connect(pb.SHARED_MEMORY)
            if self.physicsClientId < 0:
                if self._is_rendering:
                    # self.physicsClientId = pb.connect(pb.GUI, options="--opengl2")
                    self.physicsClientId = pb.connect(pb.GUI)
                # pb.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
                else:
                    self.physicsClientId = pb.connect(pb.DIRECT)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        print('BORRAR CONFIGURE DEBUG VISUALIZER')
        pb.configureDebugVisualizer(pb.COV_ENABLE_WIREFRAME, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 0)
        # pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        # pb.configureDebugVisualizer(pb.COV_ENABLE_TINY_RENDERER, 1)

        if self._is_rendering:
            pb.resetDebugVisualizerCamera(cameraDistance=self._vis_data['distance'],
                                          cameraYaw=self._vis_data['yaw'],
                                          cameraPitch=self._vis_data['pitch'],
                                          cameraTargetPosition=self._vis_data['target_pos'])

        if self._scene is None:
            self._scene = SingleRobotScene(self.gravity, self.sim_timestep,
                                           self.frame_skip)

        # Restart Scene
        if not self._scene.multiplayer:
            self._scene.episode_restart()
        # self.robot.scene = self.scene

        obs = self.reset_model(**kwargs)

        # TODO: Here something related with viewer_setup
        # if self.viewer is not None:
        #     self.viewer.autoscale()
        #     self.viewer_setup()

        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

        return obs

    @property
    def dt(self):
        return self.sim_timestep * self.frame_skip

    def _render(self, mode='rgb_array', close=False):
        if mode == 'human':
            if self._is_rendering is False:
                self.set_rendering(True)
            # self._is_rendering = True

        if mode == 'rgb_array':
            view_matrix = pb.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self._render_data['target_pos'],
                distance=self._render_data['distance'],
                yaw=self._render_data['yaw'],
                pitch=self._render_data['pitch'],
                roll=0,
                upAxisIndex=2)

            proj_matrix = pb.computeProjectionMatrixFOV(
                fov=60, aspect=float(self._render_data['width'])/self._render_data['height'],
                nearVal=0.1, farVal=100.0)

            (_, _, px, _, _) = pb.getCameraImage(
                width=self._render_data['width'],
                height=self._render_data['height'],
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                # renderer=pb.ER_TINY_RENDERER
                renderer=pb.ER_BULLET_HARDWARE_OPENGL
            )

            rgb_array = np.array(px)
            rgb_array = rgb_array[:, :, :3]

            return rgb_array

        else:
            return np.array([])

    def set_rendering(self, option):
        if self.physicsClientId >= 0:
            pb.disconnect(self.physicsClientId)
            self.physicsClientId = -1
        self._is_rendering = option

    def _close(self):
        if self.physicsClientId >= 0:
            pb.disconnect(self.physicsClientId)
            self.physicsClientId = -1

    def do_simulation(self):
        self._scene.global_step()

    def HUD(self, state, a, done):
        pass

    def update_obs_space(self):
        high = np.inf * np.ones([self._obs_dim])
        low = -high
        self.observation_space = gym.spaces.Box(low, high)

    def get_action_dim(self):
        return self._act_dim

    def get_state_dim(self):
        return self._state_dim

    def get_obs_dim(self):
        return self._obs_dim
