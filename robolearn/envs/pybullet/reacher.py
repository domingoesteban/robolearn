import numpy as np
from gym import utils
# from gym.envs.mujoco import mujoco_env
import pybullet as p

from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
# from pybullet_envs.scene_abstract import SingleRobotEmptyScene

from pybullet_envs.env_bases import MJCFBaseBulletEnv


class ReacherBulletEnv(MJCFBaseBulletEnv):
    def __init__(self, robot, render=False):
        print('ReacherBase::__init__')
        MJCFBaseBulletEnv.__init__(self, robot, render)

        self.stadium_scene = None

        '''
        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        '''

    def create_single_player_scene(self):
        self.stadium_scene = SinglePlayerStadiumScene(gravity=9.8, timestep=0.0165 / 4, frame_skip=4)
        return self.stadium_scene

    def _reset(self):
        r = MJCFBaseBulletEnv._reset(self)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
            self.stadium_scene.ground_plane_mjcf)
        self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in
                               self.foot_ground_object_names])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return r

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(init_x, init_y,
                      init_z)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)


    '''
    def _step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

        return None

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
    '''