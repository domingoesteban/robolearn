#!/usr/bin/env python

import numpy as np
import tensorflow as tf

from spinup import sac
from spinup.algos.sac.core import mlp_actor_critic

from robolearn_gym_envs.pybullet import CentauroTrayEnv
from robolearn_gym_envs.pybullet import Reacher2D3DofGoalCompoEnv
from robolearn.envs.normalized_box_env import NormalizedBoxEnv

from spinup.utils.run_utils import setup_logger_kwargs

EPOCHS = 500

Tend = 10.0  # Seconds

SIM_TIMESTEP = 0.001
FRAME_SKIP = 10
DT = SIM_TIMESTEP * FRAME_SKIP

PATH_LENGTH = int(np.ceil(Tend / DT))
PATHS_PER_EPOCH = 1
PATHS_PER_EVAL = 2
PATHS_PER_HARD_UPDATE = 12
BATCH_SIZE = 256

SEED = 610
SUBTASK = None

EXP_NAME = 'prueba_reacher1_sac'

env_params = dict(
    is_render=False,
    # obs_distances=False,
    obs_distances=True,
    obs_with_img=False,
    # obs_with_ori=True,
    obs_with_ori=False,
    obs_with_goal=True,
    # obs_with_goal=False,
    # goal_pose=(0.65, 0.65),
    goal_pose=(0.65, 0.35),
    # rdn_goal_pos=True,
    rdn_goal_pos=False,
    robot_config=None,
    rdn_robot_config=True,
    goal_cost_weight=4.0e0,
    ctrl_cost_weight=5.0e-1,
    goal_tolerance=0.01,
    use_log_distances=True,
    log_alpha=1e-6,
    # max_time=PATH_LENGTH*DT,
    max_time=None,
    sim_timestep=SIM_TIMESTEP,
    frame_skip=FRAME_SKIP,
    half_env=True,
    subtask=SUBTASK,
    seed=SEED,
)


def main():
    # Environment Fcn
    env_fn = lambda: \
        NormalizedBoxEnv(
            Reacher2D3DofGoalCompoEnv(**env_params),
            # normalize_obs=True,
            normalize_obs=False,
            online_normalization=False,
            obs_mean=None,
            obs_var=None,
            obs_alpha=0.001,
        )

    # Logger kwargs
    logger_kwargs = setup_logger_kwargs(EXP_NAME, SEED)

    with tf.Graph().as_default():
        sac(
            env_fn,
            actor_critic=mlp_actor_critic,
            ac_kwargs=dict(hidden_sizes=(128, 128, 128)),
            seed=SEED,
            steps_per_epoch=PATHS_PER_EPOCH * PATH_LENGTH,
            epochs=EPOCHS,
            replay_size=int(1e6),
            gamma=0.99,
            polyak=0.995,  # Polyak avg target pol (0-1)
            lr=1e-3,
            alpha=0.2,  # entropy regularization coefficient (inv rew scale)
            batch_size=BATCH_SIZE,
            start_steps=10000,
            max_ep_len=PATH_LENGTH,  # Max length for trajectory
            logger_kwargs=logger_kwargs,
            save_freq=1
        )
        # ppo(
        #     env_fn,
        #     actor_critic=mlp_actor_critic,
        #     ac_kwargs=dict(hidden_sizes=(128, 128, 128)),
        #     seed=SEED,
        #     steps_per_epoch=PATHS_PER_EPOCH * PATH_LENGTH,
        #     epochs=10000,
        #     gamma=0.99,  # Discount factor (0-1)
        #     clip_ratio=0.2,  # clip pol objective (0.1-0.3)
        #     pi_lr=3e-4,
        #     vf_lr=1e-3,
        #     train_pi_iters=80,  # Max grad steps in pol loss per epoch
        #     train_v_iters=80,  # Max grad steps in val loss per epoch
        #     lam=0.97,  # Lambda for GAE-Lambda (0-1)
        #     max_ep_len=PATH_LENGTH,  # Max length for trajectory
        #     target_kl=0.01,  # KLdiv between new and old policies
        #     logger_kwargs=logger_kwargs,
        #     save_freq=10,
        # )


if __name__ == '__main__':
    main()
