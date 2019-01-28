#!/usr/bin/env python

import numpy as np
import tensorflow as tf

from spinup import sac
from spinup.algos.sac.core import mlp_actor_critic

from robolearn_gym_envs.pybullet import CentauroTrayEnv
from robolearn.envs.normalized_box_env import NormalizedBoxEnv

from spinup.utils.run_utils import setup_logger_kwargs

EPOCHS = 1000

Tend = 10.0  # Seconds

SIM_TIMESTEP = 0.01
FRAME_SKIP = 1
DT = SIM_TIMESTEP * FRAME_SKIP

PATH_LENGTH = int(np.ceil(Tend / DT))
PATHS_PER_EPOCH = 1
PATHS_PER_EVAL = 2
BATCH_SIZE = 128

SEED = 1010
# NP_THREADS = 6

SUBTASK = None

EXP_NAME = 'prueba_centauro1_sac'

env_params = dict(
    is_render=False,
    # obs_distances=False,
    obs_distances=True,
    obs_with_img=False,
    # obs_with_ori=True,
    active_joints='RA',
    control_mode='joint_tasktorque',
    # _control_mode='torque',
    balance_cost_weight=1.0,
    fall_cost_weight=1.0,
    tgt_cost_weight=3.0,
    # tgt_cost_weight=50.0,
    balance_done_cost=0.,  # 2.0*PATH_LENGTH,  # TODO: dont forget same balance weight
    tgt_done_reward=0.,  # 20.0,
    ctrl_cost_weight=1.0e-1,
    use_log_distances=True,
    log_alpha_pos=1e-4,
    log_alpha_ori=1e-4,
    goal_tolerance=0.05,
    min_obj_height=0.60,
    max_obj_height=1.20,
    max_obj_distance=0.20,
    max_time=None,
    sim_timestep=SIM_TIMESTEP,
    frame_skip=FRAME_SKIP,
    subtask=SUBTASK,
    random_init=True,
    seed=SEED,
)


def main():
    # Environment Fcn
    env_fn = lambda: \
        NormalizedBoxEnv(
            CentauroTrayEnv(**env_params),
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


if __name__ == '__main__':
    main()
