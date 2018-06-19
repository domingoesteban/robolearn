"""
Run PyTorch Soft Q-learning on TwoGoalEnv.

NOTE: You need PyTorch 0.4
"""

import numpy as np

import robolearn.torch.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.utils.data_management import SimpleReplayBuffer
from robolearn_gym_envs.pybullet import Pusher2D3DofObstacleBulletEnv

from robolearn.torch.rl_algos.sql import SQL

from robolearn.torch.models import NNQFunction
from robolearn.torch.policies import SamplingPolicy


import argparse


def experiment(variant):
    ptu.set_gpu_mode(variant['gpu'])

    env = NormalizedBoxEnv(
        Pusher2D3DofObstacleBulletEnv(**variant['env_params'])
    )

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    qf = NNQFunction(obs_dim=obs_dim,
                     action_dim=action_dim,
                     hidden_sizes=(net_size, net_size))
    if ptu.gpu_enabled():
        qf.cuda()

    # _i_policy = TanhGaussianPolicy(
    policy = SamplingPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[net_size, net_size],
    )
    if ptu.gpu_enabled():
        policy.cuda()

    replay_buffer = SimpleReplayBuffer(
        variant['algo_params']['replay_buffer_size'],
        np.prod(env.observation_space.shape),
        np.prod(env.action_space.shape),
    )
    variant['algo_params']['replay_buffer'] = replay_buffer

    # QF Plot
    variant['algo_params']['_epoch_plotter'] = None

    algorithm = SQL(
        env=env,
        training_env=env,
        save_environment=False,
        qf=qf,
        policy=policy,
        # algo_interface='torch',
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train(online=True)

    return algorithm


path_length = 500
paths_per_epoch = 5
paths_per_eval = 1

expt_params = dict(
    env_params=dict(
        goal=(-1, -1),
    ),
    algo_params=dict(
        # Common RLAlgo params
        num_steps_per_epoch=paths_per_epoch*path_length,  # epoch_length
        num_epochs=1000,  # n_epochs
        num_updates_per_env_step=1,  # Like n_train_repeat??
        num_steps_per_eval=paths_per_eval*path_length,  # like eval_n_episodes??
        # EnvSampler params
        max_path_length=path_length,  # max_path_length
        render=False,
        # ReplayBuffer params
        batch_size=64,  # batch_size
        replay_buffer_size=1e4,
        # SoftQLearning params
        # TODO: _epoch_plotter
        policy_lr=3e-4,
        qf_lr=3e-4,
        value_n_particles=16,
        use_hard_updates=False,  # Hard update for target Q-fcn
        hard_update_period=5000,  # td_target_update_interval (steps)
        soft_target_tau=0.001,  # Not used if use_hard_updates=True
        # TODO:kernel_fn
        kernel_n_particles=32,
        kernel_update_ratio=0.5,
        discount=0.99,
        reward_scale=10,
    ),
    net_size=128
)

env_params = dict(
    is_render=False,
    obs_with_img=False,
    rdn_tgt_pose=True,
    sim_timestep=0.001,
    frame_skip=10,
    obs_distances=True,
    tgt_cost_weight=30.0,
    obst_cost_weight=1.0,
    ctrl_cost_weight=1.0e-4,
    safe_radius=0.15,
    inside_cost=1,
    outside_cost=0,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_size', type=int, default=None)
    parser.add_argument('--expt_name', type=str, default=None)
    # parser.add_argument('--expt_name', type=str, default=timestamp())
    # Logging arguments
    parser.add_argument('--snap_mode', type=str, default='gap_and_last')
    parser.add_argument('--snap_gap', type=int, default=50)
    # parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--gpu', action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    expt_variant = expt_params

    # Net size
    if args.net_size is not None:
        expt_variant['net_size'] = args.net_size

    expt_variant['gpu'] = args.gpu

    # Experiment name
    if args.expt_name is None:
        expt_name = 'manipulator'
    else:
        expt_name = args.expt_name

    expt_variant['algo_params']['render'] = args.render

    expt_variant['env_params'] = env_params
    expt_variant['env_params']['is_render'] = args.render

    setup_logger(expt_name,
                 variant=expt_variant,
                 snapshot_mode=args.snap_mode,
                 snapshot_gap=args.snap_gap,
                 log_dir=args.log_dir)
    algorithm = experiment(expt_variant)

    input('Press a key to close the script...')
