"""
Run PyTorch Soft Actor Critic on some Gym Envs.

NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
"""
import gym
import numpy as np

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger

from robolearn.torch.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from robolearn.torch.utils.nn import FlattenMlp

import argparse


def experiment(variant):
    ptu._use_gpu = variant['gpu']
    env = NormalizedBoxEnv(gym.make(variant['env_name']))

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']

    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    algorithm = SoftActorCritic(
        env=env,
        training_env=env,
        save_environment=False,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()

    algorithm.train()

    return algorithm


SHARED_PARAMS = dict(
    # Common RLAlgo params
    num_steps_per_epoch=1000,  # Epoch length
    num_updates_per_env_step=1,  # Like n_train_repeat??
    num_steps_per_eval=1000,  # like eval_n_episodes??
    # EnvSampler params
    max_path_length=1000,
    render=False,
    # ReplayBuffer params
    batch_size=128,
    min_buffer_size=1000,  # Minimum buffer size to start training
    replay_buffer_size=1e6,
    # SoftAC params
    soft_target_tau=0.001,
    policy_lr=3e-4,
    qf_lr=3e-4,
    vf_lr=3e-4,

    # Reward params
    discount=0.99,

)

ENV_PARAMS = {
    'cogimon': dict(
        env_name='CogimonLocomotionBulletEnv-v0',
        algo_params=dict(
            num_epochs=10000,
            max_path_length=1000,
            min_buffer_size=1000,
            reward_scale=10,
        ),
        net_size=500,
    ),
    'cogimon-render': dict(
        env_name='CogimonLocomotionBulletEnvRender-v0',
        algo_params=dict(
            num_epochs=10000,
            max_path_length=1000,
            min_buffer_size=1000,
            reward_scale=10,
        ),
        net_size=500,
    ),
    'manipulator': dict(
        env_name='Reacher2D3DofObstacleEnv-v0',
        algo_params=dict(
            num_epochs=1000,
            max_path_length=500,
            min_buffer_size=500,
            reward_scale=10,
        ),
        net_size=128,
    ),
    'manipulator-render': dict(
        env_name='Pusher2D3DofObstacleBulletEnvRender-v0',
        algo_params=dict(
            num_epochs=1000,
            max_path_length=500,
            min_buffer_size=500,
            reward_scale=10,
        ),
        net_size=128,
    ),
}

AVAILABLE_ENVS = list(ENV_PARAMS.keys())
DEFAULT_ENV = 'cogimon'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        choices=AVAILABLE_ENVS,
                        default=DEFAULT_ENV)
    parser.add_argument('--net_size', type=int, default=None)
    parser.add_argument('--expt_name', type=str, default=None)
    # parser.add_argument('--expt_name', type=str, default=timestamp())
    # Logging arguments
    parser.add_argument('--snap_mode', type=str, default='last')
    parser.add_argument('--snap_gap', type=int, default=100)
    # parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--gpu', action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.env is None:
        expt_variant = ENV_PARAMS[DEFAULT_ENV]
    else:
        if args.render:
            args.env = args.env + '-render'
        expt_variant = ENV_PARAMS[args.env]

    default_algo_params = SHARED_PARAMS
    for param in default_algo_params:
        if param not in expt_variant['algo_params'].keys():
            expt_variant['algo_params'][param] = default_algo_params[param]

    # Net size
    if args.net_size is not None:
        expt_variant['net_size'] = args.net_size

    expt_variant['gpu'] = args.gpu

    # Experiment name
    if args.expt_name is None:
        expt_name = expt_variant['env_name']
    else:
        expt_name = args.expt_name

    expt_variant['algo_params']['render'] = args.render

    setup_logger(expt_name,
                 variant=expt_variant,
                 snapshot_mode=args.snap_mode,
                 snapshot_gap=args.snap_gap,
                 log_dir=args.log_dir)
    algorithm = experiment(expt_variant)

    input('Press a key to close the script...')
