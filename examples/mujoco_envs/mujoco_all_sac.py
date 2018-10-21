"""
Run PyTorch Soft Actor Critic on some Gym Envs.

NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
"""
import gym
import numpy as np

import robolearn.torch.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.torch.policies import TanhGaussianPolicy
from robolearn.torch.rl_algos.sac import SAC
from robolearn.torch.nn import FlattenMlp

import argparse


def experiment(variant):
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
    algorithm = SAC(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


ENV_PARAMS = {
    'half-cheetah': dict(
        env_name='HalfCheetah-v2',
        algo_params=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=999,
            discount=0.99,

            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        net_size=300
    ),
    'reacher': dict(
        env_name='Reacher-v2',
        algo_params=dict(
            num_epochs=1000,
            num_steps_per_epoch=2000,  # The time in reacher is 50 for done=True
            num_steps_per_eval=50,  # Not sure how it works | For now, it adds to max_path_length
            batch_size=128,
            max_path_length=50,  # The time in reacher is 50 for done=True
            discount=0.99,

            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        net_size=100
    ),
    'ant': dict(
        env_name='Ant-v2',
        algo_params=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=999,
            discount=0.99,

            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        net_size=500
    ),
}

AVAILABLE_ENVS = list(ENV_PARAMS.keys())
DEFAULT_ENV = 'half-cheetah'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        choices=AVAILABLE_ENVS,
                        default=DEFAULT_ENV)
    parser.add_argument('--net_size', type=int, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    # parser.add_argument('--exp_name', type=str, default=timestamp())
    # parser.add_argument('--mode', type=str, default='local')
    # parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.env is None:
        variant = ENV_PARAMS[DEFAULT_ENV]
    else:
        variant = ENV_PARAMS[args.env]

    # Net size
    if args.net_size is not None:
        variant['net_size'] = args.net_size

    # Experiment name
    if args.exp_name is None:
        exp_name = variant['env_name']
    else:
        exp_name = args.exp_name

    setup_logger(exp_name, variant=variant)
    experiment(variant)
