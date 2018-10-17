"""
Run PyTorch Reinforce on Pusher2D3DofGoalCompoEnv.

NOTE: You need PyTorch 0.4
"""

import numpy as np

import robolearn.torch.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.utils.data_management import SimpleReplayBuffer
import gym

from robolearn.torch.rl_algos.sac import SAC

from robolearn.torch.models import NNQFunction, NNVFunction
from robolearn.torch.policies import TanhGaussianPolicy

import argparse


def experiment(variant):
    ptu.set_gpu_mode(variant['gpu'])

    env = NormalizedBoxEnv(
        gym.make(variant['env_name'])
    )

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']

    qf = NNQFunction(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[net_size, net_size]
    )
    vf = NNVFunction(
        obs_dim=obs_dim,
        hidden_sizes=[net_size, net_size]
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[net_size, net_size],
    )

    replay_buffer = SimpleReplayBuffer(
        variant['algo_params']['replay_buffer_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    variant['algo_params']['replay_buffer'] = replay_buffer

    # QF Plot
    # variant['algo_params']['epoch_plotter'] = None

    algorithm = SAC(
        env=env,
        # training_env=env,
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


PATH_LENGTH = 1000
PATHS_PER_EPOCH = 1
PATHS_PER_EVAL = 1

SHARED_PARAMS = dict(
    algo_params=dict(
        # Common RLAlgo params
        num_epochs=100,  # n_epochs
        num_steps_per_epoch=PATHS_PER_EPOCH * PATH_LENGTH,
        num_updates_per_train_call=1,  # How to many run algorithm train fcn
        num_steps_per_eval=PATHS_PER_EVAL * PATH_LENGTH,
        # EnvSampler params
        max_path_length=PATH_LENGTH,  # max_path_length
        render=False,
        # ReplayBuffer params
        batch_size=64,  # batch_size
        replay_buffer_size=1e4,
        # SAC params
        policy_lr=1e-3,
        qf_lr=1e-3,
        vf_lr=1e-3,
        soft_target_tau=1e-3,
        policy_mean_reg_weight=1e-3,
        policy_std_reg_weight=1e-3,
        policy_pre_activation_weight=0.,

        discount=0.99,
        reward_scale=1.0,
    ),
    net_size=64
)

ENV_PARAMS = dict(
    mountaincar=dict(
        env_name='MountainCarContinuous-v0',
        algo_params=dict(
            reward_scale=1e-8,
        ),
        net_size=64,
    )
)

AVAILABLE_ENVS = list(ENV_PARAMS.keys())
DEFAULT_ENV = 'mountaincar'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,
                        choices=AVAILABLE_ENVS,
                        default=DEFAULT_ENV)
    parser.add_argument('--net_size', type=int, default=None)
    parser.add_argument('--expt_name', type=str, default=None)
    # parser.add_argument('--expt_name', type=str, default=timestamp())
    # Logging arguments
    parser.add_argument('--snap_mode', type=str, default='gap_and_last')
    parser.add_argument('--snap_gap', type=int, default=10)
    # parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--gpu', action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    specific_variant = ENV_PARAMS[args.env]

    expt_variant = SHARED_PARAMS
    for param in specific_variant:
        if param != 'algo_params':
            expt_variant[param] = specific_variant[param]
        else:
            for algo_param in specific_variant['algo_params']:
                expt_variant['algo_params'][algo_param] = \
                    specific_variant['algo_params'][algo_param]

    # Net size
    if args.net_size is not None:
        expt_variant['net_size'] = args.net_size

    expt_variant['gpu'] = args.gpu

    # Experiment name
    if args.expt_name is None:
        expt_name = 'gym_'+args.env
    else:
        expt_name = args.expt_name

    expt_variant['algo_params']['render'] = args.render

    if args.env == 'mountaincar':
        expt_variant['env_name'] = 'MountainCarContinuous-v0'
    else:
        raise NotImplementedError

    setup_logger(expt_name,
                 variant=expt_variant,
                 snapshot_mode=args.snap_mode,
                 snapshot_gap=args.snap_gap,
                 log_dir=args.log_dir)
    algo = experiment(expt_variant)

    input('Press a key to close the script...')
