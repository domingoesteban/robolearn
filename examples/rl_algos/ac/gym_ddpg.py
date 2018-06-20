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

from robolearn.torch.rl_algos.ddpg import DDPG

from robolearn.torch.models import NNQFunction
from robolearn.torch.policies import TanhMlpPolicy
from robolearn.utils.exploration_strategies import OUStrategy
from robolearn.utils.exploration_strategies import PolicyWrappedWithExplorationStrategy

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
    policy = TanhMlpPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[net_size, net_size],
    )
    es = OUStrategy(
        action_space=env.action_space,
        mu=0,
        theta=0.15,
        max_sigma=0.3,
        min_sigma=0.3,
        decay_period=100000,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    replay_buffer = SimpleReplayBuffer(
        variant['algo_params']['replay_buffer_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    variant['algo_params']['replay_buffer'] = replay_buffer

    # QF Plot
    # variant['algo_params']['epoch_plotter'] = None

    algorithm = DDPG(
        env=env,
        # training_env=env,
        save_environment=False,
        policy=policy,
        exploration_policy=exploration_policy,
        qf=qf,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

    return algorithm


PATH_LENGTH = 1000
PATHS_PER_EPOCH = 1
PATHS_PER_EVAL = 1

expt_params = dict(
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
        # DDPG params
        # TODO: epoch_plotter
        policy_learning_rate=1e-4,
        qf_learning_rate=1e-3,
        use_soft_update=True,
        tau=1e-2,

        discount=0.99,
        reward_scale=1.0,
    ),
    net_size=64
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='mountaincar')
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

    expt_variant = expt_params

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
