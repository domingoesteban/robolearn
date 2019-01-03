"""
Run PyTorch Multi Soft Actor Critic on Navigation2dGoalCompoEnv.

<<<<<<< HEAD
NOTE: You need PyTorch 0.4
=======
NOTE: You need PyTorch 0.3 or more (to have torch.distributions)
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d
"""
import numpy as np

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.torch.utils.nn import FlattenMlp
from robolearn.envs.simple_envs.multigoal_env import MultiCompositionEnv
from robolearn.torch.policies import TanhGaussianPolicy
from robolearn.torch.sac.multisac import MultiSoftActorCritic

import argparse


def experiment(variant):
    env = NormalizedBoxEnv(MultiCompositionEnv(
        actuation_cost_coeff=1,
        distance_cost_coeff=0.1,
        goal_reward=1,
        init_sigma=0.1,
    ))

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    qfs = list((
        FlattenMlp(
            hidden_sizes=[net_size, net_size],
            input_size=obs_dim + action_dim,
            output_size=1,
        ),
        FlattenMlp(
            hidden_sizes=[net_size, net_size],
            input_size=obs_dim + action_dim,
            output_size=1,
        ),
        FlattenMlp(
            hidden_sizes=[net_size, net_size],
            input_size=obs_dim + action_dim,
            output_size=1,
        ),
        FlattenMlp(
            hidden_sizes=[net_size, net_size],
            input_size=obs_dim + action_dim,
            output_size=1,
        ),
    ))
    vfs = list((
        FlattenMlp(
            hidden_sizes=[net_size, net_size],
            input_size=obs_dim,
            output_size=1,
        ),
        FlattenMlp(
            hidden_sizes=[net_size, net_size],
            input_size=obs_dim,
            output_size=1,
        ),
        FlattenMlp(
            hidden_sizes=[net_size, net_size],
            input_size=obs_dim,
            output_size=1,
        ),
        FlattenMlp(
            hidden_sizes=[net_size, net_size],
            input_size=obs_dim,
            output_size=1,
        ),

    ))
    policies = list((
        TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size],
            obs_dim=obs_dim,
            action_dim=action_dim,
        ),
        TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size],
            obs_dim=obs_dim,
            action_dim=action_dim,
        ),
        TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size],
            obs_dim=obs_dim,
            action_dim=action_dim,
        ),
        TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size],
            obs_dim=obs_dim,
            action_dim=action_dim,
        ),
    ))
    algorithm = MultiSoftActorCritic(
        env=env,
        policies=policies,
        qfs=qfs,
        vfs=vfs,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


env_params = dict(
    algo_params=dict(
        num_epochs=1000,  # n_epochs
        num_steps_per_epoch=1000,  # epoch_length
        num_steps_per_eval=100,  # like eval_n_episodes??
        batch_size=64,  # batch_size
        max_path_length=30,  # max_path_length
        discount=0.99,

        soft_target_tau=0.001,
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
    ),
    net_size=100
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_size', type=int, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    # parser.add_argument('--exp_name', type=str, default=timestamp())
    # parser.add_argument('--mode', type=str, default='local')
    # parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    variant = env_params

    # Net size
    if args.net_size is not None:
        variant['net_size'] = args.net_size

    # Experiment name
    if args.exp_name is None:
        exp_name = 'multigoal-sac'
    else:
        exp_name = args.exp_name

    setup_logger(exp_name, variant=variant)
    experiment(variant)
