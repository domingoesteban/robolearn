"""
Run PyTorch Soft Q-learning on TwoGoalEnv.

NOTE: You need PyTorch 0.4
"""

import numpy as np

import robolearn.torch.pytorch_util as ptu
from robolearn.envs.wrappers import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.utils.data_management.multi_env_replay_buffer import MultiEnvReplayBuffer
from robolearn_gym_envs.pybullet import Reacher2D3DofObstacleEnv

from robolearn.torch.sql.policies import StochasticPolicy
from robolearn.torch.sql.iu_sql import IUSQL
from robolearn.torch.sql.value_functions import NNQFunction

import argparse


def experiment(variant):
    exploration_pol_id = 1

    variant['algo_params']['exploration_pol_id'] = exploration_pol_id

    n_demons = 2

    ptu._use_gpu = variant['gpu']

    env = NormalizedBoxEnv(Reacher2D3DofObstacleEnv(
        is_render=False, obs_with_img=False,
        rdn_tgt_pose=True, sim_timestep=0.001,
        frame_skip=10, obs_distances=True,
        tgt_cost_weight=1.0,
        obst_cost_weight=3.0,
        ctrl_cost_weight=1.0e-2, safe_radius=0.15,
        inside_cost=1, outside_cost=0
    ))

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    # qfs = [FlattenMlp(
    #        hidden_sizes=[net_size, net_size],
    #        input_size=obs_dim + action_dim,
    #        output_size=1) for _ in range(n_demons)]
    qfs = [NNQFunction(obs_dim=obs_dim,
                       action_dim=action_dim,
                       hidden_sizes=(net_size, net_size))
           for _ in range(n_demons)]
    if ptu.gpu_enabled():
        for qf in qfs:
            qf.cuda()

    policies = [StochasticPolicy(
                hidden_sizes=[net_size, net_size],
                obs_dim=obs_dim,
                action_dim=action_dim) for _ in range(n_demons)]
    if ptu.gpu_enabled():
        for policy in policies:
            policy.cuda()

    replay_buffer = MultiEnvReplayBuffer(
        variant['algo_params']['replay_buffer_size'],
        env,
        reward_vector_size=n_demons,
    )
    variant['algo_params']['replay_buffer'] = replay_buffer

    # QF Plot
    variant['algo_params']['_epoch_plotter'] = None

    algorithm = IUSQL(
        env=env,
        u_qfs=qfs,
        u_policies=policies,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

    return algorithm


env_params = dict(
    algo_params=dict(
        # Common RLAlgo params
        num_steps_per_epoch=100,  # epoch_length
        num_epochs=1000,  # n_epochs
        num_updates_per_env_step=1,  # Like n_train_repeat??
        num_steps_per_eval=100,  # like eval_n_episodes??
        # EnvSampler params
        max_path_length=30,  # max_path_length
        render=False,
        # ReplayBuffer params
        batch_size=64,  # batch_size
        replay_buffer_size=1e6,
        # SoftQLearning params
        # TODO: _epoch_plotter
        policy_lr=3e-4,
        qf_lr=3e-4,
        value_n_particles=16,
        use_hard_updates=True,  # Hard update for target Q-fcn
        hard_update_period=1000,  # td_target_update_interval (steps)
        soft_target_tau=0.001,  # Not used if use_hard_updates=True
        # TODO:kernel_fn
        kernel_n_particles=32,
        kernel_update_ratio=0.5,
        discount=0.99,
        reward_scale=0.1,
    ),
    net_size=128
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_size', type=int, default=None)
    parser.add_argument('--expt_name', type=str, default=None)
    # parser.add_argument('--expt_name', type=str, default=timestamp())
    # Logging arguments
    parser.add_argument('--snap_mode', type=str, default='last')
    parser.add_argument('--snap_gap', type=int, default=1)
    # parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--gpu', action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    expt_variant = env_params

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

    setup_logger(expt_name,
                 variant=expt_variant,
                 snapshot_mode=args.snap_mode,
                 snapshot_gap=args.snap_gap,
                 log_dir=args.log_dir)
    algorithm = experiment(expt_variant)

    input('Press a key to close the script...')
