"""
Run PyTorch Soft Q-learning on PusherEnv.

NOTE: You need PyTorch 0.4
"""

import numpy as np

import robolearn.torch.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.utils.data_management.multigoal_replay_buffer import MultiGoalReplayBuffer

from robolearn.envs.simple_envs.goal_composition.goal_composition_env import GoalCompositionEnv

from robolearn.torch.sql.iu_sql import IUSQL
from robolearn.envs.simple_envs.goal_composition.goal_composition_q_plot import QFPolicyPlotter

from robolearn.torch.sql.value_functions import NNQFunction
from robolearn.torch.sql.value_functions import SumNNQFunction
# from robolearn.torch.policies import TanhGaussianPolicy
from robolearn.torch.sql.policies import StochasticPolicy

import argparse


def experiment(variant):
    render_q = variant['render_q']
    save_q_path = '/home/desteban/logs/goalcompo_q_plots'

    ptu.set_gpu_mode(variant['gpu'])

    env = NormalizedBoxEnv(
        GoalCompositionEnv(**variant['env_params'])
    )

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    n_unintentional = 2

    net_size = variant['net_size']
    u_qfs = [NNQFunction(obs_dim=obs_dim,
                         action_dim=action_dim,
                         hidden_sizes=(net_size, net_size))
             for _ in range(n_unintentional)]
    # i_qf = AvgNNQFunction(obs_dim=obs_dim,
    i_qf = SumNNQFunction(obs_dim=obs_dim,
                          action_dim=action_dim,
                          q_functions=u_qfs)

    # _i_policy = TanhGaussianPolicy(
    u_policies = [StochasticPolicy(
                hidden_sizes=[net_size, net_size],
                obs_dim=obs_dim,
                action_dim=action_dim,
                ) for _ in range(n_unintentional)]
    i_policy = StochasticPolicy(
                hidden_sizes=[net_size, net_size],
                obs_dim=obs_dim,
                action_dim=action_dim,)

    replay_buffer = MultiGoalReplayBuffer(
        variant['algo_params']['replay_buffer_size'],
        np.prod(env.observation_space.shape),
        np.prod(env.action_space.shape),
        n_unintentional
    )
    variant['algo_params']['replay_buffer'] = replay_buffer

    # QF Plot
    goal_pos = expt_variant['env_params']['goal_position']
    q_fcn_positions = [
        (goal_pos[0], 0.0),
        (0.0, 0.0),
        (0.0, goal_pos[1])
    ]
    plotter = QFPolicyPlotter(
        i_qf=i_qf,
        i_policy=i_policy,
        u_qfs=u_qfs,
        u_policies=u_policies,
        obs_lst=q_fcn_positions,
        default_action=[np.nan, np.nan],
        n_samples=100,
        render=render_q,
        save_path=save_q_path,
    )
    variant['algo_params']['_epoch_plotter'] = plotter
    # variant['algo_params']['_epoch_plotter'] = None

    algorithm = IUSQL(
        env=env,
        training_env=env,
        save_environment=False,
        u_qfs=u_qfs,
        u_policies=u_policies,
        i_policy=i_policy,
        i_qf=i_qf,
        algo_interface='torch',
        min_buffer_size=variant['algo_params']['batch_size'],
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

    return algorithm


path_length = 30  # time steps
paths_per_epoch = 3
paths_per_eval = 3
paths_per_hard_update = 35

expt_params = dict(
    algo_params=dict(
        # Common RLAlgo params
        num_steps_per_epoch=paths_per_epoch * path_length,
        num_epochs=200,  # n_epochs
        num_updates_per_env_step=1,  # Like n_train_repeat??
        num_steps_per_eval=paths_per_eval * path_length,
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
        use_hard_updates=True,  # Hard update for target Q-fcn
        hard_update_period=paths_per_hard_update*path_length,  # td_target_update_interval (steps)
        soft_target_tau=0.001,  # Not used if use_hard_updates=True
        # TODO:kernel_fn
        kernel_n_particles=32,
        kernel_update_ratio=0.5,
        discount=0.99,
        reward_scale=0.1,
    ),
    net_size=32,
)

env_params = dict(
    goal_reward=10,
    actuation_cost_coeff=30,
    distance_cost_coeff=1,
    init_sigma=2.0,
    goal_position=None,
    dynamics_sigma=0.1
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
    parser.add_argument('--render_q', action="store_true")
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

    expt_variant['render_q'] = args.render_q

    # Experiment name
    if args.expt_name is None:
        expt_name = 'goalcompo'
    else:
        expt_name = args.expt_name

    expt_variant['algo_params']['render'] = args.render

    expt_variant['env_params'] = env_params

    # TODO: MAKE THIS A SCRIPT ARGUMENT
    expt_variant['env_params']['goal_position'] = (5, 5)

    setup_logger(expt_name,
                 variant=expt_variant,
                 snapshot_mode=args.snap_mode,
                 snapshot_gap=args.snap_gap,
                 log_dir=args.log_dir)
    algorithm = experiment(expt_variant)

    input('Press a key to close the script...')
