"""
Run PyTorch Soft Q-learning on TwoGoalEnv.

NOTE: You need PyTorch 0.4
"""

import numpy as np

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.utils.data_management.multigoal_replay_buffer import MultiGoalReplayBuffer

from robolearn_gym_envs.pybullet import Pusher2D3DofGoalCompoEnv

from robolearn.torch.algorithms.rl_algos import IUSAC

from robolearn.torch.models import NNQFunction, NNVFunction
# from robolearn.torch.sac.value_functions import AvgNNQFunction, AvgNNVFunction
# from robolearn.torch.sac.value_functions import SumNNQFunction, SumNNVFunction
from robolearn.torch.policies import TanhGaussianPolicy

import argparse


def experiment(variant):
    ptu.set_gpu_mode(variant['gpu'])

    goal = variant['env_params'].get('goal')
    variant['env_params']['goal_poses'] = \
        [goal, (goal[0], 'any'), ('any', goal[1])]
    variant['env_params'].pop('goal')

    env = NormalizedBoxEnv(
        Pusher2D3DofGoalCompoEnv(**variant['env_params'])
    )

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    n_unintentional = 2

    net_size = variant['net_size']
    u_qfs = [NNQFunction(obs_dim=obs_dim,
                         action_dim=action_dim,
                         hidden_sizes=(net_size, net_size))
             for _ in range(n_unintentional)]
    i_qf = NNQFunction(obs_dim=obs_dim,
                       action_dim=action_dim,
                       hidden_sizes=(net_size, net_size))
    # i_qf = AvgNNQFunction(obs_dim=obs_dim,
    #                       action_dim=action_dim,
    #                       q_functions=u_qfs)
    # i_qf = SumNNQFunction(obs_dim=obs_dim,
    #                       action_dim=action_dim,
    #                       q_functions=u_qfs)

    u_vfs = [NNVFunction(obs_dim=obs_dim,
                         hidden_sizes=(net_size, net_size))
             for _ in range(n_unintentional)]
    i_vf = NNVFunction(obs_dim=obs_dim,
                       hidden_sizes=(net_size, net_size))

    # i_vf = AvgNNVFunction(obs_dim=obs_dim,
    #                       v_functions=u_vfs)
    # i_vf = SumNNVFunction(obs_dim=obs_dim,
    #                       v_functions=u_vfs)

    u_policies = [TanhGaussianPolicy(
                      obs_dim=obs_dim,
                      action_dim=action_dim,
                      hidden_sizes=[net_size, net_size],
                      )
                  for _ in range(n_unintentional)]
    i_policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[net_size, net_size],
        )

    replay_buffer = MultiGoalReplayBuffer(
        variant['algo_params']['replay_buffer_size'],
        np.prod(env.observation_space.shape),
        np.prod(env.action_space.shape),
        n_unintentional
    )
    variant['algo_params']['replay_buffer'] = replay_buffer

    # QF Plot
    variant['algo_params']['epoch_plotter'] = None

    algorithm = IUSAC(
        env=env,
        training_env=env,
        save_environment=False,
        u_policies=u_policies,
        u_qfs=u_qfs,
        u_vfs=u_vfs,
        i_policy=i_policy,
        i_qf=i_qf,
        i_vf=i_vf,
        algo_interface='torch',
        min_buffer_size=variant['algo_params']['batch_size'],
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

    return algorithm


PATH_LENGTH = 500
PATHS_PER_EPOCH = 5
PATHS_PER_EVAL = 1

expt_params = dict(
    algo_params=dict(
        # Common RLAlgo params
        num_steps_per_epoch=PATHS_PER_EPOCH * PATH_LENGTH,
        num_epochs=1000,  # n_epochs
        num_updates_per_env_step=1,  # Like n_train_repeat??
        num_steps_per_eval=PATHS_PER_EVAL * PATH_LENGTH,
        # EnvSampler params
        max_path_length=PATH_LENGTH,  # max_path_length
        render=False,
        # ReplayBuffer params
        batch_size=64,  # batch_size
        replay_buffer_size=1e4,
        # SAC params
        # TODO: epoch_plotter
        # iu_mode='composition',
        iu_mode='intentional',
        policy_lr=1e-3,
        qf_lr=1e-3,
        vf_lr=1e-3,
        # use_hard_updates=False,  # Hard update for target Q-fcn
        soft_target_tau=1e-2,  # Not used if use_hard_updates=True
        # TODO:kernel_fn
        policy_mean_reg_weight=1e-3,
        policy_std_reg_weight=1e-3,
        policy_pre_activation_weight=0.,

        discount=0.99,
        reward_scale=0.1,
    ),
    net_size=64,
)

DT = 0.01

env_params = dict(
    is_render=False,
    obs_with_img=False,
    goal_poses=None,
    rdn_goal_pose=True,
    tgt_pose=None,
    rdn_tgt_object_pose=True,
    sim_timestep=0.001,
    frame_skip=10,
    obs_distances=False,
    tgt_cost_weight=1.5,
    # goal_cost_weight=1.0,
    goal_cost_weight=0.0,
    ctrl_cost_weight=1.0e-4,
    max_time=PATH_LENGTH*DT,
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
        expt_name = 'pusher_compo'
    else:
        expt_name = args.expt_name

    expt_variant['algo_params']['render'] = args.render

    expt_variant['env_params'] = env_params
    expt_variant['env_params']['is_render'] = args.render

    # TODO: MAKE THIS A SCRIPT ARGUMENT
    expt_variant['env_params']['goal'] = (0.75, 0.75)
    expt_variant['env_params']['tgt_pose'] = (0.6, 0.25, 1.4660)

    setup_logger(expt_name,
                 variant=expt_variant,
                 snapshot_mode=args.snap_mode,
                 snapshot_gap=args.snap_gap,
                 log_dir=args.log_dir)
    algorithm = experiment(expt_variant)

    input('Press a key to close the script...')
