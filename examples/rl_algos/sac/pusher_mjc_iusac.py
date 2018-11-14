"""
Run PyTorch Soft Q-learning on TwoGoalEnv.

NOTE: You need PyTorch 0.4
"""

import numpy as np

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.utils.data_management.multigoal_replay_buffer import MultiGoalReplayBuffer

# from robolearn_gym_envs.pybullet import Pusher2D3DofGoalCompoEnv
from robolearn.envs.rllab_envs.pusher import PusherEnv

from robolearn.torch.sac.iu_sac import IUSAC

from robolearn.torch.sac.value_functions import NNQFunction, NNVFunction
from robolearn.torch.sac.policies import TanhGaussianPolicy

import argparse


def experiment(variant):
    ptu.set_gpu_mode(variant['gpu'])

    env = NormalizedBoxEnv(
        PusherEnv(**variant['env_params'])
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
                       action_dim=action_dim)
    # i_qf = AvgNNQFunction(obs_dim=obs_dim,
    #                       action_dim=action_dim,
    #                       q_functions=u_qfs)
    # i_qf = SumNNQFunction(obs_dim=obs_dim,
    #                       action_dim=action_dim,
    #                       q_functions=u_qfs)

    u_vfs = [NNVFunction(obs_dim=obs_dim,
                         hidden_sizes=(net_size, net_size))
             for _ in range(n_unintentional)]
    i_vf = NNVFunction(obs_dim=obs_dim)
    # i_vf = AvgNNVFunction(obs_dim=obs_dim,
    #                       v_functions=u_vfs)
    # i_vf = SumNNVFunction(obs_dim=obs_dim,
    #                       v_functions=u_vfs)

    u_policies = [TanhGaussianPolicy(
                hidden_sizes=[net_size, net_size],
                obs_dim=obs_dim,
                action_dim=action_dim,
                ) for _ in range(n_unintentional)]
    i_policy = TanhGaussianPolicy(
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
        min_buffer_size=variant['algo_params']['batch_size'],
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

    return algorithm


path_length = 500
paths_per_epoch = 5
paths_per_eval = 1
paths_per_hard_update = 12

expt_params = dict(
    algo_params=dict(
        # Common RLAlgo params
        num_steps_per_epoch=paths_per_epoch * path_length,
        num_epochs=1000,  # n_epochs
        num_updates_per_env_step=1,  # Like n_train_repeat??
        num_steps_per_eval=paths_per_eval * path_length,
        # EnvSampler params
        max_path_length=path_length,  # max_path_length
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
        # hard_update_period=PATHS_PER_HARD_UPDATE*PATH_LENGTH,  # td_target_update_interval (steps)
        soft_target_tau=1e-2,  # Not used if use_hard_updates=True
        # TODO:kernel_fn
        policy_mean_reg_weight=1e-3,
        policy_std_reg_weight=1e-3,
        policy_pre_activation_weight=0.,

        discount=0.99,
        reward_scale=1.0,
    ),
    net_size=64,
)

env_params = dict(
    goal=(-1, -1),
    arm_distance_coeff=0,
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
        expt_name = 'pusher_mjc_compo'
    else:
        expt_name = args.expt_name

    expt_variant['algo_params']['render'] = args.render

    expt_variant['env_params'] = env_params

    setup_logger(expt_name,
                 variant=expt_variant,
                 snapshot_mode=args.snap_mode,
                 snapshot_gap=args.snap_gap,
                 log_dir=args.log_dir)
    algorithm = experiment(expt_variant)

    input('Press a key to close the script...')
