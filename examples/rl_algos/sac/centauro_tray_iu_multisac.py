"""
Run PyTorch IU Multi Soft Actor Critic on CentauroTrayEnv.

NOTE: You need PyTorch 0.4
"""

import os
import numpy as np

import robolearn.torch.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.utils.data_management import MultiGoalReplayBuffer

from robolearn_gym_envs.pybullet import CentauroTrayEnv

from robolearn.torch.rl_algos.sac.iu_multisac import IUMultiSAC


from robolearn.torch.models import NNQFunction, NNVFunction
from robolearn.torch.models import NNMultiQFunction, NNMultiVFunction

from robolearn.torch.policies import TanhGaussianMultiPolicy
from robolearn.torch.policies import MixtureTanhGaussianMultiPolicy

import argparse
import joblib

# np.seterr(all='raise')  # WARNING RAISE ERROR IN NUMPY

Tend = 3.0  # Seconds

SIM_TIMESTEP = 0.01
FRAME_SKIP = 1
DT = SIM_TIMESTEP * FRAME_SKIP

PATH_LENGTH = int(np.ceil(Tend / DT))
PATHS_PER_EPOCH = 2
PATHS_PER_EVAL = 1
PATHS_PER_HARD_UPDATE = 12
BATCH_SIZE = 256

SEED = 110
# NP_THREADS = 6

POLICY = TanhGaussianWeightedMultiPolicy3


def experiment(variant):

    # os.environ['OMP_NUM_THREADS'] = str(NP_THREADS)

    np.random.seed(SEED)

    ptu.set_gpu_mode(variant['gpu'])
    ptu.seed(SEED)

    env = NormalizedBoxEnv(
        CentauroTrayEnv(**variant['env_params']),
        # normalize_obs=True,
        normalize_obs=False,
        online_normalization=False,
        obs_mean=None,
        obs_var=None,
        obs_alpha=0.001,
    )

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    n_unintentional = 2

    net_size = variant['net_size']
    u_qf = NNMultiQFunction(obs_dim=obs_dim,
                            action_dim=action_dim,
                            n_qs=n_unintentional,
                            # shared_hidden_sizes=[net_size, net_size],
                            shared_hidden_sizes=[],
                            unshared_hidden_sizes=[net_size, net_size, net_size])
    i_qf = NNQFunction(obs_dim=obs_dim,
                       action_dim=action_dim,
                       hidden_sizes=[net_size, net_size])
    # i_qf = WeightedNNMultiVFunction(u_qf)

    u_vf = NNMultiVFunction(obs_dim=obs_dim,
                            n_vs=n_unintentional,
                            # shared_hidden_sizes=[net_size, net_size],
                            shared_hidden_sizes=[],
                            unshared_hidden_sizes=[net_size, net_size, net_size])
    i_vf = NNVFunction(obs_dim=obs_dim,
                       hidden_sizes=[net_size, net_size])
    # i_vf = WeightedNNMultiVFunction(u_vf)

    u_policy = TanhGaussianMultiPolicy(obs_dim=obs_dim,
                                       action_dim=action_dim,
                                       n_policies=n_unintentional,
                                       # shared_hidden_sizes=[net_size, net_size],
                                       shared_hidden_sizes=[],
                                       unshared_hidden_sizes=[net_size, net_size, net_size])
    i_policy = MixtureTanhGaussianMultiPolicy(u_policy,
                                              mix_hidden_sizes=[net_size, net_size],
                                              pol_idxs=None,
                                              optimize_multipolicy=False)

    replay_buffer = MultiGoalReplayBuffer(
        variant['algo_params']['replay_buffer_size'],
        np.prod(env.observation_space.shape),
        np.prod(env.action_space.shape),
        n_unintentional
    )
    variant['algo_params']['replay_buffer'] = replay_buffer

    # QF Plot
    variant['algo_params']['epoch_plotter'] = None

    algorithm = IUMultiSAC(
        env=env,
        training_env=env,
        save_environment=False,
        u_policy=u_policy,
        u_qf=u_qf,
        u_vf=u_vf,
        i_policy=i_policy,
        i_qf=i_qf,
        i_vf=i_vf,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

    return algorithm


expt_params = dict(
    algo_name=IUMultiSAC.__name__,
    algo_params=dict(
        # Common RLAlgo params
        num_steps_per_epoch=PATHS_PER_EPOCH * PATH_LENGTH,
        num_epochs=1500,  # n_epochs
        num_updates_per_train_call=1,  # How to many run algorithm train fcn
        num_steps_per_eval=PATHS_PER_EVAL * PATH_LENGTH,
        # EnvSampler params
        max_path_length=PATH_LENGTH,  # max_path_length
        render=False,
        # ReplayBuffer params
        batch_size=64,  # batch_size
        replay_buffer_size=1e6,
        # SAC params
        # TODO: epoch_plotter
        min_buffer_size=65,
        iu_mode='composition',
        i_policy_lr=1e-5,
        u_policy_lr=1e-3,
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
        # reward_scale=1.0,
        # reward_scale=5.0,
        # reward_scale=2.0,
        reward_scale=1.0,
    ),
    net_size=64,
)


env_params = dict(
    is_render=False,
    obs_with_img=False,
    active_joints='RA',
    control_type='tasktorque',
    # control_type='torque',
    sim_timestep=SIM_TIMESTEP,
    frame_skip=FRAME_SKIP,
    obs_distances=True,
    tgt_cost_weight=10.0,
    balance_cost_weight=6.0,
    fall_cost_weight=5.0,
    balance_done_cost=0.0,
    tgt_done_reward=0.0,
    # tgt_cost_weight=5.0,
    # balance_cost_weight=0.0,
    # fall_cost_weight=0.0,
    # tgt_cost_weight=0.0,
    # balance_cost_weight=5.0,
    # fall_cost_weight=7.0,
    ctrl_cost_weight=1.0e-4,
    use_log_distances=False,
    log_alpha=1e-6,
    goal_tolerance=0.05,
    min_obj_height=0.75,
    max_obj_height=1.10,
    max_obj_distance=0.20,
    max_time=None,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_size', type=int, default=None)
    parser.add_argument('--expt_name', type=str, default=None)
    # parser.add_argument('--expt_name', type=str, default=timestamp())
    # Logging arguments
    parser.add_argument('--snap_mode', type=str, default='gap_and_last')
    parser.add_argument('--snap_gap', type=int, default=25)
    # parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    # GPU arguments
    parser.add_argument('--gpu', action="store_true")
    # Other arguments
    parser.add_argument('--render', action="store_true")
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
        expt_name = 'centauro_tray_multisac'
    else:
        expt_name = args.expt_name

    expt_variant['algo_params']['render'] = args.render

    expt_variant['env_params'] = env_params
    expt_variant['env_params']['is_render'] = args.render

    expt_variant['log_dir'] = args.log_dir

    setup_logger(expt_name,
                 variant=expt_variant,
                 snapshot_mode=args.snap_mode,
                 snapshot_gap=args.snap_gap,
                 log_dir=args.log_dir)
    algo = experiment(expt_variant)

    input('Press a key to close the script...')
