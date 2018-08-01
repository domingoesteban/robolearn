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

from robolearn.torch.rl_algos.sac.iu_weightedmultisac import IUWeightedMultiSAC

from robolearn.torch.models import NNQFunction, NNVFunction
from robolearn.torch.models import NNMultiQFunction, NNMultiVFunction

from robolearn.torch.policies import TanhGaussianWeightedMultiPolicy

import argparse
import joblib

Tend = 0.5  # Seconds

SIM_TIMESTEP = 0.01
FRAME_SKIP = 1
DT = SIM_TIMESTEP * FRAME_SKIP

PATH_LENGTH = int(np.ceil(Tend / DT))
PATHS_PER_EPOCH = 5
# PATHS_PER_LOCAL_POL = 2
PATHS_PER_EVAL = 1
PATHS_PER_HARD_UPDATE = 12
BATCH_SIZE = 128

SEED = 10
NP_THREADS = 4


def experiment(variant):

    os.environ['OMP_NUM_THREADS'] = str(NP_THREADS)

    np.random.seed(SEED)

    ptu.set_gpu_mode(variant['gpu'])
    ptu.seed(SEED)

    env = NormalizedBoxEnv(
        CentauroTrayEnv(**variant['env_params']),
        reward_scale=1.,
        obs_mean=None,
        obs_std=None,
        online_normalization=True,
        # online_normalization=False,
    )

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    n_unintentional = 1

    if variant['log_dir']:
        params_file = os.path.join(variant['log_dir'], 'params.pkl')
        data = joblib.load(params_file)
        start_epoch = data['epoch']
        u_qf = data['u_qf']
        i_qf = data['qf']
        u_vf = data['u_vf']
        i_vf = data['vf']
        policy = data['policy']
        env._obs_mean = data['obs_mean']
        env._obs_std = data['obs_std']
    else:
        start_epoch = 0
        net_size = variant['net_size']
        u_qf = NNMultiQFunction(obs_dim=obs_dim,
                                action_dim=action_dim,
                                n_qs=n_unintentional,
                                # shared_hidden_sizes=[net_size, net_size],
                                shared_hidden_sizes=[],
                                unshared_hidden_sizes=[net_size, net_size])
        i_qf = NNQFunction(obs_dim=obs_dim,
                           action_dim=action_dim,
                           hidden_sizes=[net_size, net_size])

        u_vf = NNMultiVFunction(obs_dim=obs_dim,
                                n_vs=n_unintentional,
                                # shared_hidden_sizes=[net_size, net_size],
                                shared_hidden_sizes=[],
                                unshared_hidden_sizes=[net_size, net_size])
        i_vf = NNVFunction(obs_dim=obs_dim,
                           hidden_sizes=[net_size, net_size])

        policy = TanhGaussianWeightedMultiPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_policies=n_unintentional,
            # shared_hidden_sizes=[net_size, net_size],
            shared_hidden_sizes=[],
            unshared_hidden_sizes=[net_size, net_size],
            unshared_mix_hidden_sizes=[net_size, net_size],
            stds=None,
            shared_layer_norm=variant['shared_layer_norm'],
            policies_layer_norm=variant['policies_layer_norm'],
            mixture_layer_norm=variant['mixture_layer_norm'],
            mixing_temperature=1,
        )

    replay_buffer = MultiGoalReplayBuffer(
        max_replay_buffer_size=variant['replay_buffer_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_vector_size=n_unintentional,
    )

    # QF Plot
    variant['algo_params']['epoch_plotter'] = None

    algorithm = IUWeightedMultiSAC(
        env=env,
        policy=policy,
        u_qf=u_qf,
        u_vf=u_vf,
        replay_buffer=replay_buffer,
        batch_size=BATCH_SIZE,  # batch_size
        i_qf=i_qf,
        i_vf=i_vf,
        eval_env=env,
        save_environment=False,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train(start_epoch=start_epoch)

    return algorithm


expt_params = dict(
    algo_name=IUWeightedMultiSAC.__name__,
    algo_params=dict(
        # Common RL algorithm params
        num_steps_per_epoch=PATHS_PER_EPOCH * PATH_LENGTH,
        num_epochs=30000,  # n_epochs
        num_updates_per_train_call=1,  # How to many run algorithm train fcn
        num_steps_per_eval=PATHS_PER_EVAL * PATH_LENGTH,
        # EnvSampler params
        max_path_length=PATH_LENGTH,  # max_path_length
        render=False,
        # ReplayBuffer params
        # SoftActorCritic params
        min_steps_start_train=BATCH_SIZE,  # Min nsteps to start to train (or batch_size)
        min_start_eval=PATHS_PER_EPOCH * PATH_LENGTH,  # Min nsteps to start to eval
        policy_lr=1.e-4,
        policies_lr=1.e-4,
        mixing_lr=1.e-4,
        i_qf_lr=1.e-3,
        i_vf_lr=1.e-2,
        u_qf_lr=1.e-3,
        u_vf_lr=1.e-2,
        i_soft_target_tau=1.e-2,
        u_soft_target_tau=1.e-2,
        # policy_mean_regu_weight=1e-3,
        # policy_std_regu_weight=1e-3,
        # policy_mixing_coeff_weight=1e-3,
        i_policy_mean_regu_weight=0.,
        i_policy_std_regu_weight=0.,
        i_policy_pre_activation_weight=0.,
        i_policy_mixing_coeff_weight=0,

        u_policy_mean_regu_weight=[0., 0.],
        u_policy_std_regu_weight=[0., 0.],
        u_policy_pre_activation_weight=[0., 0.],

        # discount=0.99,
        discount=0.90,
        # discount=0.000,
        # reward_scale=1.0,
        # reward_scale=0.01,
        reward_scale=10.0,
        # reward_scale=1000.0,
        u_reward_scales=[10.0, 10.0],
    ),
    net_size=64,
    replay_buffer_size=1e6,
    # shared_layer_norm=False,
    # policies_layer_norm=False,
    # mixture_layer_norm=False,
    shared_layer_norm=True,
    policies_layer_norm=True,
    mixture_layer_norm=True,
)


env_params = dict(
    is_render=False,
    obs_with_img=False,
    active_joints='RA',
    control_type='tasktorque',
    # control_type='torque',
    # control_type='velocity',
    sim_timestep=SIM_TIMESTEP,
    frame_skip=FRAME_SKIP,
    obs_distances=False,
    balance_cost_weight=2.0,
    fall_cost_weight=2.0,
    tgt_cost_weight=2.0,
    balance_done_cost=0.0,
    tgt_done_reward=20.0,
    # tgt_cost_weight=5.0,
    # balance_cost_weight=0.0,
    # fall_cost_weight=0.0,
    # tgt_cost_weight=0.0,
    # balance_cost_weight=5.0,
    # fall_cost_weight=7.0,
    ctrl_cost_weight=1.0e-4,
    use_log_distances=True,
    log_alpha_pos=1e-2,
    log_alpha_ori=1e-2,
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
        expt_name = 'centauro_tray'
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
