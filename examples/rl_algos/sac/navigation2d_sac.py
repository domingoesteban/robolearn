"""
Run PyTorch SAC on Navigation2dGoalCompoEnv.

NOTE: You need PyTorch 0.4
"""

import os
import numpy as np

import robolearn.torch.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.utils.data_management import SimpleReplayBuffer

from robolearn.envs.simple_envs.navigation2d import Navigation2dGoalCompoEnv

from robolearn.torch.rl_algos.sac import SAC

from robolearn.torch.models import NNQFunction
from robolearn.torch.models import NNVFunction

from robolearn.torch.policies import TanhGaussianPolicy

import argparse
import joblib

np.set_printoptions(suppress=True, precision=4)
# np.seterr(all='raise')  # WARNING RAISE ERROR IN NUMPY


PATH_LENGTH = 20  # time steps
PATHS_PER_EPOCH = 5
PATHS_PER_EVAL = 3
PATHS_PER_HARD_UPDATE = 35
BATCH_SIZE = 128

SEED = 110
# NP_THREADS = 6

POLICY = TanhGaussianPolicy
REPARAM_POLICY = True

USE_Q2 = False

OPTIMIZER = 'adam'
# OPTIMIZER = 'rmsprop'

expt_params = dict(
    algo_name=SAC.__name__,
    policy_name=POLICY.__name__,
    algo_params=dict(
        # Common RLAlgo params
        num_steps_per_epoch=PATHS_PER_EPOCH * PATH_LENGTH,
        num_epochs=10000,  # n_epochs
        num_updates_per_train_call=1,  # How to many run algorithm train fcn
        num_steps_per_eval=PATHS_PER_EVAL * PATH_LENGTH,
        min_steps_start_train=BATCH_SIZE,  # Min nsteps to start to train (or batch_size)
        min_start_eval=PATHS_PER_EPOCH * PATH_LENGTH,  # Min nsteps to start to eval
        # EnvSampler params
        max_path_length=PATH_LENGTH,  # max_path_length
        render=False,
        # SAC params
        reparameterize=REPARAM_POLICY,
        action_prior='uniform',
        entropy_scale=1.0e-0,
        # Learning rates
        policy_lr=1e-4,
        qf_lr=1e-4,
        vf_lr=1e-4,
        # Soft target update
        soft_target_tau=1.e-3,
        # Regularization terms
        policy_mean_regu_weight=1e-3,
        policy_std_regu_weight=1e-3,
        policy_pre_activation_weight=0.,
        # Weight decays
        policy_weight_decay=0.,
        q_weight_decay=0.,
        v_weight_decay=0.,

        discount=0.99,
        reward_scale=1.0e+1,
    ),
    net_size=32,
    replay_buffer_size=1e4,
    shared_layer_norm=False,
    # hidden_activation='relu',
    # hidden_activation='tanh',
    hidden_activation='elu',
)

env_params = dict(
    goal_reward=50,
    actuation_cost_coeff=0.5,
    distance_cost_coeff=1.5,
    log_distance_cost_coeff=10.0,
    alpha=1e-3,
    # Initial Condition
    init_position=(4., 4.),
    init_sigma=10.00,
    # Goal
    goal_position=(-2.0, -2.0),  # TODO: Make this a script param
    goal_threshold=0.50,
    # Others
    # dynamics_sigma=0.1,
    dynamics_sigma=0.0,
    # horizon=PATH_LENGTH,
    horizon=None,
)


def experiment(variant):

    # os.environ['OMP_NUM_THREADS'] = str(NP_THREADS)

    np.random.seed(SEED)

    ptu.set_gpu_mode(variant['gpu'])
    ptu.seed(SEED)

    env = NormalizedBoxEnv(
        Navigation2dGoalCompoEnv(**variant['env_params']),
        # normalize_obs=True,
        normalize_obs=False,
        online_normalization=False,
        obs_mean=None,
        obs_var=None,
        obs_alpha=0.001,
    )

    obs_dim = env.obs_dim
    action_dim = env.action_dim

    if variant['log_dir']:
        params_file = os.path.join(variant['log_dir'], 'params.pkl')
        data = joblib.load(params_file)
        start_epoch = data['epoch']
        policy = data['policy']
        qf = data['qf']
        qf2 = data['qf2']
        vf = data['vf']
        env._obs_mean = data['obs_mean']
        env._obs_var = data['obs_var']
    else:
        start_epoch = 0
        net_size = variant['net_size']

        qf = NNQFunction(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_activation=expt_params['hidden_activation'],
            hidden_sizes=[net_size, net_size],
        )
        if USE_Q2:
            qf2 = NNQFunction(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_activation=expt_params['hidden_activation'],
                hidden_sizes=[net_size, net_size],
            )
        else:
            qf2 = None
        vf = NNVFunction(
            obs_dim=obs_dim,
            hidden_activation=expt_params['hidden_activation'],
            hidden_sizes=[net_size, net_size],
        )
        policy = POLICY(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_activation=expt_params['hidden_activation'],
            hidden_sizes=[net_size, net_size],
            reparameterize=REPARAM_POLICY,
        )

        # Clamp model parameters
        qf.clamp_all_params(min=-0.003, max=0.003)
        vf.clamp_all_params(min=-0.003, max=0.003)
        policy.clamp_all_params(min=-0.003, max=0.003)
        if USE_Q2:
            qf2.clamp_all_params(min=-0.003, max=0.003)

    replay_buffer = SimpleReplayBuffer(
        max_replay_buffer_size=variant['replay_buffer_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    algorithm = SAC(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        replay_buffer=replay_buffer,
        batch_size=BATCH_SIZE,
        qf2=qf2,
        eval_env=env,
        save_environment=False,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()

    # algorithm.pretrain(PATH_LENGTH*2)
    algorithm.train(start_epoch=start_epoch)

    return algorithm


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
    parser.add_argument('--render_q', action="store_true")
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
        expt_name = 'navigation2d'
    else:
        expt_name = args.expt_name

    expt_variant['algo_params']['render'] = args.render

    expt_variant['env_params'] = env_params

    expt_variant['log_dir'] = args.log_dir

    setup_logger(expt_name,
                 variant=expt_variant,
                 snapshot_mode=args.snap_mode,
                 snapshot_gap=args.snap_gap,
                 log_dir=args.log_dir)
    algo = experiment(expt_variant)

    # input('Press a key to close the script...')
