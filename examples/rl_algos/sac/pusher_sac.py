"""
Run PyTorch SAC on Pusher2D3DofGoalCompoEnv.

NOTE: You need PyTorch 0.4
"""

import os
from shutil import copyfile
import numpy as np

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.utils.data_management import SimpleReplayBuffer

from robolearn_gym_envs.pybullet import Pusher2D3DofGoalCompoEnv

from robolearn.torch.algorithms.rl_algos import SAC

from robolearn.torch.models import NNQFunction
from robolearn.torch.models import NNVFunction

from robolearn.torch.policies import TanhGaussianPolicy

import argparse
import joblib

np.set_printoptions(suppress=True, precision=4)
# np.seterr(all='raise')  # WARNING RAISE ERROR IN NUMPY

Tend = 5.0  # Seconds

SIM_TIMESTEP = 0.001
FRAME_SKIP = 10
DT = SIM_TIMESTEP * FRAME_SKIP

PATH_LENGTH = int(np.ceil(Tend / DT))
PATHS_PER_EPOCH = 3
PATHS_PER_EVAL = 2
PATHS_PER_HARD_UPDATE = 12
BATCH_SIZE = 512

SEED = 110
# NP_THREADS = 6

SUBTASK = None

POLICY = TanhGaussianPolicy
REPARAM_POLICY = True

USE_Q2 = False

OPTIMIZER = 'adam'
# OPTIMIZER = 'rmsprop'

NORMALIZE_OBS = False

expt_params = dict(
    algo_name=SAC.__name__,
    policy_name=POLICY.__name__,
    path_length=PATH_LENGTH,
    algo_params=dict(
        # Common RL algorithm params
        num_steps_per_epoch=PATHS_PER_EPOCH * PATH_LENGTH,
        num_epochs=5000,  # n_epochs
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
        entropy_scale=1.0e-1,
        policy_lr=1e-4,
        qf_lr=1e-4,
        vf_lr=1e-4,
        soft_target_tau=1.e-3,
        policy_mean_regu_weight=1.e-3,
        policy_std_regu_weight=1.e-3,
        policy_pre_activation_weight=0.e-3,
        policy_weight_decay=0.e-5,
        q_weight_decay=0.e-5,
        v_weight_decay=0.e-5,

        discount=0.99,
        reward_scale=1.0e-0,
    ),
    replay_buffer_size=1e6,
    net_size=128,
    shared_layer_norm=False,
    # NN Activations
    # --------------
    # hidden_activation='relu',
    # hidden_activation='tanh',
    hidden_activation='elu',
    # NN Initialization
    # -----------------
    # pol_hidden_w_init='xavier_normal',
    # pol_output_w_init='xavier_normal',
    pol_hidden_w_init='uniform',
    pol_output_w_init='uniform',
    # q_hidden_w_init='xavier_normal',
    # q_output_w_init='xavier_normal',
    q_hidden_w_init='uniform',
    q_output_w_init='uniform',
    # v_hidden_w_init='xavier_normal',
    # v_output_w_init='xavier_normal',
    v_hidden_w_init='uniform',
    v_output_w_init='uniform',
)

env_params = dict(
    is_render=False,
    # obs_distances=False,  # If True obs contain 'distance' vectors instead poses
    obs_distances=True,  # If True obs contain 'distance' vectors instead poses
    obs_with_img=False,
    obs_with_ori=False,
    goal_pose=(0.65, 0.65),
    rdn_goal_pose=True,
    tgt_pose=(0.5, 0.25, 1.4660),
    rdn_tgt_object_pose=True,
    robot_config=None,
    rdn_robot_config=True,
    tgt_cost_weight=3.0,
    goal_cost_weight=1.0,
    ctrl_cost_weight=1.0e-2,
    goal_tolerance=0.01,
    # max_time=PATH_LENGTH*DT,
    max_time=None,
    sim_timestep=SIM_TIMESTEP,
    frame_skip=FRAME_SKIP,
    subtask=SUBTASK,
    seed=SEED,
    sequential_reward=False,
)


def experiment(variant):

    # os.environ['OMP_NUM_THREADS'] = str(NP_THREADS)

    # Set seeds
    np.random.seed(variant['seed'])
    ptu.set_gpu_mode(variant['gpu'])
    ptu.seed(variant['seed'])
    variant['env_params']['seed'] = variant['seed']

    env = NormalizedBoxEnv(
        Pusher2D3DofGoalCompoEnv(**variant['env_params']),
        # normalize_obs=True,
        normalize_obs=False,
        online_normalization=False,
        obs_mean=None,
        obs_var=None,
        obs_alpha=0.001,
    )

    obs_dim = env.obs_dim
    action_dim = env.action_dim

    if variant['load_dir']:
        params_file = os.path.join(variant['log_dir'], 'params.pkl')
        data = joblib.load(params_file)
        start_epoch = data['epoch']
        qf = data['qf']
        qf2 = data['qf2']
        vf = data['vf']
        policy = data['policy']
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

        # # Clamp model parameters
        # qf.clamp_all_params(min=-0.003, max=0.003)
        # vf.clamp_all_params(min=-0.003, max=0.003)
        # policy.clamp_all_params(min=-0.003, max=0.003)
        # if USE_Q2:
        #     qf2.clamp_all_params(min=-0.003, max=0.003)

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
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--expt_name', type=str, default=None)
    parser.add_argument('--subtask', type=int, default=-1)
    # Logging arguments
    parser.add_argument('--snap_mode', type=str, default='gap_and_last')
    parser.add_argument('--snap_gap', type=int, default=25)
    # parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--load_dir', type=str, default=None)
    # GPU arguments
    parser.add_argument('--gpu', action="store_true")
    # Other arguments
    parser.add_argument('--render', action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # Experiment name
    if args.expt_name is None:
        expt_name = 'pusher'
    else:
        expt_name = args.expt_name

    # Default experiment parameters
    expt_variant = expt_params

    # Default environment parameters
    expt_variant['env_params'] = env_params
    expt_variant['env_params']['is_render'] = args.render

    # Custom parameters
    if args.subtask >= 0:
        expt_variant['env_params']['subtask'] = args.subtask

    expt_variant['log_dir'] = args.log_dir
    expt_variant['load_dir'] = args.load_dir

    # Net size
    if args.net_size is not None:
        expt_variant['net_size'] = args.net_size

    expt_variant['gpu'] = args.gpu

    expt_variant['seed'] = args.seed

    # Algo params
    expt_variant['algo_params']['render'] = args.render

    log_dir = setup_logger(expt_name,
                           variant=expt_variant,
                           snapshot_mode=args.snap_mode,
                           snapshot_gap=args.snap_gap,
                           log_dir=args.log_dir)
    dir_filename = os.path.realpath(__file__)
    filename = os.path.split(dir_filename)[1]
    copyfile(dir_filename, os.path.join(log_dir, filename))

    algo = experiment(expt_variant)

    # input('Press a key to close the script...')
