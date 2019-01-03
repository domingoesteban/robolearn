"""
Run PyTorch DDPG on Reacher2D3DofEnv.

NOTE: You need PyTorch 0.4
"""

import os
from shutil import copyfile
import numpy as np

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.utils.data_management import SimpleReplayBuffer

from robolearn_gym_envs.pybullet import Reacher2D3DofGoalCompoEnv

from robolearn.torch.algorithms.rl_algos.ddpg import DDPG

from robolearn.torch.models import NNQFunction

from robolearn.torch.policies import TanhMlpPolicy

from robolearn.utils.exploration_strategies import OUStrategy
from robolearn.utils.exploration_strategies import PolicyWrappedWithExplorationStrategy

import argparse
import joblib

np.set_printoptions(suppress=True, precision=4)
# np.seterr(all='raise')  # WARNING RAISE ERROR IN NUMPY

Tend = 4.0  # Seconds

SIM_TIMESTEP = 0.001
FRAME_SKIP = 10
DT = SIM_TIMESTEP * FRAME_SKIP

PATH_LENGTH = int(np.ceil(Tend / DT))
PATHS_PER_EPOCH = 3
PATHS_PER_EVAL = 2
PATHS_PER_HARD_UPDATE = 12
BATCH_SIZE = 128

SEED = 110
# NP_THREADS = 6

SUBTASK = None

POLICY = TanhMlpPolicy

expt_params = dict(
    algo_name=DDPG.__name__,
    policy_name=POLICY.__name__,
    path_length=PATH_LENGTH,
    algo_params=dict(
        # Common RL algorithm params
        num_steps_per_epoch=PATHS_PER_EPOCH * PATH_LENGTH,
        num_epochs=500,  # n_epochs
        num_updates_per_train_call=1,  # How to many run algorithm train fcn
        num_steps_per_eval=PATHS_PER_EVAL * PATH_LENGTH,
        min_steps_start_train=BATCH_SIZE,  # Min nsteps to start to train (or batch_size)
        min_start_eval=PATHS_PER_EPOCH * PATH_LENGTH,  # Min nsteps to start to eval
        # EnvSampler params
        max_path_length=PATH_LENGTH,  # max_path_length
        render=False,
        # DDPG params
        policy_learning_rate=3e-4,
        qf_learning_rate=3e-4,
        use_soft_update=True,
        tau=1e-2,

        discount=0.99,
        reward_scale=1.0,
    ),
    replay_buffer_size=1e6,
    net_size=128,
)


env_params = dict(
    is_render=False,
    # obs_distances=False,  # If True obs contain 'distance' vectors instead poses
    obs_distances=True,  # If True obs contain 'distance' vectors instead poses
    obs_with_img=False,
    # obs_with_ori=True,
    obs_with_ori=False,
    obs_with_goal=True,
    # obs_with_goal=False,
    # goal_pose=(0.65, 0.65),
    goal_pose=(0.65, 0.35),
    # rdn_goal_pos=True,
    rdn_goal_pos=False,
    robot_config=None,
    rdn_robot_config=True,
    goal_cost_weight=1.0e1,
    # goal_cost_weight=5.0e1,
    goal_tolerance=0.01,
    # ctrl_cost_weight=5.0e-0,  # It works with log_dist=False
    ctrl_cost_weight=1.0e+1,
    use_log_distances=True,
    log_alpha=1e-6,
    # max_time=PATH_LENGTH*DT,
    max_time=None,
    sim_timestep=SIM_TIMESTEP,
    frame_skip=FRAME_SKIP,
    half_env=True,
    subtask=SUBTASK,
    seed=SEED,
)


def experiment(variant):

    # os.environ['OMP_NUM_THREADS'] = str(NP_THREADS)

    # Set seeds
    np.random.seed(variant['seed'])
    ptu.set_gpu_mode(variant['gpu'], gpu_id=0)
    ptu.seed(variant['seed'])
    variant['env_params']['seed'] = variant['seed']

    env = NormalizedBoxEnv(
        Reacher2D3DofGoalCompoEnv(**variant['env_params']),
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
        raise NotImplementedError
    else:
        start_epoch = 0
        net_size = variant['net_size']

        qf = NNQFunction(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[net_size, net_size]
        )
        policy = POLICY(
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
        max_replay_buffer_size=variant['replay_buffer_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    algorithm = DDPG(
        env=env,
        policy=policy,
        exploration_policy=exploration_policy,
        qf=qf,
        replay_buffer=replay_buffer,
        batch_size=BATCH_SIZE,
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
        expt_name = 'reacher'
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
