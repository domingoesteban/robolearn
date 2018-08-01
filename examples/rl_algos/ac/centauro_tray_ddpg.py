"""
Run PyTorch DDPG on CentauroTrayEnv.

NOTE: You need PyTorch 0.4
"""

import os
import numpy as np

import robolearn.torch.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.utils.data_management import SimpleReplayBuffer
from robolearn_gym_envs.pybullet import CentauroTrayEnv

from robolearn.torch.rl_algos.ddpg import DDPG

from robolearn.torch.models import NNQFunction

from robolearn.torch.policies import TanhMlpPolicy

from robolearn.utils.exploration_strategies import OUStrategy
from robolearn.utils.exploration_strategies import PolicyWrappedWithExplorationStrategy

import argparse

Tend = 5  # Seconds

SIM_TIMESTEP = 0.01
FRAME_SKIP = 1
DT = SIM_TIMESTEP * FRAME_SKIP

PATH_LENGTH = int(np.ceil(Tend / DT))
PATHS_PER_EPOCH = 5
# PATHS_PER_LOCAL_POL = 2
PATHS_PER_EVAL = 1
PATHS_PER_HARD_UPDATE = 12

SEED = 10
NP_THREADS = 4


def experiment(variant):

    os.environ['OMP_NUM_THREADS'] = str(NP_THREADS)

    np.random.seed(SEED)

    ptu.set_gpu_mode(variant['gpu'])
    ptu.seed(SEED)

    env = NormalizedBoxEnv(
        CentauroTrayEnv(**variant['env_params'])
    )

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']

    qf = NNQFunction(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[net_size, net_size]
    )
    policy = TanhMlpPolicy(
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
        max_replay_buffer_size=variant['algo_params']['replay_buffer_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    variant['algo_params']['replay_buffer'] = replay_buffer

    # QF Plot
    # variant['algo_params']['epoch_plotter'] = None

    algorithm = DDPG(
        env=env,
        training_env=env,
        save_environment=False,
        policy=policy,
        exploration_policy=exploration_policy,
        qf=qf,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

    return algorithm


expt_params = dict(
    algo_name=DDPG.__name__,
    algo_params=dict(
        # Common RLAlgo params
        num_epochs=1000,  # n_epochs
        num_steps_per_epoch=PATHS_PER_EPOCH * PATH_LENGTH,
        num_updates_per_train_call=1,  # How to many run algorithm train fcn
        num_steps_per_eval=PATHS_PER_EVAL * PATH_LENGTH,
        # EnvSampler params
        max_path_length=PATH_LENGTH,  # max_path_length
        render=False,
        # ReplayBuffer params
        batch_size=64,  # batch_size
        replay_buffer_size=1e6,
        # DDPG params
        # TODO: epoch_plotter
        policy_learning_rate=1e-4,
        qf_learning_rate=1e-3,
        use_soft_update=True,
        tau=1e-2,

        discount=0.99,
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
        expt_name = 'centauro_tray_ddpg'
    else:
        expt_name = args.expt_name

    expt_variant['algo_params']['render'] = args.render

    expt_variant['env_params'] = env_params
    expt_variant['env_params']['is_render'] = args.render

    setup_logger(expt_name,
                 variant=expt_variant,
                 snapshot_mode=args.snap_mode,
                 snapshot_gap=args.snap_gap,
                 log_dir=args.log_dir)
    algo = experiment(expt_variant)

    input('Press a key to close the script...')
