"""
Run PyTorch DDPG on Pusher2D3DofGoalCompoEnv.

NOTE: You need PyTorch 0.4
"""

import os
import numpy as np

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.utils.data_management import SimpleReplayBuffer

from robolearn_gym_envs.pybullet import Pusher2D3DofGoalCompoEnv

from robolearn.torch.algorithms.rl_algos.ddpg import DDPG

from robolearn.torch.models import NNQFunction

from robolearn.torch.policies import TanhMlpPolicy

from robolearn.utils.exploration_strategies import OUStrategy
from robolearn.utils.exploration_strategies import \
    PolicyWrappedWithExplorationStrategy

import argparse
import joblib

np.set_printoptions(suppress=True, precision=4)
# np.seterr(all='raise')  # WARNING RAISE ERROR IN NUMPY

Tend = 3.0  # Seconds

SIM_TIMESTEP = 0.001
FRAME_SKIP = 10
DT = SIM_TIMESTEP * FRAME_SKIP

PATH_LENGTH = int(np.ceil(Tend / DT))
PATHS_PER_EPOCH = 5 #10
PATHS_PER_EVAL = 3 #3
BATCH_SIZE = 256

# SEED = 10
SEED = 110
# NP_THREADS = 6

expt_params = dict(
    algo_name=DDPG.__name__,
    algo_params=dict(
        # Common RL algorithm params
        num_steps_per_epoch=PATHS_PER_EPOCH * PATH_LENGTH,
        num_epochs=3000,  # n_epochs
        num_updates_per_train_call=1,  # How to many run algorithm train fcn
        num_steps_per_eval=PATHS_PER_EVAL * PATH_LENGTH,
        min_steps_start_train=BATCH_SIZE,  # Min nsteps to start to train (or batch_size)
        min_start_eval=PATHS_PER_EPOCH * PATH_LENGTH,  # Min nsteps to start to eval
        # EnvSampler params
        max_path_length=PATH_LENGTH,  # max_path_length
        render=False,
        # DDPG params
        policy_learning_rate=1e-4,
        qf_learning_rate=1e-3,
        use_soft_update=True,
        tau=1e-2,

        discount=0.99,
        reward_scale=1.0,
    ),
    net_size=64,
    replay_buffer_size=1e6,
)


env_params = dict(
    is_render=False,
    obs_with_img=False,
    goal_poses=None,  # It will be setted later
    rdn_goal_pose=True,
    tgt_pose=None,  # It will be setted later
    rdn_tgt_object_pose=True,
    sim_timestep=SIM_TIMESTEP,
    frame_skip=FRAME_SKIP,
    # obs_distances=False,  # If True obs contain 'distance' vectors instead poses
    obs_distances=True,  # If True obs contain 'distance' vectors instead poses
    tgt_cost_weight=2.0, #1.5
    goal_cost_weight=1.0,
    ctrl_cost_weight=1.0e-3,
    use_log_distances=True,
    # use_log_distances=False,
    log_alpha=1.e-1,  # In case use_log_distances=True
    # max_time=PATH_LENGTH*DT,
    max_time=None,
    subtask=None,
    seed=SEED,
)


def experiment(variant):

    # os.environ['OMP_NUM_THREADS'] = str(NP_THREADS)

    np.random.seed(SEED)

    ptu.set_gpu_mode(variant['gpu'])
    ptu.seed(SEED)

    goal = variant['env_params'].get('goal')
    variant['env_params']['goal_poses'] = \
        [goal, (goal[0], 'any'), ('any', goal[1])]
    variant['env_params'].pop('goal')

    env = NormalizedBoxEnv(
        Pusher2D3DofGoalCompoEnv(**variant['env_params']),
        # normalize_obs=True,
        normalize_obs=False,
        online_normalization=False,
        obs_mean=None,
        obs_var=None,
        obs_alpha=0.001,
    )

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    if variant['log_dir']:
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

        # Clamp model parameters
        qf.clamp_all_params(min=-0.003, max=0.003)
        policy.clamp_all_params(min=-0.003, max=0.003)

    replay_buffer = SimpleReplayBuffer(
        max_size=variant['replay_buffer_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    algorithm = DDPG(
        explo_env=env,
        policy=policy,
        explo_policy=exploration_policy,
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
        expt_name = 'pusher'
    else:
        expt_name = args.expt_name

    expt_variant['algo_params']['render'] = args.render

    expt_variant['env_params'] = env_params
    expt_variant['env_params']['is_render'] = args.render

    # TODO: MAKE THIS A SCRIPT ARGUMENT
    expt_variant['env_params']['goal'] = (0.65, 0.65)
    expt_variant['env_params']['tgt_pose'] = (0.5, 0.25, 1.4660)

    expt_variant['log_dir'] = args.log_dir

    setup_logger(expt_name,
                 variant=expt_variant,
                 snapshot_mode=args.snap_mode,
                 snapshot_gap=args.snap_gap,
                 log_dir=args.log_dir)
    algo = experiment(expt_variant)

    input('Press a key to close the script...')
