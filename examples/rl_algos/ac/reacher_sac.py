"""
Run PyTorch SAC on Reacher2D3DofEnv.

NOTE: You need PyTorch 0.4
"""

import numpy as np

import robolearn.torch.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.utils.data_management import SimpleReplayBuffer
from robolearn_gym_envs.pybullet import Reacher2D3DofBulletEnv

from robolearn.torch.rl_algos.sac import SoftActorCritic

from robolearn.torch.models import NNQFunction, NNVFunction
from robolearn.torch.policies import TanhGaussianPolicy

import argparse


def experiment(variant):
    ptu.set_gpu_mode(variant['gpu'])

    env = NormalizedBoxEnv(
        Reacher2D3DofBulletEnv(**variant['env_params'])
    )

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']

    qf = NNQFunction(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[net_size, net_size]
    )
    vf = NNVFunction(
        obs_dim=obs_dim,
        hidden_sizes=[net_size, net_size]
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[net_size, net_size],
    )

    replay_buffer = SimpleReplayBuffer(
        variant['algo_params']['replay_buffer_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    variant['algo_params']['replay_buffer'] = replay_buffer

    # QF Plot
    # variant['algo_params']['epoch_plotter'] = None

    algorithm = SoftActorCritic(
        env=env,
        training_env=env,
        save_environment=False,
        policy=policy,
        qf=qf,
        vf=vf,
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
    algo_name=SoftActorCritic.__name__,
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
        replay_buffer_size=1e4,
        # SoftActorCritic params
        policy_lr=1e-3,
        qf_lr=1e-3,
        vf_lr=1e-3,
        soft_target_tau=1e-3,
        policy_mean_reg_weight=1e-3,
        policy_std_reg_weight=1e-3,
        policy_pre_activation_weight=0.,

        discount=0.99,
        # reward_scale=1.0,  # NO FUNCA 20/06
        reward_scale=10.0,
    ),
    net_size=64
)

SIM_TIMESTEP = 0.001
FRAME_SKIP = 10
DT = SIM_TIMESTEP * FRAME_SKIP

env_params = dict(
    is_render=False,
    obs_with_img=False,
    rdn_tgt_pos=True,
    tgt_pose=None,
    rdn_robot_config=True,
    robot_config=None,
    sim_timestep=SIM_TIMESTEP,
    frame_skip=FRAME_SKIP,
    # obs_distances=False,  # If True obs contain 'distance' vectors instead poses
    obs_distances=True,  # If True obs contain 'distance' vectors instead poses
    tgt_cost_weight=1.0,
    ctrl_cost_weight=1.0e-2,
    # use_log_distances=True,
    use_log_distances=False,
    log_alpha=1e-6,
    tgt_tolerance=0.05,
    # max_time=PATH_LENGTH*DT,
    max_time=None,
    half_env=False,
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
        expt_name = 'reacher'
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
