"""
Run PyTorch IU-Multi-SAC on Pusher2D3DofGoalCompoEnv.

NOTE: You need PyTorch 0.4
"""

import os
import numpy as np
import torch

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.utils.data_management import MultiGoalReplayBuffer

from robolearn_gym_envs.pybullet import Pusher2D3DofGoalCompoEnv

from robolearn.torch.algorithms.rl_algos.ddpg \
    import IUWeightedMultiDDPG

from robolearn.torch.models import NNQFunction
from robolearn.torch.models import NNMultiQFunction
# from robolearn.torch.models import AvgNNQFunction, AvgNNVFunction

from robolearn.torch.policies import WeightedTanhMlpMultiPolicy

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
PATHS_PER_HARD_UPDATE = 12
BATCH_SIZE = 512

# SEED = 10
SEED = 110
# NP_THREADS = 6

# POLICY = MixtureTanhGaussianMultiPolicy
POLICY = WeightedTanhMlpMultiPolicy
REPARAM_POLICY = True
SOFTMAX_WEIGHTS = True
# SOFTMAX_WEIGHTS = False

USE_Q2 = True

expt_params = dict(
    algo_name=IUWeightedMultiDDPG.__name__,
    policy_name=POLICY.__name__,
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
        # SAC params
        reparameterize=REPARAM_POLICY,
        action_prior='uniform',

        i_policy_lr=1.e-4,
        u_policies_lr=1.e-4,
        u_mixing_lr=1.e-4,
        i_qf_lr=1.e-4,
        u_qf_lr=1.e-4,

        i_soft_target_tau=1.e-3,
        u_soft_target_tau=1.e-3,

        i_policy_pre_activation_weight=0.e-3,
        i_policy_mixing_coeff_weight=0.e-3,
        # u_policy_pre_activation_weight=[0.e-3, 0.e-3],

        i_policy_weight_decay=1.e-5,
        u_policy_weight_decay=1.e-5,
        i_q_weight_decay=1e-5,
        u_q_weight_decay=1e-5,

        discount=0.99,
        reward_scale=1.0e+0,
        # u_reward_scales=[1.0e+0, 1.0e+0],

        eval_with_target_policy=False,
    ),
    net_size=64,
    replay_buffer_size=1e4,
    shared_layer_norm=False,
    policies_layer_norm=False,
    mixture_layer_norm=False,
    # shared_layer_normTrue,
    # policies_layer_norm=True,
    # mixture_layer_norm=True,
)


env_params = dict(
    is_render=False,
    obs_distances=False,  # If True obs contain 'distance' vectors instead poses
    # obs_distances=False,  # If True obs contain 'distance' vectors instead poses
    obs_with_img=False,
    obs_with_ori=False,
    goal_poses=None,  # It will be setted later
    rdn_goal_pose=True,
    tgt_pose=None,  # It will be setted later
    rdn_tgt_object_pose=True,
    robot_config=None,
    rdn_robot_config=True,
    tgt_cost_weight=3.0,
    goal_cost_weight=1.0,
    ctrl_cost_weight=1.0e-3,
    goal_tolerance=0.01,
    # max_time=PATH_LENGTH*DT,
    max_time=None,
    sim_timestep=SIM_TIMESTEP,
    frame_skip=FRAME_SKIP,
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

    n_unintentional = 1

    if variant['log_dir']:
        params_file = os.path.join(variant['log_dir'], 'params.pkl')
        data = joblib.load(params_file)
        start_epoch = data['epoch']
        u_qf = data['u_qf']
        i_qf = data['qf']
        policy = data['policy']
        exploration_policy = data['exploration_policy']
        env._obs_mean = data['obs_mean']
        env._obs_var = data['obs_var']
    else:
        start_epoch = 0
        net_size = variant['net_size']
        u_qf = NNMultiQFunction(obs_dim=obs_dim,
                                action_dim=action_dim,
                                n_qs=n_unintentional,
                                # shared_hidden_sizes=[net_size, net_size],
                                shared_hidden_sizes=[],
                                unshared_hidden_sizes=[net_size, net_size])
        # i_qf = WeightedNNMultiVFunction(u_qf)
        i_qf = NNQFunction(obs_dim=obs_dim,
                           action_dim=action_dim,
                           hidden_sizes=[net_size, net_size])

        policy = POLICY(
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
            mixing_temperature=1.,
            reparameterize=REPARAM_POLICY,
            softmax_weights=SOFTMAX_WEIGHTS,
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
        policy.clamp_all_params(min=-0.003, max=0.003)
        u_qf.clamp_all_params(min=-0.003, max=0.003)
        i_qf.clamp_all_params(min=-0.003, max=0.003)

        if not SOFTMAX_WEIGHTS:
            set_average_mixing(policy, n_unintentional, obs_dim,
                               batch_size=50,
                               total_iters=1000)

    replay_buffer = MultiGoalReplayBuffer(
        max_replay_buffer_size=variant['replay_buffer_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_vector_size=n_unintentional,
    )

    algorithm = IUWeightedMultiDDPG(
        env=env,
        policy=policy,
        exploration_policy=exploration_policy,
        u_qf=u_qf,
        replay_buffer=replay_buffer,
        batch_size=BATCH_SIZE,
        i_qf=i_qf,
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


def set_average_mixing(policy, n_unintentional, obs_dim, batch_size=50,
                       total_iters=1000):
    mixing_optimizer = torch.optim.Adam(
        policy.mixing_parameters(),
        lr=1.0e-4,
        amsgrad=True,
        weight_decay=1e-5,
    )
    loss_fn = torch.nn.MSELoss(size_average=False)
    for ii in range(total_iters):
        dummy_obs = torch.randn((batch_size, obs_dim))
        mix_pred = policy(dummy_obs)[1]['mixing_coeff']
        mix_des = torch.ones_like(mix_pred) * 1./n_unintentional
        loss = loss_fn(mix_pred, mix_des)
        mixing_optimizer.zero_grad()
        loss.backward()
        mixing_optimizer.step()


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

    # input('Press a key to close the script...')
