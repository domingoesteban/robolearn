"""
Run PyTorch IU Multi Soft Actor Critic on Pusher2D3DofGoalCompoEnv.

NOTE: You need PyTorch 0.4
"""

import numpy as np

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.utils.data_management import MultiGoalReplayBuffer

from robolearn_gym_envs.pybullet import Pusher2D3DofGoalCompoEnv

from robolearn.torch.algorithms.rl_algos.sac.iu_multisac import IUMultiSAC


from robolearn.torch.models import NNQFunction, NNVFunction
from robolearn.torch.models import NNMultiQFunction, NNMultiVFunction

from robolearn.torch.policies import TanhGaussianMultiPolicy
from robolearn.torch.policies import TanhGaussianMixtureMultiPolicy
# from robolearn.torch.sac.policies import WeightedTanhGaussianMultiPolicy
# from robolearn.torch.sac.policies import MultiPolicySelector
# from robolearn.torch.sac.policies import BernoulliTanhGaussianMultiPolicy

import argparse

Tend = 5  # Seconds

SIM_TIMESTEP = 0.001
FRAME_SKIP = 10
DT = SIM_TIMESTEP * FRAME_SKIP

PATH_LENGTH = int(np.ceil(Tend / DT))
PATHS_PER_EPOCH = 5
# PATHS_PER_LOCAL_POL = 2
PATHS_PER_EVAL = 1
PATHS_PER_HARD_UPDATE = 12


def experiment(variant):
    ptu.set_gpu_mode(variant['gpu'])

    goal = variant['env_params'].get('goal')
    variant['env_params']['goal_poses'] = \
        [goal, (goal[0], 'any'), ('any', goal[1])]
    variant['env_params'].pop('goal')

    env = NormalizedBoxEnv(
        Pusher2D3DofGoalCompoEnv(**variant['env_params'])
    )

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    n_unintentional = 2

    net_size = variant['net_size']
    u_qf = NNMultiQFunction(obs_dim=obs_dim,
                            action_dim=action_dim,
                            n_qs=n_unintentional,
                            shared_hidden_sizes=[net_size, net_size],
                            unshared_hidden_sizes=[net_size, net_size, net_size])
    i_qf = NNQFunction(obs_dim=obs_dim,
                       action_dim=action_dim,
                       hidden_sizes=[net_size, net_size])
    # i_qf = WeightedNNMultiVFunction(u_qf)

    u_vf = NNMultiVFunction(obs_dim=obs_dim,
                            n_vs=n_unintentional,
                            shared_hidden_sizes=[net_size, net_size],
                            unshared_hidden_sizes=[net_size, net_size, net_size])
    i_vf = NNVFunction(obs_dim=obs_dim,
                       hidden_sizes=[net_size, net_size])
    # i_vf = WeightedNNMultiVFunction(u_vf)

    u_policy = TanhGaussianMultiPolicy(obs_dim=obs_dim,
                                       action_dim=action_dim,
                                       n_policies=n_unintentional,
                                       shared_hidden_sizes=[net_size, net_size],
                                       unshared_hidden_sizes=[net_size, net_size, net_size])
    i_policy = TanhGaussianMixtureMultiPolicy(u_policy,
                                              mix_hidden_sizes=[net_size, net_size],
                                              pol_idxs=None,
                                              optimize_multipolicy=False)
    # i_policy = MultiPolicySelector(u_policy, 0)
    # i_policy = BernoulliTanhGaussianMultiPolicy(u_policy, prob=0.5)

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
        min_buffer_size=variant['algo_params']['batch_size'],
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
        iu_mode='composition',
        policy_lr=1e-3,
        qf_lr=1e-4,
        vf_lr=1e-4,
        # use_hard_updates=False,  # Hard update for target Q-fcn
        # hard_update_period=PATHS_PER_HARD_UPDATE*PATH_LENGTH,  # td_target_update_interval (steps)
        soft_target_tau=1e-2,  # Not used if use_hard_updates=True
        # TODO:kernel_fn
        policy_mean_reg_weight=1e-3,
        policy_std_reg_weight=1e-3,
        policy_pre_activation_weight=0.,

        discount=0.99,
        # reward_scale=1.5,
        # reward_scale=0.5,
        # reward_scale=1.0,  # MUY ESTOCASTICA, CASI SIN MOVER. 20/06
        reward_scale=10.0,
    ),
    net_size=64,
)


env_params = dict(
    is_render=False,
    obs_with_img=False,
    goal_poses=None,
    rdn_goal_pose=True,
    tgt_pose=None,
    rdn_tgt_object_pose=True,
    sim_timestep=SIM_TIMESTEP,
    frame_skip=FRAME_SKIP,
    obs_distances=False,  # If True obs contain 'distance' vectors instead poses
    tgt_cost_weight=1.0, #1.5,
    # goal_cost_weight=1.5, #3.0,
    goal_cost_weight=1.5,
    # goal_cost_weight=0.0,
    ctrl_cost_weight=1.0e-4,
    use_log_distances=True,
    # use_log_distances=False,
    log_alpha=1e-1,  # In case use_log_distances=True
    # max_time=PATH_LENGTH*DT,
    max_time=None,
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
        expt_name = 'pusher_compo'
    else:
        expt_name = args.expt_name

    expt_variant['algo_params']['render'] = args.render

    expt_variant['env_params'] = env_params
    expt_variant['env_params']['is_render'] = args.render

    # TODO: MAKE THIS A SCRIPT ARGUMENT
    expt_variant['env_params']['goal'] = (0.65, 0.65)
    expt_variant['env_params']['tgt_pose'] = (0.5, 0.25, 1.4660)

    setup_logger(expt_name,
                 variant=expt_variant,
                 snapshot_mode=args.snap_mode,
                 snapshot_gap=args.snap_gap,
                 log_dir=args.log_dir)
    algo = experiment(expt_variant)

    input('Press a key to close the script...')
