"""
Run PyTorch IU Multi Soft Actor Critic on GoalCompositionEnv.

NOTE: You need PyTorch 0.4
"""

import numpy as np

import robolearn.torch.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.utils.data_management import MultiGoalReplayBuffer

from robolearn.envs.simple_envs.goal_composition import GoalCompositionEnv
from robolearn.envs.simple_envs.goal_composition import MultiQFPolicyPlotter

from robolearn.torch.rl_algos.sac.iu_weightedmultisac import IUWeightedMultiSAC

from robolearn.torch.models import NNQFunction, NNVFunction
from robolearn.torch.models import NNMultiQFunction, NNMultiVFunction

from robolearn.torch.policies import TanhGaussianMultiPolicy
from robolearn.torch.policies import MixtureTanhGaussianMultiPolicy
<<<<<<< HEAD
from robolearn.torch.policies import TanhGaussianWeightedMultiPolicy

=======
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d
# from robolearn.torch.sac.policies import WeightedTanhGaussianMultiPolicy
# from robolearn.torch.sac.policies import MultiPolicySelector
# from robolearn.torch.sac.policies import BernoulliTanhGaussianMultiPolicy

import argparse
import time

<<<<<<< HEAD
PATH_LENGTH = 50  # time steps
PATHS_PER_EPOCH = 3
PATHS_PER_EVAL = 3
PATHS_PER_HARD_UPDATE = 35
BATCH_SIZE = 256

SEED = 110

# POLICY = TanhGaussianWeightedMultiPolicy
POLICY = MixtureTanhGaussianMultiPolicy

=======
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d

def experiment(variant):
    render_q = variant['render_q']
    date_now = time.strftime("%Y_%m_%d_%H_%M_%S")
    save_q_path = '/home/desteban/logs/goalcompo_q_plots/goalcompo_'+date_now

    ptu.set_gpu_mode(variant['gpu'])
<<<<<<< HEAD
    ptu.seed(SEED)

    env = NormalizedBoxEnv(
        GoalCompositionEnv(**variant['env_params']),
        # normalize_obs=True,
        normalize_obs=False,
        online_normalization=False,
        obs_mean=None,
        obs_var=None,
        obs_alpha=0.001,
=======

    env = NormalizedBoxEnv(
        GoalCompositionEnv(**variant['env_params'])
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d
    )

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    n_unintentional = 2

<<<<<<< HEAD
    if variant['log_dir']:
        raise ValueError
    else:
        start_epoch = 0
        net_size = variant['net_size']
        u_qf = NNMultiQFunction(obs_dim=obs_dim,
                                action_dim=action_dim,
                                n_qs=n_unintentional,
                                shared_hidden_sizes=[net_size, net_size],
                                # shared_hidden_sizes=[],
                                unshared_hidden_sizes=[net_size, net_size, net_size])
        u_qf2 = NNMultiQFunction(obs_dim=obs_dim,
                                 action_dim=action_dim,
                                 n_qs=n_unintentional,
                                 shared_hidden_sizes=[net_size, net_size],
                                 # shared_hidden_sizes=[],
                                 unshared_hidden_sizes=[net_size, net_size, net_size])
        # i_qf = WeightedNNMultiVFunction(u_qf)
        i_qf = NNQFunction(obs_dim=obs_dim,
                           action_dim=action_dim,
                           hidden_sizes=[net_size, net_size])
        i_qf2 = NNQFunction(obs_dim=obs_dim,
                            action_dim=action_dim,
                            hidden_sizes=[net_size, net_size])

        u_vf = NNMultiVFunction(obs_dim=obs_dim,
                                n_vs=n_unintentional,
                                shared_hidden_sizes=[net_size, net_size],
                                # shared_hidden_sizes=[],
                                unshared_hidden_sizes=[net_size, net_size, net_size])
        i_vf = NNVFunction(obs_dim=obs_dim,
                           hidden_sizes=[net_size, net_size])
        # i_vf = WeightedNNMultiVFunction(u_vf)

        u_policy = TanhGaussianMultiPolicy(obs_dim=obs_dim,
                                           action_dim=action_dim,
                                           n_policies=n_unintentional,
                                           shared_hidden_sizes=[net_size, net_size],
                                           unshared_hidden_sizes=[net_size, net_size])
        policy = POLICY(u_policy,
                        mix_hidden_sizes=[net_size, net_size],
                        pol_idxs=None,
                        optimize_multipolicy=False)

=======
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
                                       unshared_hidden_sizes=[net_size, net_size])
    i_policy = MixtureTanhGaussianMultiPolicy(u_policy,
                                              mix_hidden_sizes=[net_size, net_size],
                                              pol_idxs=None,
                                              optimize_multipolicy=False)
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d
    # i_policy = MultiPolicySelector(u_policy, 0)
    # i_policy = BernoulliTanhGaussianMultiPolicy(u_policy, prob=0.5)

    replay_buffer = MultiGoalReplayBuffer(
<<<<<<< HEAD
        max_replay_buffer_size=variant['replay_buffer_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_vector_size=n_unintentional,
    )
=======
        variant['algo_params']['replay_buffer_size'],
        np.prod(env.observation_space.shape),
        np.prod(env.action_space.shape),
        n_unintentional
    )
    variant['algo_params']['replay_buffer'] = replay_buffer
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d

    # QF Plot
    goal_pos = expt_variant['env_params']['goal_position']
    init_pos = expt_variant['env_params']['init_position']
    q_fcn_positions = [
        (goal_pos[0], init_pos[1]),
        (init_pos[0], init_pos[1]),
        (init_pos[0], goal_pos[1]),
        (goal_pos[0], goal_pos[1]),
        (6.0, goal_pos[1])
    ]
    plotter = MultiQFPolicyPlotter(
        i_qf=i_qf,
<<<<<<< HEAD
        i_policy=policy,
=======
        i_policy=i_policy,
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d
        u_qf=u_qf,
        u_policy=u_policy,
        obs_lst=q_fcn_positions,
        default_action=[np.nan, np.nan],
        n_samples=100,
        render=render_q,
        save_path=save_q_path,
    )
    variant['algo_params']['epoch_plotter'] = plotter
    # variant['algo_params']['epoch_plotter'] = None

    algorithm = IUWeightedMultiSAC(
        env=env,
<<<<<<< HEAD
        policy=policy,
        u_qf=u_qf,
        u_vf=u_vf,
        replay_buffer=replay_buffer,
        batch_size=BATCH_SIZE,  # batch_size
        i_qf=i_qf,
        i_vf=i_vf,
        u_qf2=u_qf2,
        i_qf2=i_qf2,
        eval_env=env,
        save_environment=False,
=======
        training_env=env,
        save_environment=False,
        u_policy=u_policy,
        u_qf=u_qf,
        u_vf=u_vf,
        i_policy=i_policy,
        i_qf=i_qf,
        i_vf=i_vf,
        algo_interface='torch',
        min_buffer_size=variant['algo_params']['batch_size'],
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
<<<<<<< HEAD
    # algorithm.pretrain(PATH_LENGTH*2)
    algorithm.train(start_epoch=start_epoch)
=======
    algorithm.train()
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d

    return algorithm


<<<<<<< HEAD
expt_params = dict(
    # algo_name=IUMultiSAC.__name__,
    algo_name=IUWeightedMultiSAC.__name__,
    policy_name=POLICY.__name__,
    algo_params=dict(
        # Common RL algorithm params
=======
PATH_LENGTH = 50  # time steps
PATHS_PER_EPOCH = 3
PATHS_PER_EVAL = 3
PATHS_PER_HARD_UPDATE = 35

expt_params = dict(
    algo_name=IUMultiSAC.__name__,
    algo_params=dict(
        # Common RLAlgo params
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d
        num_steps_per_epoch=PATHS_PER_EPOCH * PATH_LENGTH,
        num_epochs=1000,  # n_epochs
        num_updates_per_train_call=1,  # How to many run algorithm train fcn
        num_steps_per_eval=PATHS_PER_EVAL * PATH_LENGTH,
        # EnvSampler params
        max_path_length=PATH_LENGTH,  # max_path_length
        render=False,
<<<<<<< HEAD
        # SoftActorCritic params
        min_steps_start_train=BATCH_SIZE,  # Min nsteps to start to train (or batch_size)
        min_start_eval=PATHS_PER_EPOCH * PATH_LENGTH,  # Min nsteps to start to eval
        reparameterize=True,
        action_prior='uniform',
        i_entropy_scale=1.0e-0,
        u_entropy_scale=[1.0e-0, 1.0e-0],

        i_policy_lr=1e-3,
        u_policies_lr=1.e-4,

        discount=0.99,
=======
        # ReplayBuffer params
        batch_size=64,  # batch_size
        replay_buffer_size=1e4,
        # SoftActorCritic params
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
        # discount=0.1,
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d
        # reward_scale=0.08,
        # reward_scale=0.10,  # No funciona 10/06
        reward_scale=1.00,
        # reward_scale=0.2,  # Mixture no funciona. 11/06
        # reward_scale=0.1,  # Mixture funciona con este 10/06
        # reward_scale=0.15,  # Mixture ... con este 11/06
        # reward_scale=0.5,  # Mixture ... con este 10/06
        # reward_scale=10.0,  # Mixture ... con este 10/06
        # reward_scale=1000.0,  # Mixture ... con este 10/06
    ),
    net_size=64,
<<<<<<< HEAD
    replay_buffer_size=1e4,
)

env_params = dict(
    # Costs
=======
)

env_params = dict(
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d
    goal_reward=5,
    actuation_cost_coeff=0.5,
    distance_cost_coeff=0.0,
    log_distance_cost_coeff=1.5,
    alpha=1e-6,
<<<<<<< HEAD
    # Initial Condition
    init_position=(-4., -4.),
    init_sigma=1.50,
    # Goal
    goal_position=None,
    goal_threshold=0.05,
    # Others
    dynamics_sigma=0.1,
=======
    init_position=(-4., -4.),
    init_sigma=1.50,
    goal_position=None,
    dynamics_sigma=0.1,
    goal_threshold=0.05,
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d
    # horizon=PATH_LENGTH,
    horizon=None,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_size', type=int, default=None)
    parser.add_argument('--expt_name', type=str, default=None)
    # parser.add_argument('--expt_name', type=str, default=timestamp())
    # Logging arguments
    parser.add_argument('--snap_mode', type=str, default='gap_and_last')
<<<<<<< HEAD
    parser.add_argument('--snap_gap', type=int, default=25)
    # parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    # GPU arguments
    parser.add_argument('--gpu', action="store_true")
    # Other arguments
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--render_q', action="store_true")
=======
    parser.add_argument('--snap_gap', type=int, default=50)
    # parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--render_q', action="store_true")
    parser.add_argument('--gpu', action="store_true")
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d
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
        expt_name = 'goalcompo'
    else:
        expt_name = args.expt_name

    expt_variant['algo_params']['render'] = args.render

    expt_variant['env_params'] = env_params
<<<<<<< HEAD
=======

>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d
    # TODO: MAKE THIS A SCRIPT ARGUMENT
    expt_variant['env_params']['goal_position'] = (3.5, 4.5)
    # expt_variant['env_params']['goal_position'] = (7., 7.)

<<<<<<< HEAD
    expt_variant['log_dir'] = args.log_dir

=======
>>>>>>> 359a84d1aeac5042dc64e73d031ac5d4ea688a4d
    setup_logger(expt_name,
                 variant=expt_variant,
                 snapshot_mode=args.snap_mode,
                 snapshot_gap=args.snap_gap,
                 log_dir=args.log_dir)
    algo = experiment(expt_variant)

    input('Press a key to close the script...')
