"""
Run PyTorch HIU-SAC-NEW on Navigation2dGoalCompoEnv.

NOTE: You need PyTorch 0.4
"""

import os
from shutil import copyfile
import numpy as np
import torch

import robolearn.torch.utils.pytorch_util as ptu
from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.utils.launchers.launcher_util import setup_logger
from robolearn.torch.utils.data_management import MultiGoalReplayBuffer

from robolearn.envs.simple_envs.navigation2d import Navigation2dGoalCompoEnv

from robolearn.torch.algorithms.rl_algos.sac.hiu_sac \
    import HIUSAC

from robolearn.torch.models import NNQFunction
from robolearn.torch.models import NNMultiQFunction

from robolearn.torch.policies import TanhGaussianMixtureMultiPolicy

import argparse
import joblib

np.set_printoptions(suppress=True, precision=4)
# np.seterr(all='raise')  # WARNING RAISE ERROR IN NUMPY


PATH_LENGTH = 20  # time steps
PATHS_PER_EPOCH = 5
PATHS_PER_EVAL = 3
PATHS_PER_HARD_UPDATE = 35
BATCH_SIZE = 256

SEED = 110
# NP_THREADS = 6

SUBTASK = None

POLICY = TanhGaussianMixtureMultiPolicy
REPARAM_POLICY = True

SOFTMAX_WEIGHTS = True
# SOFTMAX_WEIGHTS = False
# INIT_AVG_MIXING = True
INIT_AVG_MIXING = False

OPTIMIZER = 'adam'
# OPTIMIZER = 'rmsprop'

NORMALIZE_OBS = False

expt_params = dict(
    algo_name=HIUSAC.__name__,
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
        # SAC params
        reparameterize=REPARAM_POLICY,
        action_prior='uniform',
        i_entropy_scale=1.0e-0,
        u_entropy_scale=[1.0e-0, 1.0e-0],
        auto_alphas=True,
        i_tgt_entro=4.0e-1,
        u_tgt_entros=[4.0e-1, 4.0e-1],
        # Learning rates
        optimizer=OPTIMIZER,
        i_policy_lr=1.e-4,
        u_policies_lr=1.e-4,
        u_mixing_lr=1.e-4,
        i_qf_lr=1.e-4,
        u_qf_lr=1.e-4,
        # Soft target update
        i_soft_target_tau=1.e-3,
        u_soft_target_tau=1.e-3,
        # Regularization terms
        i_policy_mean_regu_weight=1.e-3,
        i_policy_std_regu_weight=1.e-3,
        i_policy_pre_activation_weight=0.e-3,
        i_policy_mixing_coeff_weight=1.e+1,
        u_policy_mean_regu_weight=[1.e-3, 1.e-3],
        u_policy_std_regu_weight=[1.e-3, 1.e-3],
        u_policy_pre_activation_weight=[0.e-3, 0.e-3],
        # Weight decays
        i_policy_weight_decay=0.e-5,
        u_policy_weight_decay=0.e-5,
        i_q_weight_decay=0.e-5,
        u_q_weight_decay=0.e-5,

        discount=0.99,
        reward_scale=1.0e-0,
        u_reward_scales=[1.0e-0, 1.0e-0],

        normalize_obs=NORMALIZE_OBS,
    ),
    replay_buffer_size=1e3,
    net_size=32,
    softmax_weights=SOFTMAX_WEIGHTS,
    # NN Normalizations
    # -----------------
    # input_norm=True,
    input_norm=False,
    shared_layer_norm=False,
    policies_layer_norm=False,
    mixture_layer_norm=False,
    # shared_layer_norm=True,
    # policies_layer_norm=True,
    # mixture_layer_norm=True,
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
    goal_reward=0,
    actuation_cost_coeff=5.0e+0,
    distance_cost_coeff=1.0e+0,
    log_distance_cost_coeff=2.0e+0,
    alpha=1e-1,
    # Initial Condition
    init_position=(4., 4.),
    init_sigma=1.00,
    # Goal
    goal_position=(-2.0, -2.0),  # TODO: Make this a script param
    goal_threshold=0.10,
    # Others
    dynamics_sigma=0.1,
    # dynamics_sigma=0.0,
    # horizon=PATH_LENGTH,
    horizon=None,
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

    n_unintentional = 2

    if variant['load_dir']:
        params_file = os.path.join(variant['log_dir'], 'params.pkl')
        data = joblib.load(params_file)
        start_epoch = data['epoch']
        i_qf = data['qf']
        i_qf2 = data['qf2']
        u_qf = data['u_qf']
        u_qf2 = data['u_qf2']
        policy = data['policy']
        env._obs_mean = data['obs_mean']
        env._obs_var = data['obs_var']
    else:
        start_epoch = 0
        net_size = variant['net_size']

        u_qf = NNMultiQFunction(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_qs=n_unintentional,
            hidden_activation=variant['hidden_activation'],
            # shared_hidden_sizes=[net_size, net_size],
            shared_hidden_sizes=[net_size],
            # shared_hidden_sizes=[],
            unshared_hidden_sizes=[net_size, net_size],
            hidden_w_init=variant['q_hidden_w_init'],
            output_w_init=variant['q_output_w_init'],
        )
        i_qf = NNQFunction(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_activation=variant['hidden_activation'],
            hidden_sizes=[net_size, net_size],
            hidden_w_init=variant['q_hidden_w_init'],
            output_w_init=variant['q_output_w_init'],
        )
        u_qf2 = NNMultiQFunction(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_qs=n_unintentional,
            hidden_activation=variant['hidden_activation'],
            # shared_hidden_sizes=[net_size, net_size],
            shared_hidden_sizes=[net_size],
            # shared_hidden_sizes=[],
            unshared_hidden_sizes=[net_size, net_size],
            hidden_w_init=variant['q_hidden_w_init'],
            output_w_init=variant['q_output_w_init'],
        )
        i_qf2 = NNQFunction(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=[net_size, net_size],
            hidden_w_init=variant['q_hidden_w_init'],
            output_w_init=variant['q_output_w_init'],
        )

        policy = POLICY(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_policies=n_unintentional,
            hidden_activation=variant['hidden_activation'],
            # shared_hidden_sizes=[net_size, net_size],
            shared_hidden_sizes=[net_size],
            # shared_hidden_sizes=[],
            unshared_hidden_sizes=[net_size, net_size],
            unshared_mix_hidden_sizes=[net_size, net_size],
            stds=None,
            input_norm=variant['input_norm'],
            shared_layer_norm=variant['shared_layer_norm'],
            policies_layer_norm=variant['policies_layer_norm'],
            mixture_layer_norm=variant['mixture_layer_norm'],
            reparameterize=REPARAM_POLICY,
            hidden_w_init=variant['pol_hidden_w_init'],
            output_w_init=variant['pol_output_w_init'],
        )

        if INIT_AVG_MIXING:
            set_average_mixing(
                policy, n_unintentional, obs_dim,
                batch_size=50,
                total_iters=1000,
            )

    replay_buffer = MultiGoalReplayBuffer(
        max_replay_buffer_size=variant['replay_buffer_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_vector_size=n_unintentional,
    )

    # render_q = variant['render_q']
    # date_now = time.strftime("%Y_%m_%d_%H_%M_%S")
    # save_q_path = '/home/desteban/logs/goalcompo_q_plots/goalcompo_'+date_now

    algorithm = HIUSAC(
        env=env,
        policy=policy,
        u_qf1=u_qf,
        replay_buffer=replay_buffer,
        batch_size=BATCH_SIZE,
        i_qf1=i_qf,
        u_qf2=u_qf2,
        i_qf2=i_qf2,
        eval_env=env,
        save_environment=False,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()

    # algorithm.pretrain(PATH_LENGTH*2)
    algorithm.train(start_epoch=start_epoch)

    return algorithm


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
        mix_pred = policy(dummy_obs, deterministic=True)[1]['mixing_coeff']
        mix_des = torch.ones_like(mix_pred) * 1./n_unintentional
        loss = loss_fn(mix_pred, mix_des)
        mixing_optimizer.zero_grad()
        loss.backward()
        mixing_optimizer.step()
    # Set gradient to zero again
    mixing_optimizer.zero_grad()


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
    parser.add_argument('--render_q', action="store_true")
    # Algo arguments
    parser.add_argument('--mix_weight', type=float, default=1.e+1)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # Experiment name
    if args.expt_name is None:
        expt_name = 'navigation2d'
    else:
        expt_name = args.expt_name

    # Default experiment parameters
    expt_variant = expt_params

    # Default environment parameters
    expt_variant['env_params'] = env_params

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

    expt_variant['render_q'] = args.render_q

    # Algo params
    expt_variant['algo_params']['render'] = args.render
    expt_variant['algo_params']['i_policy_mixing_coeff_weight'] = args.mix_weight

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
