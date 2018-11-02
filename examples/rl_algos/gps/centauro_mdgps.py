import numpy as np
from robolearn_gym_envs.pybullet import CentauroObstacleEnv

from robolearn.rl_algos.gps.mdgps import MDGPS

from robolearn.rl_algos.gps.policies.lin_gauss_init import init_pd
from robolearn.torch.policies.gaussian_policy import GaussianPolicy

from robolearn.rl_algos.gps.costs.cost_sum import CostSum
from robolearn.rl_algos.gps.costs.cost_state import CostState
from robolearn.rl_algos.gps.costs.cost_initial_state import CostInitialState
from robolearn.rl_algos.gps.costs.cost_safe_distance import CostSafeDistance
from robolearn.rl_algos.gps.costs.cost_action import CostAction
from robolearn.rl_algos.gps.costs.cost_utils import RAMP_CONSTANT
from robolearn.rl_algos.gps.costs.cost_utils import RAMP_FINAL_ONLY

from robolearn_gym_envs.utils.transformations_utils import create_quat_pose
from robolearn.utils.launchers.launcher_util import setup_logger
import robolearn.torch.pytorch_util as ptu

import argparse

# np.set_printoptions(precision=3, suppress=True)

TEND = 4.0
SIM_TIMESTEP = 0.01
FRAME_SKIP = 1
TS = FRAME_SKIP * SIM_TIMESTEP
T = int(TEND/TS)

GPU = True
# GPU = False

SEED = 450

ptu.set_gpu_mode(GPU)

np.random.seed(SEED)
ptu.seed(SEED)

noise_hyperparams = dict(
    smooth_noise=True,  # Apply Gaussian filter to noise generated
    smooth_noise_var=2.0e+0,  # np.power(2*Ts, 2), # Variance to apply to Gaussian Filter. In Kumar (2016) paper, it is the std dev of 2 Ts
    smooth_noise_renormalize=True,  # Renormalize smooth noise to have variance=1
    noise_var_scale=1.e-5*np.array([1., 1., 1., 1., .1, 0.1, 0.1]),  # Scale to Gaussian noise: N(0, 1)*sqrt(noise_var_scale), only if smooth_noise_renormalize
)

algo_params = dict(
    seed=SEED,
    nepochs=100,
    num_samples=3,
    test_samples=1,
    noisy_samples=True,
    # noisy_samples=False,
    train_conds=[0, 1, 2],
    test_conds=[0, 1, 2],
    base_kl_step=0.05,  # 0.01,
    # base_kl_step=5000000.0,
    global_opt_iters=5000,
    # global_opt_iters=50,
    global_opt_batch_size=128,
    global_opt_lr=1e-2,
    # TRAJ OPT
    # --------
    traj_opt_prev='nn_pol',
    # traj_opt_prev='traj',
    traj_opt_iters=1,
    traj_opt_min_eta=1e-8,
    traj_opt_max_eta=1e3,  # 1e16
)

env_params = dict(
    is_render=True,
    # is_render=False,
    obs_with_img=False,
    active_joints='RA',
    control_type='tasktorque',
    # _control_type='velocity',
    sim_timestep=SIM_TIMESTEP,
    frame_skip=FRAME_SKIP,
    obs_distances=True,
    goal_tolerance=0.02,
    max_time=None,
)

policy_params = dict(
    global_pol_params=dict(
        hidden_sizes=(128, 128),
        hidden_activation='relu',
        # output_w_init='xavier_normal',
        output_w_init='xavier_normal_0.01',
        output_b_init_val=0.0,
    ),
    local_pol_params=dict(
        max_var=0.01,
    )
)

# Fixed initial conditions  [tgtX, tgtY, tgtZ, tgtROLL, obstX, obstY]
initial_conds = [
    [1.12, -0.2, 0.99, 0.1, 1.08, -0.35],
    [1.13, -0.2, 0.99, 0.1, 1.08, -0.35],
    [1.125, -0.2, 0.95, 0.1, 1.07, -0.35],
    [1.12, -0.2, 0.94, 0.1, 1.08, -0.35],
    [1.13, -0.2, 1.0, 0.1, 1.08, -0.35],
]

cost_params = dict(
    # COSTS:
    # 0: action,
    # 1: target_state
    # 2: target_state_final
    # 3: safe_distance
    # 4: safe_distance_final
    # 5: safe_distance_final
    # 6: velocity
    costs_to_consider=[0, 1, 2, 6],
    cost_weights=[1.0e-1, 1.0e+0, 1.0e-2, 0.0e+0, 0.0e+0, 1.0e-4, 1.0e+0],
)

expt_params = dict(
    # General Algo
    algo_params=algo_params,
    cost_params=cost_params,
    env_params=env_params,
    policy_params=policy_params,
    initial_conditions=initial_conds,
)


def create_environment():
    print("\nCreating Environment...")

    env = CentauroObstacleEnv(**env_params)

    # env.set_tgt_cost_weights(tgt_weights)
    # env.set_tgt_pos(tgt_positions)

    print("Environment:%s OK!." % type(env).__name__)

    obst_height = 0.851

    for init_cond in initial_conds:
        robot_config = env._init_robot_config
        tgt_state = create_quat_pose(pos_x=init_cond[0],
                                     pos_y=init_cond[1],
                                     pos_z=init_cond[2],
                                     rot_roll=init_cond[3])

        obst_state = create_quat_pose(pos_x=init_cond[4],
                                      pos_y=init_cond[5],
                                      pos_z=obst_height,
                                      rot_yaw=0.)

        env.add_initial_condition(robot_config=robot_config,
                                  tgt_state=tgt_state,
                                  obst_state=obst_state)

    print("Conditions for Environment:%s OK!." % type(env).__name__)

    return env


def create_cost_fcn(env):

    # ########### #
    # ACTION COST #
    # ########### #
    cost_action = CostAction(
        wu=np.ones(env.action_dim),
        target=None,
    )

    # ################# #
    # TARGET STATE COST #
    # ################# #
    # TODO: ASSUMING ENV IS OBS_DISTANCES

    state_idxs = [
        # env.get_obs_info(name='position')['idx'],  # MOVE ARM
        env.get_obs_info(name='target')['idx'],  # Move to target
    ]
    target_states = [
        # [-0., 0.2, 0.8, 0.8, 0., 0.8, -0.],  # Moving only 2nd joint
        [0., 0., 0., 0., 0., 0.],  # Moving to target
    ]
    wps = [
        # np.array([1., 1., 1., 1., 1., 1., 1.]),  # MOVE ARM
        np.array([1.3, 2., 1., 0.1, 0.1, 0.1]),
    ]
    cost_target_state = CostState(
        ramp_option=RAMP_CONSTANT,
        state_idxs=state_idxs,
        target_states=target_states,
        wps=wps,
        wp_final_multiplier=1.0e0,
        cost_type='logl2',
        l1_weight=0.,
        l2_weight=1.,
        alpha=1.e-2

    )
    cost_target_state_final = CostState(
        ramp_option=RAMP_FINAL_ONLY,
        state_idxs=state_idxs,
        target_states=target_states,
        wps=wps,
        wp_final_multiplier=1.0e0,
        cost_type='logl2',
        l1_weight=0.,
        l2_weight=1.,
        alpha=1.e-2

    )


    # ################## #
    # SAFE DISTANCE COST #
    # ################## #

    safe_x = 0.15
    safe_y = 0.15
    safe_z = 0.25
    state_idxs = [
        env.get_obs_info(name='obstacle')['idx'][:3],
    ]
    safe_distances = [
        [safe_x, safe_y, safe_z],
    ]
    wps = [
        np.array([1., 1., 1.]),
    ]
    inside_costs = [
        [1., 1., 1.],
    ]
    outside_costs = [
        [0., 1., 0.],
    ]

    cost_safe_distance = CostSafeDistance(
        ramp_option=RAMP_CONSTANT,
        state_idxs=state_idxs,
        safe_distances=safe_distances,
        wps=wps,
        inside_costs=inside_costs,
        outside_costs=outside_costs,
        wp_final_multiplier=1.0,
    )

    cost_safe_distance_final = CostSafeDistance(
        ramp_option=RAMP_FINAL_ONLY,
        state_idxs=state_idxs,
        safe_distances=safe_distances,
        wps=wps,
        inside_costs=inside_costs,
        outside_costs=outside_costs,
        wp_final_multiplier=1.0,
    )

    # ######### #
    # POSE COST #
    # ######### #
    state_idxs = [
        env.get_obs_info(name='position')['idx'],  # MOVE ARM
    ]
    target_states = [
        [-0., 0.0, 0.0, 0.0, 0., 0.0, -0.],  # Moving only 2nd joint
    ]
    wps = [
        np.array([1., 1., 1., 1., 1., 1., 1.]),  # MOVE ARM
    ]
    cost_initial_state = CostInitialState(
        ramp_option=RAMP_CONSTANT,
        state_idxs=state_idxs,
        target_states=target_states,
        wps=wps,
        wp_final_multiplier=1.0e0,
        cost_type='logl2',
        l1_weight=0.,
        l2_weight=1.,
        alpha=1.e-2

    )

    # ######### #
    # POSE VELOCITY #
    # ######### #
    state_idxs = [
        env.get_obs_info(name='velocity')['idx'],  # MOVE ARM
    ]
    target_states = [
        [0., 0.0, 0.0, 0.0, 0., 0.0, 0.],
    ]
    wps = [
        np.array([1., 1., 1., 1., 1., 1., 1.]),  # MOVE ARM
    ]
    cost_velocity = CostState(
        ramp_option=RAMP_CONSTANT,
        state_idxs=state_idxs,
        target_states=target_states,
        wps=wps,
        wp_final_multiplier=1.0e0,
        cost_type='logl2',
        l1_weight=0.,
        l2_weight=1.,
        alpha=1.e-2

    )


    # ######### #
    #  COST SUM #
    # ######### #

    all_costs = [
        cost_action,
        cost_target_state,
        cost_target_state_final,
        cost_safe_distance,
        cost_safe_distance_final,
        cost_initial_state,
        cost_velocity,
    ]

    cost_idxs = cost_params['costs_to_consider']
    weights = cost_params['cost_weights']

    cost_sum = CostSum(costs=[all_costs[idx] for idx in cost_idxs],
                       weights=[weights[idx] for idx in cost_idxs])

    return cost_sum


def create_policies(env):
    train_conds = expt_params['algo_params']['train_conds']
    local_policies = list()
    for cc in train_conds:
        tvlgc_pol = init_pd(dU=env.action_dim,
                            dX=env.obs_dim,
                            T=T,
                            x0=env.initial_obs_conditions[cc],
                            state_idx=env.get_obs_info(name='position')['idx'],
                            dstate_idx=None,
                            pos_gains=0.00005,  # For dU
                            Kp=1,
                            Kv=0.00001,
                            init_var=np.array(policy_params['local_pol_params']['max_var']),
                            )

        local_policies.append(tvlgc_pol)

    global_policy = GaussianPolicy(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_sizes=policy_params['global_pol_params']['hidden_sizes'],
        hidden_activation=policy_params['global_pol_params']['hidden_activation'],
        output_w_init=policy_params['global_pol_params']['output_w_init'],
        output_b_init_val=policy_params['global_pol_params']['output_b_init_val'],
    )

    return local_policies, global_policy


def create_algo(env, local_policies, global_policy, cost_fcn):
    train_conds = algo_params['train_conds']
    test_conds = algo_params['test_conds']
    noisy_samples = algo_params['noisy_samples']
    num_samples = expt_params['algo_params']['num_samples']
    test_samples = expt_params['algo_params']['test_samples']
    seed = expt_params['algo_params']['seed']


    algo = MDGPS(
        env=env,
        local_policies=local_policies,
        global_policy=global_policy,
        cost_fcn=cost_fcn,
        eval_env=env,
        num_epochs=algo_params['nepochs'],
        num_steps_per_epoch=int(len(local_policies)*len(train_conds)*num_samples*T),
        num_steps_per_eval=int(T*len(test_conds)),
        max_path_length=T,
        train_cond_idxs=algo_params['train_conds'],
        test_cond_idxs=algo_params['test_conds'],
        num_samples=num_samples,
        test_samples=test_samples,
        noisy_samples=noisy_samples,
        noise_hyperparams=noise_hyperparams,
        seed=seed,
        base_kl_step=algo_params['base_kl_step'],
        global_opt_iters=algo_params['global_opt_iters'],
        global_opt_batch_size=algo_params['global_opt_batch_size'],
        global_opt_lr=algo_params['global_opt_lr'],
        traj_opt_prev=algo_params['traj_opt_prev'],
        traj_opt_iters=algo_params['traj_opt_iters'],
        traj_opt_min_eta=algo_params['traj_opt_min_eta'],
        traj_opt_max_eta=algo_params['traj_opt_max_eta'],
        save_algorithm=False,
        save_environment=False,
    )

    return algo


# ####################
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

args = parse_args()

# Experiment name
if args.expt_name is None:
    expt_name = 'centauro_mdgps'
else:
    expt_name = args.expt_name

log_dir = setup_logger(expt_name,
                       variant=expt_params,
                       snapshot_mode=args.snap_mode,
                       snapshot_gap=args.snap_gap,
                       log_dir=args.log_dir)

print('***********************\n'*5)
print("Logging in directory: %s" % log_dir)
print('***********************\n'*5)


env = create_environment()
#
# for cc in range(env.n_init_conds):
#     env.reset(condition=cc)
#     print(cc)
#     input('wuuu')

cost_fcn = create_cost_fcn(env)

local_policies, global_policy = create_policies(env)

mdgps_algo = create_algo(env, local_policies, global_policy, cost_fcn)

# if ptu.gpu_enabled():
#     mdgps_algo.cuda()
if ptu.gpu_enabled():
    global_policy.cuda()

start_epoch = 0
mdgps_algo.train(start_epoch=start_epoch)

# action_dim = env.action_dim
# obs_dim = env.obs_dim
# state_dim = env.state_dim
#
# print(action_dim, obs_dim, state_dim)
#
# fake_sample = dict(
#     actions=np.random.rand(10, action_dim),
#     observations=np.random.rand(10, obs_dim)
# )
# fake_sample['actions'][-1] = 0
# fake_sample['observations'][-1, env.get_obs_info(name='position')['idx']] = \
#     [-0., 0.2, 0.8, 0.8, 0., 0.8, -0.]
#
# l, lx, lu, lxx, luu, lux, cost_composition = cost_fcn.eval(fake_sample)
# print('l', l)
# for ii, cost_compo in enumerate(cost_composition):
#     print('cost_compo', ii, ': ', cost_compo)

# input('MDGPS IS OVER')
