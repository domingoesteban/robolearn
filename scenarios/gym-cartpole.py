from __future__ import print_function

import sys
import os
import signal
import numpy as np
import random
import matplotlib.pyplot as plt

from robolearn.utils.iit.iit_robots_params import bigman_params
from robolearn.envs import GymEnv
from robolearn.agents import GPSAgent

from robolearn.policies.policy_opt.policy_opt_tf import PolicyOptTf
from robolearn.policies.policy_opt.tf_models import tf_network

from robolearn.utils.sample import Sample
from robolearn.utils.sample_list import SampleList

from robolearn.costs.cost_action import CostAction
from robolearn.costs.cost_state import CostState
from robolearn.costs.cost_fk_relative import CostFKRelative
from robolearn.costs.cost_fk import CostFK
from robolearn.costs.cost_fk_target import CostFKTarget
from robolearn.costs.cost_sum import CostSum
from robolearn.costs.cost_utils import RAMP_QUADRATIC, RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_CONSTANT
from robolearn.costs.cost_utils import evall1l2term, evallogl2term

from robolearn.utils.traj_opt.traj_opt_pi2 import TrajOptPI2
from robolearn.utils.traj_opt.traj_opt_lqr import TrajOptLQR
from robolearn.utils.traj_opt.traj_opt_dreps import TrajOptDREPS
from robolearn.utils.traj_opt.traj_opt_mdreps import TrajOptMDREPS
from robolearn.utils.dynamics.dynamics_lr_prior import DynamicsLRPrior
from robolearn.utils.dynamics.dynamics_prior_gmm import DynamicsPriorGMM

from robolearn.algos.gps.mdgps import MDGPS
from robolearn.algos.gps.pigps import PIGPS
from robolearn.algos.trajopt.ilqr import ILQR
from robolearn.algos.trajopt.pi2 import PI2
from robolearn.algos.trajopt.dreps import DREPS
from robolearn.algos.trajopt.mdreps import MDREPS
from robolearn.policies.lin_gauss_init import init_lqr, init_pd, init_demos
from robolearn.policies.policy_prior import ConstantPolicyPrior  # For MDGPS

from robolearn.utils.sampler import Sampler
from robolearn.policies.traj_reprod_policy import TrajectoryReproducerPolicy
from robolearn.utils.joint_space_control_sampler import JointSpaceControlSampler
from robolearn.policies.computed_torque_policy import ComputedTorquePolicy

from robolearn.utils.reach_drill_utils import create_drill_relative_pose
from robolearn.utils.reach_drill_utils import reset_condition_bigman_drill_gazebo, Reset_condition_bigman_drill_gazebo
from robolearn.utils.reach_drill_utils import spawn_drill_gazebo
from robolearn.utils.reach_drill_utils import set_drill_gazebo_pose
from robolearn.utils.reach_drill_utils import create_bigman_drill_condition
from robolearn.utils.reach_drill_utils import create_hand_relative_pose
from robolearn.utils.reach_drill_utils import generate_reach_joints_trajectories
from robolearn.utils.reach_drill_utils import generate_lift_joints_trajectories
from robolearn.utils.reach_drill_utils import task_space_torque_control_demos, load_task_space_torque_control_demos
from robolearn.utils.reach_drill_utils import task_space_torque_control_dual_demos, load_task_space_torque_control_dual_demos

from robolearn.utils.robot_model import RobotModel
from robolearn.utils.transformations import create_quat_pose
from robolearn.utils.algos_utils import IterationData
from robolearn.utils.algos_utils import TrajectoryInfo
from robolearn.utils.print_utils import change_print_color
from robolearn.utils.plot_utils import plot_joint_info

import time
import datetime

np.set_printoptions(precision=4, suppress=True, linewidth=1000)


def kill_everything(_signal=None, _frame=None):
    print("\n\033[1;31mThe script has been kill by the user!!")
    os._exit(1)

signal.signal(signal.SIGINT, kill_everything)


# ################## #
# ################## #
# ### PARAMETERS ### #
# ################## #
# ################## #
learning_algorithm = 'MDREPS'
env_name = 'Pendulum-v0'
# Task parameters
Ts = 0.01
Treach = 5
Tlift = 0  # 3.8
Tinter = 0  # 0.5
Tend = 0  # 0.7
# EndTime = 4  # Using final time to define the horizon
EndTime = Treach + Tinter + Tlift + Tend  # Using final time to define the horizon
init_with_demos = False
generate_dual_sets = True
demos_dir = None  # 'TASKSPACE_TORQUE_CTRL_DEMO_2017-07-21_16:32:39'
dual_dir = 'DUAL_DEMOS_2017-08-23_07:10:35'
seed = 6

random.seed(seed)
np.random.seed(seed)

# ################### #
# ################### #
# ### ENVIRONMENT ### #
# ################### #
# ################### #
change_print_color.change('BLUE')
print("\nCreating GYM environment...")

# reset_condition_bigman_drill_gazebo_fcn = Reset_condition_bigman_drill_gazebo()


# Create a BIGMAN ROS EnvInterface
gym_env = GymEnv(name=env_name, render=True, seed=seed)

action_dim = gym_env.get_action_dim()
state_dim = gym_env.get_state_dim()
observation_dim = gym_env.get_obs_dim()

print("Gym Environment OK. name:%s (action_dim=%d), (state_dim=%d)" % (env_name, action_dim, state_dim))

# ################# #
# ################# #
# ##### AGENT ##### #
# ################# #
# ################# #
change_print_color.change('CYAN')
print("\nCreating Gym Agent...")

policy_params = {
    'network_model': tf_network,  # tf_network, multi_modal_network, multi_modal_network_fp
    'network_params': {
        'n_layers': 1,  # Hidden layers??
        'dim_hidden': [40],  # List of size per n_layers
        'obs_names': gym_env.get_obs_info()['names'],
        'obs_dof': gym_env.get_obs_info()['dimensions'],  # DoF for observation data tensor
    },
    # Initialization.
    'init_var': 0.1,  # Initial policy variance.
    'ent_reg': 0.0,  # Entropy regularizer (Used to update policy variance)
    # Solver hyperparameters.
    'iterations': 5000,  # Number of iterations per inner iteration (Default:5000). Recommended: 1000?
    'batch_size': 15,
    'lr': 0.001,  # Base learning rate (by default it's fixed).
    'lr_policy': 'fixed',  # Learning rate policy.
    'momentum': 0.9,  # Momentum.
    'weight_decay': 0.005,  # Weight decay.
    'solver_type': 'Adam',  # Solver type (e.g. 'SGD', 'Adam', etc.).
    # set gpu usage.
    'gpu_mem_percentage': 0.05,
    'use_gpu': 1,  # Whether or not to use the GPU for training.
    'gpu_id': 0,
    'random_seed': 1,
    'fc_only_iterations': 0,  # TODO: Only forwardcontrol? if it is CNN??
    # 'weights_file_prefix': EXP_DIR + 'policy',
}
policy_opt = {
    'type': PolicyOptTf,
    'hyperparams': policy_params
    }

gym_agent = GPSAgent(act_dim=action_dim, obs_dim=observation_dim, state_dim=state_dim, policy_opt=policy_opt)
print("Bigman Agent:%s OK\n" % type(gym_agent))


# ################# #
# ################# #
# ##### COSTS ##### #
# ################# #
# ################# #
# Action Cost
act_cost = {
    'type': CostAction,
    'wu': np.ones(action_dim) * 1e-4,
    'target': None,   # Target action value
}

# State Cost
target_distance_state = np.zeros(6)
state_cost_distance = {
    'type': CostState,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'l1': 0.1,  # Weight for l1 norm
    'l2': 1.0,  # Weight for l2 norm
    'alpha': 1e-2,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 10.0,  # Weight multiplier on final time step.
    'data_types': {
        'distance_right_arm': {
            # 'wp': np.ones_like(target_state),  # State weights - must be set.
            'wp': np.array([1.0, 1.0, 1.0, 3.0, 3.0, 1.0]),  # State weights - must be set.
            'target_state': target_distance_state,  # Target state - must be set.
            'average': None,  # (12, 3),
            'data_idx': gym_env.get_state_info(name='gym_state')['idx']
        },
    },
}

cost_sum = {
    'type': CostSum,
    'costs': [act_cost, state_cost_distance],
    'weights': [1.0e-1, 1.0e-0],
}


# ########## #
# ########## #
# Conditions #
# ########## #
# ########## #
drill_relative_poses = []  # Used only in dual demos
condition0 = 2  # Seed number
gym_env.add_condition(condition0)

# #################### #
# #################### #
# ## DEMONSTRATIONS ## #
# #################### #
# #################### #
demos_samples = None

# DUAL SAMPLES
good_trajs = None
bad_trajs = None


# ######################## #
# ######################## #
# ## LEARNING ALGORITHM ## #
# ######################## #
# ######################## #
change_print_color.change('YELLOW')
print("\nConfiguring learning algorithm...\n")

# Learning params
resume_training_itr = None  # Resume from previous training iteration
# data_files_dir = 'GPS_2017-09-01_15:22:55'  # None  # In case we want to resume from previous training
data_files_dir = None  # In case we want to resume from previous training

if demos_samples is None:
    #      # init_traj_distr values can be lists if they are different for each condition
    #      init_traj_distr = {'type': init_lqr,
    #                         # Parameters to calculate initial COST function based on stiffness
    #                         'init_var': 3.0e-1,  # Initial Variance
    #                         'stiffness': 5.0e-1,  # Stiffness (multiplies q)
    #                         'stiffness_vel': 0.01,  # 0.5,  # Stiffness_vel*stiffness (multiplies qdot)
    #                         'final_weight': 10.0,  # Multiplies cost at T
    #                         # Parameters for guessing dynamics
    #                         'init_acc': np.zeros(action_dim),  # dU vector(np.array) of accelerations, default zeros.
    #                         #'init_gains': 1.0*np.ones(action_dim),  # dU vector(np.array) of gains, default ones.
    #                         #'init_gains': 1.0/np.array([5000.0, 8000.0, 5000.0, 5000.0, 300.0, 2000.0, 300.0]),  # dU vector(np.array) of gains, default ones.
    #                         'init_gains': np.ones(action_dim),  # dU vector(np.array) of gains, default ones.
    #                         }
    init_traj_distr = {'type': init_pd,
                       'init_var': np.array([3.0e-1])*1.0e-00,
                       'pos_gains': 0.001,  #np.array([1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 5.0e-2, 5.0e-2, 5.0e-2])*1.0e+1,  # 0.001,  # Position gains (Default:10)
                       'vel_gains_mult': 0.01,  # Velocity gains multiplier on pos_gains
                       'init_action_offset': None,
                       'dJoints': 1,  # Total joints in state
                       'state_to_pd': 'joints',  # Joints
                       'dDistance': None,
                       }
else:
    init_traj_distr = {'type': init_demos,
                       'sample_lists': demos_samples
                       }

# Trajectory Optimization Options
traj_opt_lqr = {'type': TrajOptLQR,
                'del0': 1e-4,  # Dual variable updates for non-SPD Q-function (non-SPD correction step).
                # 'eta_error_threshold': 1e16, # TODO: REMOVE, it is not used
                'min_eta': 1e-8,  # At min_eta, kl_div > kl_step
                'max_eta': 1e16,  # At max_eta, kl_div < kl_step
                'cons_per_step': False,  # Whether or not to enforce separate KL constraints at each time step.
                'use_prev_distr': False,  # Whether or not to measure expected KL under the previous traj distr.
                'update_in_bwd_pass': True,  # Whether or not to update the TVLG controller during the bwd pass.
                }

traj_opt_pi2 = {'type': TrajOptPI2,
                'del0': 1e-4,  # Dual variable updates for non-PD Q-function.
                'kl_threshold': 1.0,   # KL-divergence threshold between old and new policies.
                'covariance_damping': 10.0,  # 2.0,  # If greater than zero, covariance is computed as a multiple of the old
                # covariance. Multiplier is taken to the power (1 / covariance_damping).
                # If greater than one, slows down convergence and keeps exploration noise high for more iterations.
                'min_temperature': 0.001,  # Minimum bound of the temperature optimization for the soft-max
                # probabilities of the policy samples.
                'use_sumexp': False,
                'pi2_use_dgd_eta': True,  # False,
                'pi2_cons_per_step': True,
                }

traj_opt_dreps = {'type': TrajOptDREPS,
                  'epsilon': 1.0,   # KL-divergence threshold between old and new policies.
                  'xi': 5.0,
                  'chi': 2.0,
                  'dreps_cons_per_step': True,
                  'min_eta': 0.001,  # Minimum bound of the temperature optimization for the soft-max
                  'covariance_damping': 2.0,
                  'del0': 1e-4,  # Dual variable updates for non-SPD Q-function (non-SPD correction step).
                  }

traj_opt_mdreps = {'type': TrajOptMDREPS,
                   'del0': 1e-4,  # Eta updates for non-SPD Q-function (non-SPD correction step).
                   'del0_good': 1e-4,  # Omega updates for non-SPD Q-function (non-SPD correction step).
                   'del0_bad': 1e-8,  # Nu updates for non-SPD Q-function (non-SPD correction step).
                   # 'eta_error_threshold': 1e16, # TODO: REMOVE, it is not used
                   'min_eta': 1e-8,  # At min_eta, kl_div > kl_step
                   'max_eta': 1e16,  # At max_eta, kl_div < kl_step
                   'min_omega': 0,#1e-8,  # At min_omega, kl_div > kl_step
                   'max_omega': 0,#1e16,  # At max_omega, kl_div < kl_step
                   'min_nu': 0,#1e-8,  # At min_nu, kl_div > kl_step
                   'max_nu': 0,#2.0e1,  # At max_nu, kl_div < kl_step,
                   'step_tol': 0.1,
                   'bad_tol': 0.2,
                   'good_tol': 0.3,
                   'cons_per_step': False,  # Whether or not to enforce separate KL constraints at each time step.
                   'use_prev_distr': False,  # Whether or not to measure expected KL under the previous traj distr.
                   'update_in_bwd_pass': True,  # Whether or not to update the TVLG controller during the bwd pass.
                   }

# Dynamics
learned_dynamics = {'type': DynamicsLRPrior,
                    'regularization': 1e-6,
                    'prior': {
                        'type': DynamicsPriorGMM,
                        'max_clusters': 20,  # Maximum number of clusters to fit.
                        'min_samples_per_cluster': 40,  # Minimum samples per cluster.
                        'max_samples': 20,  # Max. number of trajectories to use for fitting the GMM at any given time.
                        'strength': 1.0,  # Adjusts the strength of the prior.
                        },
                    }

# GPS algo hyperparameters
mdgps_hyperparams = {'init_pol_wt': 0.01,  # TODO: remove need for init_pol_wt in MDGPS (It should not work with MDGPS)
                     'policy_sample_mode': 'add',
                     'step_rule': 'laplace',  # Whether to use 'laplace' or 'mc' cost in step adjustment
                     'policy_prior': {'type': ConstantPolicyPrior,
                                      'strength': 1e-4,
                                      },
                     }

pigps_hyperparams = {'init_pol_wt': 0.01,
                     'policy_sample_mode': 'add'
                     }

ilqr_hyperparams = {'inner_iterations': 1,
                    }

pi2_hyperparams = {'inner_iterations': 1,
                   'fit_dynamics': False,  # Dynamics fitting is not required for PI2.
                   }

dreps_hyperparams = {'inner_iterations': 1,
                     'good_samples': good_trajs,
                     'bad_samples': bad_trajs,
                     }

mdreps_hyperparams = {'inner_iterations': 1,
                      'good_samples': good_trajs,
                      'bad_samples': bad_trajs,
                      'n_bad_samples': 2,  # Number of bad samples per each trajectory
                      'n_good_samples': 2,  # Number of bad samples per each trajectory
                      'base_kl_good': 1.0,  # (xi) to be used with multiplier | kl_div_g <= kl_good
                      'base_kl_bad': 2.5,  # (chi) to be used with multiplier | kl_div_b >= kl_bad
                      'init_eta': 4.62,
                      'init_nu': 0,#0.5,
                      'init_omega': 0,#1.0,
                      'min_good_mult': 0.01,  # Min possible value of step multiplier (multiplies base_kl_good in LQR)
                      'max_good_mult': 20.0,  # Max possible value of step multiplier (multiplies base_kl_good in LQR)
                      'min_bad_mult': 0.01,  # Min possible value of step multiplier (multiplies base_kl_bad in LQR)
                      'max_bad_mult': 20.0,  # Max possible value of step multiplier (multiplies base_kl_bad in LQR)
                      'min_good_var': np.array([3.0e-1, 3.0e-1, 3.0e-1, 3.0e-1, 1.0e-1, 1.0e-1, 1.0e-1])*1.0e-00,
                      'min_bad_var': np.array([3.0e-1, 3.0e-1, 3.0e-1, 3.0e-1, 1.0e-1, 1.0e-1, 1.0e-1])*1.0e-00,
                      }


if learning_algorithm.upper() == 'MDGPS':
    gps_algo_hyperparams = mdgps_hyperparams
    traj_opt_method = traj_opt_lqr
    test_after_iter = True
    sample_on_policy = False
    use_global_policy = True

elif learning_algorithm.upper() == 'PIGPS':
    mdgps_hyperparams.update(pigps_hyperparams)
    gps_algo_hyperparams = mdgps_hyperparams
    traj_opt_method = traj_opt_pi2
    test_after_iter = True
    sample_on_policy = False
    use_global_policy = True

elif learning_algorithm.upper() == 'ILQR':
    gps_algo_hyperparams = ilqr_hyperparams
    traj_opt_method = traj_opt_lqr
    test_after_iter = False
    sample_on_policy = False
    use_global_policy = False

elif learning_algorithm.upper() == 'PI2':
    gps_algo_hyperparams = pi2_hyperparams
    traj_opt_method = traj_opt_pi2
    test_after_iter = False
    sample_on_policy = False
    use_global_policy = False

elif learning_algorithm.upper() == 'DREPS':
    gps_algo_hyperparams = dreps_hyperparams
    traj_opt_method = traj_opt_dreps
    test_after_iter = False
    sample_on_policy = False
    use_global_policy = False

elif learning_algorithm.upper() == 'MDREPS':
    gps_algo_hyperparams = mdreps_hyperparams
    traj_opt_method = traj_opt_mdreps
    test_after_iter = False
    sample_on_policy = False
    use_global_policy = False
else:
    raise AttributeError("Wrong learning algorithm %s" % learning_algorithm.upper())


gps_hyperparams = {
    'T': int(EndTime/Ts),  # Total points
    'dt': Ts,
    'iterations': 100,  # 100  # 2000  # GPS episodes, "inner iterations" --> K iterations
    'test_after_iter': test_after_iter,  # If test the learned policy after an iteration in the RL algorithm
    'test_samples': 2,  # Samples from learned policy after an iteration PER CONDITION (only if 'test_after_iter':True)
    # Samples
    'num_samples': 6,  # 20  # Samples for exploration trajs --> N samples
    'noisy_samples': True,
    'sample_on_policy': sample_on_policy,  # Whether generate on-policy samples or off-policy samples
    #'noise_var_scale': np.array([5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2]),  # Scale to Gaussian noise: N(0,1)*sqrt(noise_var_scale)
    #'noise_var_scale': np.array([1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1])*10,  # Scale to Gaussian noise: N(0,1)*sqrt(noise_var_scale)
    'smooth_noise': True,  # Apply Gaussian filter to noise generated
    'smooth_noise_var': 5.0e+0,  # np.power(2*Ts, 2), # Variance to apply to Gaussian Filter. In Kumar (2016) paper, it is the std dev of 2 Ts
    'smooth_noise_renormalize': True,  # Renormalize smooth noise to have variance=1
    'noise_var_scale': np.ones(action_dim),  # Scale to Gaussian noise: N(0, 1)*sqrt(noise_var_scale), only if smooth_noise_renormalize
    'cost': cost_sum,
    # Conditions
    'conditions': len(gym_env.get_conditions()),  # Total number of initial conditions
    'train_conditions': range(len(gym_env.get_conditions())),  # Indexes of conditions used for training
    'test_conditions': range(len(gym_env.get_conditions())),  # Indexes of conditions used for testing
    # KL step (epsilon)
    'kl_step': 0.2,  # Kullback-Leibler step (base_step)
    'min_step_mult': 0.01,  # Min possible value of step multiplier (multiplies kl_step in LQR)
    'max_step_mult': 10.0,  # Previous 23/08 -> 1.0 #3 # 10.0,  # Max possible value of step multiplier (multiplies kl_step in LQR)
    # Others
    'gps_algo_hyperparams': gps_algo_hyperparams,
    'init_traj_distr': init_traj_distr,
    'fit_dynamics': True,
    'dynamics': learned_dynamics,
    'initial_state_var': 1e-6,  #1e-2,# 1e-6,  # Max value for x0sigma in trajectories  # TODO: CHECK THIS VALUE, maybe it is too low
    'traj_opt': traj_opt_method,
    'max_ent_traj': 0.0,  # Weight of maximum entropy term in trajectory optimization  # CHECK THIS VALUE!!!, I AM USING ZERO!!
    'use_global_policy': use_global_policy,
    'data_files_dir': data_files_dir,
}


if learning_algorithm.upper() == 'MDGPS':
    learn_algo = MDGPS(agent=gym_agent, env=gym_env, **gps_hyperparams)

elif learning_algorithm.upper() == 'PIGPS':
    learn_algo = PIGPS(agent=gym_agent, env=gym_env, **gps_hyperparams)

elif learning_algorithm.upper() == 'ILQR':
    learn_algo = ILQR(agent=gym_agent, env=gym_env, **gps_hyperparams)

elif learning_algorithm.upper() == 'PI2':
    learn_algo = PI2(agent=gym_agent, env=gym_env, **gps_hyperparams)

elif learning_algorithm.upper() == 'DREPS':
    learn_algo = DREPS(agent=gym_agent, env=gym_env, **gps_hyperparams)

elif learning_algorithm.upper() == 'MDREPS':
    learn_algo = MDREPS(agent=gym_agent, env=gym_env, **gps_hyperparams)

else:
    raise AttributeError("Wrong learning algorithm %s" % learning_algorithm.upper())

print("Learning algorithm: %s OK\n" % type(learn_algo))

# import numpy as np
# dX = gym_env.get_state_dim()
# dU = gym_env.get_action_dim()
# dO = gym_env.get_obs_dim()
# T = gps_hyperparams['T']
# all_actions = np.zeros((T, dU))
# all_states = np.tile(np.expand_dims(np.linspace(0.5, 0, T), axis=1), (1, dX))
# all_obs = np.tile(np.expand_dims(np.linspace(0.5, 0, T), axis=1), (1, dO))
# sample = Sample(gym_env, T)
# sample.set_acts(all_actions)  # Set all actions at the same time
# sample.set_obs(all_obs)  # Set all obs at the same time
# sample.set_states(all_states)  # Set all states at the same time
# costs = learn_algo._eval_conditions_sample_list_cost([SampleList([sample])])
# raw_input('zacataaaaaaaaa')


# Optimize policy using learning algorithm
print("Running Learning Algorithm!!!")
training_successful = learn_algo.run(resume_training_itr)
if training_successful:
    print("Learning Algorithm has finished SUCCESSFULLY!")
else:
    print("Learning Algorithm has finished WITH ERRORS!")


# ############################## #
# ############################## #
# ## SAMPLE FROM FINAL POLICY ## #
# ############################## #
# ############################## #
if training_successful:
    conditions_to_sample = gps_hyperparams['test_conditions']
    change_print_color.change('GREEN')
    n_samples = 1
    noisy = False
    sampler_hyperparams = {
        'noisy': noisy,
        'noise_var_scale': 0.0001,  # It can be a np.array() with dim=dU
        'smooth_noise': False,  # Whether or not to perform smoothing of noise
        'smooth_noise_var': 0.01,   # If smooth=True, applies a Gaussian filter with this variance. E.g. 0.01
        'smooth_noise_renormalize': False,  # If smooth=True, renormalizes data to have variance 1 after smoothing.
        'T': int(EndTime/Ts)*1,  # Total points
        'dt': Ts
        }
    #sampler = Sampler(gym_agent.policy, gym_env, **sampler_hyperparams)
    sampler = Sampler(learn_algo.cur[0].traj_distr, gym_env, **sampler_hyperparams)
    print("Sampling from final policy!!!")
    sample_lists = list()
    for cond_idx in conditions_to_sample:
        raw_input("\nSampling %d times from condition %d and with policy:%s (noisy:%s). \n Press a key to continue..." %
              (n_samples, cond_idx, type(gym_agent.policy), noisy))
        sample_list = sampler.take_samples(n_samples, cond=cond_idx, noisy=noisy)
        # costs = learn_algo._eval_conditions_sample_list_cost([sample_list])
        # # print(costs)
        # # raw_input('pppp')
        # sample_lists.append(sample_list)

    gym_env.reset(time=1, cond=0)




print("The script has finished!")
os._exit(0)

