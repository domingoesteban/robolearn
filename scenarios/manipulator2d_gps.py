from __future__ import print_function

import os
import random
import signal
import numpy as np

from robolearn.agents import GPSAgent
from robolearn.algos.gps.gps import GPS
from robolearn.costs.cost_action import CostAction
from robolearn.costs.cost_fk import CostFK
# from robolearn.costs.cost_state import CostState
from robolearn.costs.cost_sum import CostSum
from robolearn.costs.cost_utils import RAMP_FINAL_ONLY, RAMP_CONSTANT
from robolearn.costs.cost_utils import evall1l2term
# from robolearn.envs import BigmanEnv
from robolearn.policies.lin_gauss_init import init_pd, init_dual_demos
from robolearn.policies.policy_opt.policy_opt_tf import PolicyOptTf
from robolearn.policies.policy_opt.tf_models import tf_network
from robolearn.policies.policy_prior import ConstantPolicyPrior  # For MDGPS
from robolearn.utils.dynamics.dynamics_lr_prior import DynamicsLRPrior
from robolearn.utils.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
# from robolearn.utils.iit.iit_robots_params import bigman_params
# from robolearn.utils.robot_model import RobotModel
# from robolearn.utils.tasks.bigman.reach_drill_utils import Reset_condition_bigman_drill_gazebo
# from robolearn.utils.tasks.bigman.reach_drill_utils import create_bigman_drill_condition
# from robolearn.utils.tasks.bigman.reach_drill_utils import create_drill_relative_pose
# from robolearn.utils.tasks.bigman.reach_drill_utils import create_hand_relative_pose
# from robolearn.utils.tasks.bigman.reach_drill_utils import spawn_drill_gazebo
# from robolearn.utils.tasks.bigman.reach_drill_utils import task_space_torque_control_dual_demos, \
#     load_task_space_torque_control_dual_demos
# from robolearn.utils.traj_opt.traj_opt_mdreps import TrajOptMDREPS
from robolearn.utils.traj_opt.traj_opt_lqr import TrajOptLQR


from robolearn.envs.manipulator2d.manipulator2d_env import Manipulator2dEnv
from robolearn.utils.print_utils import change_print_color

np.set_printoptions(precision=4, suppress=True, linewidth=1000)


def kill_everything(_signal=None, _frame=None):
    print("\n\033[1;31mThe script has been kill by the user!!")
    manipulator2d_env.gz_ros_process.stop()
    os._exit(1)

signal.signal(signal.SIGINT, kill_everything)


# ################## #
# ################## #
# ### PARAMETERS ### #
# ################## #
# ################## #
learning_algorithm = 'MDREPS'
# Task parameters
Ts = 0.02
EndTime = 5
seed = 0

random.seed(seed)
np.random.seed(seed)

# ################### #
# ################### #
# ### ENVIRONMENT ### #
# ################### #
# ################### #
change_print_color.change('BLUE')
print("\nCreating Manipulator2d environment...")

# Robot configuration
interface = 'ros'
body_part_active = 'RA'
body_part_sensed = 'RA'
command_type = 'effort'

# reset_condition_bigman_drill_gazebo_fcn = Reset_condition_bigman_drill_gazebo()

# Create a BIGMAN ROS EnvInterface
manipulator2d_env = Manipulator2dEnv()

raw_input('esperameeee')


action_dim = manipulator2d_env.get_action_dim()
state_dim = manipulator2d_env.get_state_dim()
observation_dim = manipulator2d_env.get_obs_dim()

print("Manipulator2d Environment OK. \n action_dim=%02d, obs_dim=%02d, state_dim=%0.02d" % (action_dim, observation_dim,
                                                                                            state_dim))



# ################# #
# ################# #
# ##### AGENT ##### #
# ################# #
# ################# #
change_print_color.change('CYAN')
print("\nCreating Bigman Agent...")

policy_params = {'network_model': tf_network,  # tf_network, multi_modal_network, multi_modal_network_fp
                 'network_params': {
                     'n_layers': 2,  # Hidden layers??
                     'dim_hidden': [40, 40],  # List of size per n_layers
                     'obs_names': manipulator2d_env.get_obs_info()['names'],
                     'obs_dof': manipulator2d_env.get_obs_info()['dimensions'],  # DoF for observation data tensor
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
                 'use_gpu': 1,  # Whether or not to use the GPU for training.
                 'gpu_id': 0,
                 'random_seed': 1,
                 'fc_only_iterations': 0,  # TODO: Only forwardcontrol? if it is CNN??
                 'gpu_mem_percentage': 0.2,
                 # 'weights_file_prefix': EXP_DIR + 'policy',
             }

policy_opt = {
    'type': PolicyOptTf,
    'hyperparams': policy_params
    }

manipulator2d_agent = GPSAgent(act_dim=action_dim, obs_dim=observation_dim, state_dim=state_dim, policy_opt=policy_opt,
                               agent_name="bigman_agent")


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
# target_distance_hand = np.zeros(6)

# target_distance_object = np.zeros(6)
# fk_l2_cost = {
#     'type': CostFK,
#     'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
#     'target_pose': target_distance_hand,
#     'tgt_data_type': 'state',  # 'state' or 'observation'
#     'tgt_idx': manipulator2d_env.get_state_info(name='distance_hand')['idx'],
#     'op_point_name': hand_name,
#     'op_point_offset': hand_offset,
#     'joints_idx': manipulator2d_env.get_state_info(name='link_position')['idx'],
#     'joint_ids': bigman_params['joint_ids'][body_part_active],
#     'robot_model': robot_model,
#     # 'wp': np.array([1.0, 1.0, 1.0, 0.7, 0.8, 0.6]),  # one dim less because 'quat' error | 1)orient 2)pos
#     #'wp': np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
#     'wp': np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
#     'evalnorm': evall1l2term,
#     #'evalnorm': evallogl2term,
#     'l1': 0.0,  # 1.0,  # 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
#     'l2': 1.0,  # 1.0,  #1.0e-3,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
#     'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
#     'wp_final_multiplier': 1,  # 10
# }
#
# fk_l2_final_cost = {
#     'type': CostFK,
#     'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
#     'target_pose': target_distance_hand,
#     'tgt_data_type': 'state',  # 'state' or 'observation'
#     'tgt_idx': manipulator2d_env.get_state_info(name='distance_hand')['idx'],
#     'op_point_name': hand_name,
#     'op_point_offset': hand_offset,
#     'joints_idx': manipulator2d_env.get_state_info(name='link_position')['idx'],
#     'joint_ids': bigman_params['joint_ids'][body_part_active],
#     'robot_model': robot_model,
#     #'wp': np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
#     'wp': np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
#     'evalnorm': evall1l2term,
#     #'evalnorm': evallogl2term,
#     'l1': 0.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
#     'l2': 1.0,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
#     'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
#     'wp_final_multiplier': 50,
# }

cost_sum = {
    'type': CostSum,
    'costs': [act_cost],  #, fk_l2_cost, fk_l2_final_cost],
    'weights': [1.0e-1],  # 1.5e-1, 1.0e-0],
}


# ########## #
# ########## #
# Conditions #
# ########## #
# ########## #

# # REACH FROM TOP +0.2
# q3[24] = np.deg2rad(-31.8328)
# q3[25] = np.deg2rad(-39.7085)
# q3[26] = np.deg2rad(11.934)
# q3[27] = np.deg2rad(-81.7872)
# q3[28] = np.deg2rad(43.8094)
# q3[29] = np.deg2rad(-7.5974)
# q3[30] = np.deg2rad(4.1521)
# drill_pose3 = create_drill_relative_pose(drill_x=drill_x+0.16, drill_y=drill_y-0.2276, drill_z=drill_z+0.17, drill_yaw=drill_yaw)  # TODO: CHECK IF IT IS OK +0.17 WITH DRILL
# condition3 = create_bigman_drill_condition(q3, drill_pose3, bigman_env.get_state_info(),
#                                            joint_idxs=bigman_params['joint_ids'][body_part_sensed])
# manipulator2d_env.add_condition(condition3)
# # reset_condition_bigman_drill_gazebo_fcn.add_reset_poses(drill_pose3)
# # drill_relative_poses.append(drill_pose3)

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
data_files_dir = None  # 'GPS_2017-09-05_13:07:23'  # None  # In case we want to resume from previous training

init_traj_distr = {'type': init_pd,
                   #'init_var': np.ones(len(bigman_params['joint_ids'][body_part_active]))*0.3e-1,  # Initial variance (Default:10)
                   'init_var': np.array([3.0e-1, 3.0e-1, 3.0e-1, 3.0e-1, 1.0e-1, 1.0e-1, 1.0e-1])*1.0e-00,
                   #'init_var': np.array([3.0e-1, 3.0e-1, 3.0e-1, 3.0e-1, 2.0e-1, 2.0e-1, 2.0e-1])*1.0e+0,
                   #'init_var': np.ones(7)*0.5,
                   #'init_var': np.ones(len(bigman_params['joint_ids'][body_part_active])),  # Initial variance (Default:10)
                   # 'init_var': np.array([3.0e-1, 3.0e-1, 3.0e-1, 3.0e-1, 1.0e-1, 1.0e-1, 1.0e-1,
                   #                       3.0e-1, 3.0e-1, 3.0e-1, 3.0e-1, 1.0e-1, 1.0e-1, 1.0e-1])*1.0,  # Initial variance (Default:10)
                   'pos_gains': 0.001,  #np.array([1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 5.0e-2, 5.0e-2, 5.0e-2])*1.0e+1,  # 0.001,  # Position gains (Default:10)
                   'vel_gains_mult': 0.01,  # Velocity gains multiplier on pos_gains
                   'init_action_offset': None,
                   'dJoints': manipulator2d_env.get_action_dim(),  # Total joints in state
                   'state_to_pd': 'joints',  # Joints
                   'dDistance': 6,
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

traj_opt_method = {'type': TrajOptLQR,
                   'del0': 1e-4,  # Dual variable updates for non-SPD Q-function (non-SPD correction step).
                   # 'eta_error_threshold': 1e16, # TODO: REMOVE, it is not used
                   'min_eta': 1e-8,  # At min_eta, kl_div > kl_step
                   'max_eta': 1e16,  # At max_eta, kl_div < kl_step
                   'cons_per_step': False,  # Whether or not to enforce separate KL constraints at each time step.
                   'use_prev_distr': False,  # Whether or not to measure expected KL under the previous traj distr.
                   'update_in_bwd_pass': True,  # Whether or not to update the TVLG controller during the bwd pass.
                   }

gps_algo_hyperparams = [
    {'inner_iterations': 1,
     'init_eta': 4.62,
     'init_pol_wt': 0.01,  # TODO: remove need for init_pol_wt in MDGPS (It should not work with MDGPS)
     'policy_sample_mode': 'add',
     'step_rule': 'laplace',  # Whether to use 'laplace' or 'mc' cost in step adjustment
     'policy_prior': {'type': ConstantPolicyPrior,
                      'strength': 1e-4,
                      },
     },
]

sample_on_policy = False
use_global_policy = True
test_after_iter = True
#use_global_policy = False


gps_hyperparams = {
    'T': int(EndTime/Ts),  # Total points
    'dt': Ts,
    'iterations': 25,  # 100  # 2000  # GPS episodes, "inner iterations" --> K iterations
    'test_after_iter': test_after_iter,  # If test the learned policy after an iteration in the RL algorithm
    'test_samples': 2,  # Samples from learned policy after an iteration PER CONDITION (only if 'test_after_iter':True)
    # Samples
    'num_samples': 6,  # 20  # Samples for exploration trajs --> N samples
    'noisy_samples': True,
    'sample_on_policy': sample_on_policy,  # Whether generate on-policy samples or off-policy samples
    #'noise_var_scale': np.array([5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2]),  # Scale to Gaussian noise: N(0,1)*sqrt(noise_var_scale)
    #'noise_var_scale': np.array([1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1])*10,  # Scale to Gaussian noise: N(0,1)*sqrt(noise_var_scale)
    'smooth_noise': True,  # Apply Gaussian filter to noise generated
    #'smooth_noise_var': 5.0e+0,  # np.power(2*Ts, 2), # Variance to apply to Gaussian Filter. In Kumar (2016) paper, it is the std dev of 2 Ts
    'smooth_noise_var': 8.0e+0,  # np.power(2*Ts, 2), # Variance to apply to Gaussian Filter. In Kumar (2016) paper, it is the std dev of 2 Ts
    'smooth_noise_renormalize': True,  # Renormalize smooth noise to have variance=1
    'noise_var_scale': np.ones(manipulator2d_env.get_action_dim()),  # Scale to Gaussian noise: N(0, 1)*sqrt(noise_var_scale), only if smooth_noise_renormalize
    'cost': cost_sum,
    # Conditions
    'conditions': len(manipulator2d_env.get_conditions()),  # Total number of initial conditions
    'train_conditions': range(len(manipulator2d_env.get_conditions())),  # Indexes of conditions used for training
    'test_conditions': range(len(manipulator2d_env.get_conditions())),  # Indexes of conditions used for testing
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

learn_algo = GPS(manipulator2d_agent, manipulator2d_env, **gps_hyperparams)

print("Learning algorithm: %s OK\n" % type(learn_algo))

# Optimize policy using learning algorithm
raw_input("Press a key to start...")
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
# if training_successful:
#     conditions_to_sample = gps_hyperparams['test_conditions']
#     change_print_color.change('GREEN')
#     n_samples = 1
#     noisy = False
#     sampler_hyperparams = {
#         'noisy': noisy,
#         'noise_var_scale': 0.0001,  # It can be a np.array() with dim=dU
#         'smooth_noise': False,  # Whether or not to perform smoothing of noise
#         'smooth_noise_var': 0.01,   # If smooth=True, applies a Gaussian filter with this variance. E.g. 0.01
#         'smooth_noise_renormalize': False,  # If smooth=True, renormalizes data to have variance 1 after smoothing.
#         'T': int(EndTime/Ts)*1,  # Total points
#         'dt': Ts
#         }
#     sampler = Sampler(learn_algo.cur[0].traj_distr, manipulator2d_env, **sampler_hyperparams)
#     print("Sampling from final policy!!!")
#     sample_lists = list()
#     for cond_idx in conditions_to_sample:
#         raw_input("\nSampling %d times from condition %d and with policy:%s (noisy:%s). \n Press a key to continue..." %
#               (n_samples, cond_idx, type(bigman_agent.policy), noisy))
#         sample_list = sampler.take_samples(n_samples, cond=cond_idx, noisy=noisy)
#         # costs = learn_algo._eval_conditions_sample_list_cost([sample_list])
#         # # print(costs)
#         # # raw_input('pppp')
#         # sample_lists.append(sample_list)
#
#     manipulator2d_env.reset(time=1, cond=0)




print("The script has finished!")
os._exit(0)

