from __future__ import print_function

import os
import random
import signal

import numpy as np
from robolearn.old_utils.sampler import Sampler

from robolearn.old_agents import GPSAgent
from robolearn.old_algos.gps.multi_mdgps import MultiMDGPS
from robolearn.old_costs.cost_action import CostAction
from robolearn.old_costs.cost_fk import CostFK
from robolearn.old_costs.cost_sum import CostSum
from robolearn.old_costs.cost_utils import RAMP_FINAL_ONLY, RAMP_CONSTANT
from robolearn.old_envs import BigmanEnv
from robolearn.old_policies.lin_gauss_init import init_pd, init_demos
from robolearn.old_policies.policy_opt.policy_opt_tf import PolicyOptTf
from robolearn.old_policies.policy_opt.tf_models import tf_network
from robolearn.old_policies.policy_prior import ConstantPolicyPrior  # For MDGPS
from robolearn.old_utils.dynamics.dynamics_lr_prior import DynamicsLRPrior
from robolearn.old_utils.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from robolearn.old_utils.iit.iit_robots_params import bigman_params
from robolearn.old_utils.print_utils import change_print_color
from robolearn.old_utils.robot_model import RobotModel
from robolearn.old_utils.tasks.bigman.lift_box_utils import Reset_condition_bigman_box_gazebo
from robolearn.old_utils.tasks.bigman.lift_box_utils import create_bigman_box_condition
from robolearn.old_utils.tasks.bigman.lift_box_utils import create_box_relative_pose
from robolearn.old_utils.tasks.bigman.lift_box_utils import create_hand_relative_pose
from robolearn.old_utils.tasks.bigman.lift_box_utils import spawn_box_gazebo
from robolearn.old_utils.tasks.bigman.lift_box_utils import task_space_torque_control_demos, \
    load_task_space_torque_control_demos
from robolearn.old_utils.traj_opt.traj_opt_lqr import TrajOptLQR

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
# Task parameters
Ts = 0.01
Treach = 5
Tlift = 0  # 3.8
Tinter = 0  # 0.5
Tend = 0  # 0.7
# EndTime = 4  # Using final time to define the horizon
EndTime = Treach + Tinter + Tlift + Tend  # Using final time to define the horizon
init_with_demos = False
demos_dir = None  # 'TASKSPACE_TORQUE_CTRL_DEMO_2017-07-21_16:32:39'
seed = 6

random.seed(seed)
np.random.seed(seed)

# BOX
box_x = 0.70
box_y = 0.00
box_z = 0.0184
box_yaw = 0  # Degrees
box_size = [0.4, 0.5, 0.3]
final_box_height = 0.0
box_relative_pose = create_box_relative_pose(box_x=box_x, box_y=box_y, box_z=box_z, box_yaw=box_yaw)

# Robot Model (It is used to calculate the IK cost)
#robot_urdf_file = os.environ["ROBOTOLOGY_ROOT"]+'/configs/ADVR_shared/bigman/urdf/bigman.urdf'
robot_urdf_file = os.environ["ROBOTOLOGY_ROOT"]+'/robots/iit-bigman-ros-pkg/bigman_urdf/urdf/bigman.urdf'
robot_model = RobotModel(robot_urdf_file)
LH_name = 'LWrMot3'
RH_name = 'RWrMot3'
l_soft_hand_offset = np.array([0.000, -0.030, -0.210])
r_soft_hand_offset = np.array([0.000, 0.030, -0.210])

touching_box_config = np.array([0.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0.,  0.,  0.,  0.,  0.,
                                0.,  0.,  0.,
                                0.0568,  0.2386, -0.2337, -1.6803,  0.2226,  0.0107,  0.5633,
                                #0.,  0.,  0.,  -1.5708,  0.,  0., 0.,
                                0.,  0.,
                                0.0568,  -0.2386, 0.2337, -1.6803,  -0.2226,  0.0107,  -0.5633])
                                #0.,  0.,  0.,  -1.5708,  0.,  0., 0.])

# ################### #
# ################### #
# ### ENVIRONMENT ### #
# ################### #
# ################### #
change_print_color.change('BLUE')
print("\nCreating Bigman environment...")

# Robot configuration
interface = 'ros'
body_part_active = 'BA'
body_part_sensed = 'BA'
command_type = 'effort'


left_hand_rel_pose = create_hand_relative_pose([0, 0, 0, 1, 0, 0, 0],
                                               hand_x=0.0, hand_y=box_size[1]/2-0.02, hand_z=0.0, hand_yaw=0)
# left_hand_rel_pose[:] = left_hand_rel_pose[[3, 4, 5, 6, 0, 1, 2]]  # Changing from 'pos+orient' to 'orient+pos'
right_hand_rel_pose = create_hand_relative_pose([0, 0, 0, 1, 0, 0, 0],
                                                hand_x=0.0, hand_y=-box_size[1]/2+0.02, hand_z=0.0, hand_yaw=0)
# right_hand_rel_pose[:] = right_hand_rel_pose[[3, 4, 5, 6, 0, 1, 2]]  # Changing from 'pos+orient' to 'orient+pos'

reset_condition_bigman_box_gazebo_fcn = Reset_condition_bigman_box_gazebo()

observation_active = [{'name': 'joint_state',
                       'type': 'joint_state',
                       'ros_topic': '/xbotcore/bigman/joint_states',
                       # 'fields': ['link_position', 'link_velocity', 'effort'],
                       'fields': ['link_position', 'link_velocity'],
                       # 'joints': bigman_params['joint_ids']['UB']},
                       'joints': bigman_params['joint_ids'][body_part_sensed]},

                      {'name': 'prev_cmd',
                       'type': 'prev_cmd'},

                      {'name': 'distance_left_arm',
                       'type': 'fk_pose',
                       'body_name': LH_name,
                       'body_offset': l_soft_hand_offset,
                       'target_offset': left_hand_rel_pose,
                       'fields': ['orientation', 'position']},

                      {'name': 'distance_right_arm',
                       'type': 'fk_pose',
                       'body_name': RH_name,
                       'body_offset': r_soft_hand_offset,
                       'target_offset': right_hand_rel_pose,
                       'fields': ['orientation', 'position']},
                      ]

state_active = [{'name': 'joint_state',
                 'type': 'joint_state',
                 'fields': ['link_position', 'link_velocity'],
                 'joints': bigman_params['joint_ids'][body_part_sensed]},

                {'name': 'prev_cmd',
                 'type': 'prev_cmd'},

                {'name': 'distance_left_arm',
                 'type': 'fk_pose',
                 'body_name': LH_name,
                 'body_offset': l_soft_hand_offset,
                 'target_offset': left_hand_rel_pose,
                 'fields': ['orientation', 'position']},

                {'name': 'distance_right_arm',
                 'type': 'fk_pose',
                 'body_name': RH_name,
                 'body_offset': r_soft_hand_offset,
                 'target_offset': right_hand_rel_pose,
                 'fields': ['orientation', 'position']},
                ]


# Spawn Box first because it is simulation
spawn_box_gazebo(box_relative_pose, box_size=box_size)

# Create a BIGMAN ROS EnvInterface
bigman_env = BigmanEnv(interface=interface, mode='simulation',
                       body_part_active=body_part_active, command_type=command_type,
                       observation_active=observation_active,
                       state_active=state_active,
                       cmd_freq=int(1/Ts),
                       robot_dyn_model=robot_model,
                       reset_simulation_fcn=reset_condition_bigman_box_gazebo_fcn)
                       # reset_simulation_fcn=reset_condition_bigman_box_gazebo)

action_dim = bigman_env.action_dim
state_dim = bigman_env.state_dim
observation_dim = bigman_env.obs_dim

print("Bigman Environment OK. body_part_active:%s (action_dim=%d). Command_type:%s" % (body_part_active, action_dim,
                                                                                       command_type))


# ################# #
# ################# #
# ##### AGENT ##### #
# ################# #
# ################# #
change_print_color.change('CYAN')
print("\nCreating Bigman Agent...")

policy_params = {
    'network_model': tf_network,  # tf_network, multi_modal_network, multi_modal_network_fp
    'network_params': {
        'n_layers': 1,  # Hidden layers??
        'dim_hidden': [40],  # List of size per n_layers
        'obs_names': bigman_env.get_obs_info()['names'],
        'obs_dof': bigman_env.get_obs_info()['dimensions'],  # DoF for observation data tensor
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
    # 'weights_file_prefix': EXP_DIR + 'policy',
}
policy_opt = {
    'type': PolicyOptTf,
    'hyperparams': policy_params
    }

bigman_agent = GPSAgent(act_dim=action_dim, obs_dim=observation_dim, state_dim=state_dim, policy_opt=policy_opt)
print("Bigman Agent:%s OK\n" % type(bigman_agent))


# ################# #
# ################# #
# ##### COSTS ##### #
# ################# #
# ################# #
# Action Cost
wu_LA = np.zeros(action_dim)
wu_LA[:7] = 1
act_cost_LA = {
    'type': CostAction,
    'wu': wu_LA * 1e-4,
    'target': None,   # Target action value
}

wu_RA = np.zeros(action_dim)
wu_RA[7:] = 1
act_cost_RA = {
    'type': CostAction,
    'wu': wu_RA * 1e-4,
    'target': None,   # Target action value
}

# State Cost
target_distance_left_arm = np.zeros(6)
target_distance_right_arm = np.zeros(6)
# state_cost_distance = {
#     'type': CostState,
#     'ramp_option': RAMP_QUADRATIC,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
#     'l1': 0.1,  # Weight for l1 norm
#     'l2': 1.0,  # Weight for l2 norm
#     'alpha': 1e-2,  # Constant added in square root in l1 norm
#     'wp_final_multiplier': 10.0,  # Weight multiplier on final time step.
#     'data_types': {
#         'distance_left_arm': {
#             # 'wp': np.ones_like(target_state),  # State weights - must be set.
#             'wp': np.array([1.0, 1.0, 1.0, 3.0, 3.0, 1.0]),  # State weights - must be set.
#             'target_state': target_distance_left_arm,  # Target state - must be set.
#             'average': None,  # (12, 3),
#             'data_idx': bigman_env.get_state_info(name='distance_left_arm')['idx']
#         },
#         'distance_right_arm': {
#             # 'wp': np.ones_like(target_state),  # State weights - must be set.
#             'wp': np.array([1.0, 1.0, 1.0, 3.0, 3.0, 1.0]),  # State weights - must be set.
#             'target_state': target_distance_right_arm,  # Target state - must be set.
#             'average': None,  # (12, 3),
#             'data_idx': bigman_env.get_state_info(name='distance_right_arm')['idx']
#         },
#     },
# }

LAfk_cost = {
    'type': CostFK,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_left_arm,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': bigman_env.get_state_info(name='distance_left_arm')['idx'],
    'op_point_name': LH_name,
    'op_point_offset': l_soft_hand_offset,
    'joints_idx': bigman_env.get_state_info(name='link_position')['idx'][:7],
    'joint_ids': bigman_params['joint_ids']['LA'],
    'robot_model': robot_model,
    # 'wp': np.array([1.0, 1.0, 1.0, 0.7, 0.8, 0.6]),  # one dim less because 'quat' error | 1)orient 2)pos
    'wp': np.array([1.0, 1.0, 1.0, 6.0, 6.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'l1': 1.0,  # 1.0,  # 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 1.0,  # 1.0,  #1.0e-3,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-2,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 1,  # 10
}

LAfk_final_cost = {
    'type': CostFK,
    'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_left_arm,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': bigman_env.get_state_info(name='distance_left_arm')['idx'],
    'op_point_name': LH_name,
    'op_point_offset': l_soft_hand_offset,
    'joints_idx': bigman_env.get_state_info(name='link_position')['idx'][:7],
    'joint_ids': bigman_params['joint_ids']['LA'],
    'robot_model': robot_model,
    'wp': np.array([1.0, 1.0, 1.0, 8.0, 10.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'l1': 0.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 1.0,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 10,
}

LAcost_sum = {
    'type': CostSum,
    'costs': [act_cost_LA, LAfk_cost, LAfk_final_cost],
    'weights': [1.0e-2, 1.0e-0, 1.0e-0],
}

RAfk_cost = {
    'type': CostFK,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_right_arm,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': bigman_env.get_state_info(name='distance_right_arm')['idx'],
    'op_point_name': RH_name,
    'op_point_offset': r_soft_hand_offset,
    'joints_idx': bigman_env.get_state_info(name='link_position')['idx'][7:],
    'joint_ids': bigman_params['joint_ids']['RA'],
    'robot_model': robot_model,
    # 'wp': np.array([1.0, 1.0, 1.0, 0.7, 0.8, 0.6]),  # one dim less because 'quat' error | 1)orient 2)pos
    'wp': np.array([1.0, 1.0, 1.0, 6.0, 6.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'l1': 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 1.0,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-2,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 1,  # 10
}

RAfk_final_cost = {
    'type': CostFK,
    'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_right_arm,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': bigman_env.get_state_info(name='distance_right_arm')['idx'],
    'op_point_name': RH_name,
    'op_point_offset': r_soft_hand_offset,
    'joints_idx': bigman_env.get_state_info(name='link_position')['idx'][7:],
    'joint_ids': bigman_params['joint_ids']['RA'],
    'robot_model': robot_model,
    'wp': np.array([1.0, 1.0, 1.0, 8.0, 10.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'l1': 0.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 1.0,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 10,
}

RAcost_sum = {
    'type': CostSum,
    'costs': [act_cost_RA, RAfk_cost, RAfk_final_cost],
    'weights': [1.0e-2, 1.0e-0, 1.0e-0],
}

local_agent_costs = [LAcost_sum, RAcost_sum]

BAcost_sum = {
    'type': CostSum,
    'costs': [act_cost_LA, LAfk_cost, LAfk_final_cost, act_cost_RA, RAfk_cost, RAfk_final_cost],
    'weights': [1.0e-2, 1.0e-0, 1.0e-0, 1.0e-2, 1.0e-0, 1.0e-0],
}

bimanual_cost = BAcost_sum


# ########## #
# ########## #
# Conditions #
# ########## #
# ########## #
q0 = np.zeros(31)
q0[15] = np.deg2rad(25)
q0[16] = np.deg2rad(40)
q0[18] = np.deg2rad(-75)
#q0[15:15+7] = [0.0568,  0.2386, -0.2337, -1.6803,  0.2226,  0.0107,  0.5633]
q0[24] = np.deg2rad(25)
q0[25] = np.deg2rad(-40)
q0[27] = np.deg2rad(-75)
#q0[24:24+7] = [0.0568,  -0.2386, 0.2337, -1.6803,  -0.2226,  0.0107,  -0.5633]
box_pose0 = box_relative_pose.copy()
condition0 = create_bigman_box_condition(q0, box_pose0, bigman_env.get_state_info(),
                                         joint_idxs=bigman_params['joint_ids'][body_part_sensed])
bigman_env.add_condition(condition0)
reset_condition_bigman_box_gazebo_fcn.add_reset_poses(box_pose0)

#q1 = np.zeros(31)
q1 = q0.copy()
q1[15] = np.deg2rad(25)
q1[18] = np.deg2rad(-45)
q1[24] = np.deg2rad(25)
q1[27] = np.deg2rad(-45)
box_pose1 = create_box_relative_pose(box_x=box_x+0.02, box_y=box_y+0.02, box_z=box_z, box_yaw=box_yaw+5)
condition1 = create_bigman_box_condition(q1, box_pose1, bigman_env.get_state_info(),
                                         joint_idxs=bigman_params['joint_ids'][body_part_sensed])
bigman_env.add_condition(condition1)
reset_condition_bigman_box_gazebo_fcn.add_reset_poses(box_pose1)

q2 = q0.copy()
q2[16] = np.deg2rad(50)
q2[18] = np.deg2rad(-50)
q2[25] = np.deg2rad(-50)
q2[27] = np.deg2rad(-50)
box_pose2 = create_box_relative_pose(box_x=box_x-0.02, box_y=box_y-0.02, box_z=box_z, box_yaw=box_yaw-5)
condition2 = create_bigman_box_condition(q2, box_pose2, bigman_env.get_state_info(),
                                         joint_idxs=bigman_params['joint_ids'][body_part_sensed])
bigman_env.add_condition(condition2)
reset_condition_bigman_box_gazebo_fcn.add_reset_poses(box_pose2)

# q3 = q0.copy()
# q3[16] = np.deg2rad(0)
# q3[18] = np.deg2rad(0)
# q3[25] = np.deg2rad(0)
# q3[27] = np.deg2rad(0)
# box_pose3 = create_box_relative_pose(box_x=box_x, box_y=box_y, box_z=box_z, box_yaw=box_yaw+5)
# condition3 = create_bigman_box_condition(q3, box_pose3, bigman_env.get_state_info(),
#                                          joint_idxs=bigman_params['joint_ids'][body_part_sensed])
# bigman_env.add_condition(condition3)
# reset_condition_bigman_box_gazebo_fcn.add_reset_poses(box_pose3)

# q4 = q0.copy()
# box_pose4 = create_box_relative_pose(box_x=box_x, box_y=box_y, box_z=box_z, box_yaw=box_yaw-5)
# condition4 = create_bigman_box_condition(q4, box_pose4, bigman_env.get_state_info(),
#                                          joint_idxs=bigman_params['joint_ids'][body_part_sensed])
# bigman_env.add_condition(condition4)
# reset_condition_bigman_box_gazebo_fcn.add_reset_poses(box_pose4)






# #################### #
# #################### #
# ## DEMONSTRATIONS ## #
# #################### #
# #################### #
if init_with_demos is True:
    change_print_color.change('GREEN')
    if demos_dir is None:
        task_space_torque_control_demos_params = {
            'n_samples': 5,
            'conditions_to_sample': range(len(bigman_env.get_conditions())),
            'Treach': Treach,
            'Tlift': Tlift,
            'Tinter': Tinter,
            'Tend': Tend,
            'Ts': Ts,
            'noisy': False,
            'noise_hyperparams': {
                'noise_var_scale': 0.0001,  # It can be a np.array() with dim=dU
                'smooth_noise': False,  # Whether or not to perform smoothing of noise
                'smooth_noise_var': 0.01,   # If smooth=True, applies a Gaussian filter with this variance. E.g. 0.01
                'smooth_noise_renormalize': False,  # If smooth=True, renormalizes data to have variance 1 after smoothing.
            },
            'bigman_env': bigman_env,
            'box_relative_pose': box_relative_pose,
            'box_size': box_size,
            'final_box_height': final_box_height,
        }
        demos_samples = task_space_torque_control_demos(**task_space_torque_control_demos_params)
        bigman_env.reset(time=2, cond=0)
    else:
        demos_samples = load_task_space_torque_control_demos(demos_dir)
        print('Demos samples has been obtained from directory %s' % demos_dir)
else:
    demos_samples = None


# ######################## #
# ######################## #
# ## LEARNING ALGORITHM ## #
# ######################## #
# ######################## #
change_print_color.change('YELLOW')
print("\nConfiguring learning algorithm...\n")

# Learning params
resume_training_itr = 8  # Resume from previous training iteration
data_files_dir = 'GPS_2017-08-09_14:11:15'  # In case we want to resume from previous training

traj_opt_method = {'type': TrajOptLQR,
                   'del0': 1e-4,  # Dual variable updates for non-SPD Q-function (non-SPD correction step).
                   # 'eta_error_threshold': 1e16, # TODO: REMOVE, it is not used
                   'min_eta': 1e-8,  # At min_eta, kl_div > kl_step
                   'max_eta': 1e16,  # At max_eta, kl_div < kl_step
                   'cons_per_step': False,  # Whether or not to enforce separate KL constraints at each time step.
                   'use_prev_distr': False,  # Whether or not to measure expected KL under the previous traj distr.
                   'update_in_bwd_pass': True,  # Whether or not to update the TVLG controller during the bwd pass.
                   }
# traj_opt_method = {'type': TrajOptPI2,
#                    'del0': 1e-4,  # Dual variable updates for non-PD Q-function.
#                    'kl_threshold': 1.0,   # KL-divergence threshold between old and new policies.
#                    'covariance_damping': 2.0,  # If greater than zero, covariance is computed as a multiple of the old
#                                              # covariance. Multiplier is taken to the power (1 / covariance_damping).
#                                              # If greater than one, slows down convergence and keeps exploration noise
#                                              # high for more iterations.
#                    'min_temperature': 0.001,  # Minimum bound of the temperature optimization for the soft-max
#                                               # probabilities of the policy samples.
#                    'use_sumexp': False,
#                    'pi2_use_dgd_eta': False,
#                    'pi2_cons_per_step': True,
#                    }

# init_traj_distr values can be lists if they are different for each condition
# init_traj_distr = {'type': init_lqr,
#                    # Parameters to calculate initial COST function based on stiffness
#                    'init_var': 8.0e-2,  # Initial Variance
#                    'stiffness': 1.0e-1,  # Stiffness (multiplies q)
#                    'stiffness_vel': 0.5,  # Stiffness_vel*stiffness (multiplies qdot)
#                    'final_weight': 10.0,  # Multiplies cost at T
#                    # Parameters for guessing dynamics
#                    'init_acc': np.zeros(action_dim),  # dU vector(np.array) of accelerations, default zeros.
#                    #'init_gains': 1.0*np.ones(action_dim),  # dU vector(np.array) of gains, default ones.
#                    'init_gains': 1.0/np.array([5000.0, 8000.0, 5000.0, 5000.0, 300.0, 2000.0, 300.0]),  # dU vector(np.array) of gains, default ones.
#                    }
if demos_samples is None:
    init_traj_distr = [{'type': init_pd,
                        'init_var': np.array([3.0e-1, 3.0e-1, 3.0e-1, 3.0e-1, 1.0e-1, 1.0e-1, 1.0e-1])*1.0,
                        'pos_gains': 0.1,  # 0.001,  # Position gains (Default:10)
                        'vel_gains_mult': 0.01,  # Velocity gains multiplier on pos_gains
                        'init_action_offset': None,
                        'dQ': len(bigman_params['joint_ids']['LA']),  # Total joints in state
                        },
                       {'type': init_pd,
                        'init_var': np.array([3.0e-1, 3.0e-1, 3.0e-1, 3.0e-1, 1.0e-1, 1.0e-1, 1.0e-1])*1.0,
                        'pos_gains': 0.1,  # 0.001,  # Position gains (Default:10)
                        'vel_gains_mult': 0.01,  # Velocity gains multiplier on pos_gains
                        'init_action_offset': None,
                        'dQ': len(bigman_params['joint_ids']['LA']),  # Total joints in state
                        },
                       ]
else:
    init_traj_distr = {'type': init_demos,
                       'sample_lists': demos_samples
                       }

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

# gps_algo = 'pigps'
# gps_algo_hyperparams = {'init_pol_wt': 0.01,
#                         'policy_sample_mode': 'add'
#                         }
gps_algo = 'mdgps'
gps_algo_hyperparams = {'init_pol_wt': 0.01,  # TODO: remove need for init_pol_wt in MDGPS (It should not work with MDGPS)
                        'policy_sample_mode': 'add',
                        'step_rule': 'laplace',  # Whether to use 'laplace' or 'mc' cost in step adjustment
                        'policy_prior': {'type': ConstantPolicyPrior,
                                         'strength': 1e-4,
                                         },
                        }

left_arm_state_idx = bigman_env.get_state_info(name='link_position')['idx'][:7] + \
                     bigman_env.get_state_info(name='link_velocity')['idx'][:7] + \
                     bigman_env.get_state_info(name='prev_cmd')['idx'][:7] + \
                     bigman_env.get_state_info(name='distance_left_arm')['idx']

right_arm_state_idx = bigman_env.get_state_info(name='link_position')['idx'][7:] + \
                      bigman_env.get_state_info(name='link_velocity')['idx'][7:] + \
                      bigman_env.get_state_info(name='prev_cmd')['idx'][7:] + \
                      bigman_env.get_state_info(name='distance_right_arm')['idx']

gps_hyperparams = {
    'T': int(EndTime/Ts),  # Total points
    'dt': Ts,
    'iterations': 100,  # 100  # 2000  # GPS episodes, "inner iterations" --> K iterations
    'test_after_iter': True,  # If test the learned policy after an iteration in the RL algorithm
    'test_samples': 2,  # Samples from learned policy after an iteration PER CONDITION (only if 'test_after_iter':True)
    # Samples
    'num_samples': 5,  # 20  # Samples for exploration trajs --> N samples
    'noisy_samples': True,
    'sample_on_policy': False,  # Whether generate on-policy samples or off-policy samples
    'noise_var_scale': 1.0e-0,  # Scale to Gaussian noise: N(0,1)*sqrt(noise_var_scale)
    'smooth_noise': True,  # Apply Gaussian filter to noise generated
    'smooth_noise_var': 3.0e+0,  # Variance to apply to Gaussian Filter
    'smooth_noise_renormalize': True,  # Renormalize smooth noise to have variance=1
    'cost': bimanual_cost,
    # Conditions
    'conditions': len(bigman_env.get_conditions()),  # Total number of initial conditions
    'train_conditions': range(len(bigman_env.get_conditions())),  # Indexes of conditions used for training
    'test_conditions': range(len(bigman_env.get_conditions())),  # Indexes of conditions used for testing
    # KL step (epsilon)
    'kl_step': 0.2,  # Kullback-Leibler step (base_step)
    'min_step_mult': 0.01,  # Min possible value of step multiplier (multiplies kl_step in LQR)
    'max_step_mult': 3,  # 10.0,  # Max possible value of step multiplier (multiplies kl_step in LQR)
    # Others
    'gps_algo': gps_algo,
    'gps_algo_hyperparams': gps_algo_hyperparams,
    'init_traj_distr': init_traj_distr,
    'fit_dynamics': True,
    'dynamics': learned_dynamics,
    'initial_state_var': 1e-6,  # Max value for x0sigma in trajectories
    'traj_opt': traj_opt_method,
    'max_ent_traj': 0.0,  # Weight of maximum entropy term in trajectory optimization
    'data_files_dir': data_files_dir,
    # Multi MDGPS
    'local_agent_state_masks': [left_arm_state_idx, right_arm_state_idx],
    'local_agent_action_masks': [range(7), range(7, 14)],
    'local_agent_costs': local_agent_costs,
}


learn_algo = MultiMDGPS(agent=bigman_agent, env=bigman_env, **gps_hyperparams)

print("Learning algorithm: %s OK\n" % type(learn_algo))

# import numpy as np
# dX = bigman_env.state_dim
# dU = bigman_env.action_dim
# dO = bigman_env.obs_dim
# T = gps_hyperparams['T']
# all_actions = np.zeros((T, dU))
# all_states = np.tile(np.expand_dims(np.linspace(0.5, 0, T), axis=1), (1, dX))
# all_obs = np.tile(np.expand_dims(np.linspace(0.5, 0, T), axis=1), (1, dO))
# sample = Sample(bigman_env, T)
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
    sampler = Sampler(bigman_agent.policy, bigman_env, **sampler_hyperparams)
    print("Sampling from final policy!!!")
    sample_lists = list()
    for cond_idx in conditions_to_sample:
        raw_input("\nSampling %d times from condition %d and with policy:%s (noisy:%s). \n Press a key to continue..." %
              (n_samples, cond_idx, type(bigman_agent.policy), noisy))
        sample_list = sampler.take_samples(n_samples, cond=cond_idx, noisy=noisy)
        costs = learn_algo._eval_conditions_sample_list_cost([sample_list])
        # print(costs)
        # raw_input('pppp')
        sample_lists.append(sample_list)

    bigman_env.reset(time=1, cond=0)




print("The script has finished!")
os._exit(0)

