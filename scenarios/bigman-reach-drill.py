from __future__ import print_function

import sys
import os
import signal
import numpy as np
import matplotlib.pyplot as plt

from robolearn.utils.iit.iit_robots_params import bigman_params
from robolearn.envs import BigmanEnv
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
from robolearn.utils.dynamics.dynamics_lr_prior import DynamicsLRPrior
from robolearn.utils.dynamics.dynamics_prior_gmm import DynamicsPriorGMM

from robolearn.algos.gps.mdgps import MDGPS
from robolearn.algos.gps.pigps import PIGPS
from robolearn.algos.trajopt.ilqr import ILQR
from robolearn.algos.trajopt.pi2 import PI2
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
learning_algorithm = 'PIGPS'
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

# BOX
drill_x = 0.70
drill_y = 0.00
drill_z = -0.1327
drill_yaw = 0  # Degrees
drill_size = [0.1, 0.1, 0.3]
final_drill_height = 0.0
drill_relative_pose = create_drill_relative_pose(drill_x=drill_x, drill_y=drill_y, drill_z=drill_z, drill_yaw=drill_yaw)

# Robot Model (It is used to calculate the IK cost)
#robot_urdf_file = os.environ["ROBOTOLOGY_ROOT"]+'/configs/ADVR_shared/bigman/urdf/bigman.urdf'
robot_urdf_file = os.environ["ROBOTOLOGY_ROOT"]+'/robots/iit-bigman-ros-pkg/bigman_urdf/urdf/bigman.urdf'
robot_model = RobotModel(robot_urdf_file)
LH_name = 'LWrMot3'
RH_name = 'RWrMot3'
l_soft_hand_offset = np.array([0.000, -0.030, -0.210])
r_soft_hand_offset = np.array([0.000, 0.030, -0.210])

touching_drill_config = np.array([0.,  0.,  0.,  0.,  0.,  0.,
                                  0.,  0.,  0.,  0.,  0.,  0.,
                                  0.,  0.,  0.,
                                  0.0568,  0.2386, -0.2337, -1.6803,  0.2226,  0.0107,  0.5633,
                                  0.,  0.,
                                  0.0568,  -0.2386, 0.2337, -1.6803,  -0.2226,  0.0107,  -0.5633])

# ################### #
# ################### #
# ### ENVIRONMENT ### #
# ################### #
# ################### #
change_print_color.change('BLUE')
print("\nCreating Bigman environment...")

# Robot configuration
interface = 'ros'
body_part_active = 'RA'
body_part_sensed = 'RA'
command_type = 'effort'

if body_part_active == 'RA':
    hand_y = -drill_size[1]/2+0.02
    hand_z = drill_size[2]/2
    hand_name = RH_name
    hand_offset = r_soft_hand_offset
else:
    hand_y = drill_size[1]/2-0.02
    hand_z = drill_size[2]/2
    hand_name = LH_name
    hand_offset = l_soft_hand_offset

hand_rel_pose = create_hand_relative_pose([0, 0, 0, 1, 0, 0, 0], hand_x=0.0, hand_y=hand_y, hand_z=hand_z, hand_yaw=0)

reset_condition_bigman_drill_gazebo_fcn = Reset_condition_bigman_drill_gazebo()

observation_active = [{'name': 'joint_state',
                       'type': 'joint_state',
                       'ros_topic': '/xbotcore/bigman/joint_states',
                       # 'fields': ['link_position', 'link_velocity', 'effort'],
                       'fields': ['link_position', 'link_velocity'],
                       # 'joints': bigman_params['joint_ids']['UB']},
                       'joints': bigman_params['joint_ids'][body_part_sensed]},

                      {'name': 'prev_cmd',
                       'type': 'prev_cmd'},

                      # {'name': 'ft_right_arm',
                      #  'type': 'ft_sensor',
                      #  'ros_topic': '/xbotcore/bigman/ft/r_arm_ft',
                      #  'fields': ['force', 'torque']},

                      {'name': 'distance_hand',
                       'type': 'fk_pose',
                       'body_name': hand_name,
                       'body_offset': hand_offset,
                       'target_offset': hand_rel_pose,
                       'fields': ['orientation', 'position']},
                      ]

state_active = [{'name': 'joint_state',
                 'type': 'joint_state',
                 'fields': ['link_position', 'link_velocity'],
                 'joints': bigman_params['joint_ids'][body_part_sensed]},

                {'name': 'prev_cmd',
                 'type': 'prev_cmd'},

                {'name': 'distance_hand',
                 'type': 'fk_pose',
                 'body_name': hand_name,
                 'body_offset': hand_offset,
                 'target_offset': hand_rel_pose,
                 'fields': ['orientation', 'position']},
                ]

optional_env_params = {
    'temp_object_name': 'drill'
}

# Spawn Box first because it is simulation
spawn_drill_gazebo(drill_relative_pose, drill_size=drill_size)


# Create a BIGMAN ROS EnvInterface
bigman_env = BigmanEnv(interface=interface, mode='simulation',
                       body_part_active=body_part_active, command_type=command_type,
                       observation_active=observation_active,
                       state_active=state_active,
                       cmd_freq=int(1/Ts),
                       robot_dyn_model=robot_model,
                       optional_env_params=optional_env_params,
                       reset_simulation_fcn=reset_condition_bigman_drill_gazebo_fcn)
                       # reset_simulation_fcn=reset_condition_bigman_drill_gazebo)

action_dim = bigman_env.get_action_dim()
state_dim = bigman_env.get_state_dim()
observation_dim = bigman_env.get_obs_dim()

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
    'iterations': 20,#5000,  # Number of iterations per inner iteration (Default:5000). Recommended: 1000?
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
act_cost = {
    'type': CostAction,
    'wu': np.ones(action_dim) * 1e-4,
    'target': None,   # Target action value
}

# State Cost
target_distance_hand = np.zeros(6)
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

fk_cost = {
    'type': CostFK,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_hand,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': bigman_env.get_state_info(name='distance_hand')['idx'],
    'op_point_name': hand_name,
    'op_point_offset': hand_offset,
    'joints_idx': bigman_env.get_state_info(name='link_position')['idx'],
    'joint_ids': bigman_params['joint_ids'][body_part_active],
    'robot_model': robot_model,
    # 'wp': np.array([1.0, 1.0, 1.0, 0.7, 0.8, 0.6]),  # one dim less because 'quat' error | 1)orient 2)pos
    'wp': np.array([1.0, 1.0, 1.0, 6.0, 6.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'evalnorm': evall1l2term,
    'l1': 1.0,  # 1.0,  # 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 1.0,  # 1.0,  #1.0e-3,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-2,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 1,  # 10
}

fk_l1_cost = {
    'type': CostFK,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_hand,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': bigman_env.get_state_info(name='distance_hand')['idx'],
    'op_point_name': hand_name,
    'op_point_offset': hand_offset,
    'joints_idx': bigman_env.get_state_info(name='link_position')['idx'],
    'joint_ids': bigman_params['joint_ids'][body_part_active],
    'robot_model': robot_model,
    # 'wp': np.array([1.0, 1.0, 1.0, 0.7, 0.8, 0.6]),  # one dim less because 'quat' error | 1)orient 2)pos
    'wp': np.array([1.0, 1.0, 1.0, 6.0, 6.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'evalnorm': evall1l2term,
    'l1': 1.0,  # 1.0,  # 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 0.0,  # 1.0,  #1.0e-3,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-2,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 1,  # 10
}

fk_l2_cost = {
    'type': CostFK,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_hand,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': bigman_env.get_state_info(name='distance_hand')['idx'],
    'op_point_name': hand_name,
    'op_point_offset': hand_offset,
    'joints_idx': bigman_env.get_state_info(name='link_position')['idx'],
    'joint_ids': bigman_params['joint_ids'][body_part_active],
    'robot_model': robot_model,
    # 'wp': np.array([1.0, 1.0, 1.0, 0.7, 0.8, 0.6]),  # one dim less because 'quat' error | 1)orient 2)pos
    'wp': np.array([1.0, 1.0, 1.0, 6.0, 6.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'evalnorm': evall1l2term,
    'l1': 0.0,  # 1.0,  # 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 1.0,  # 1.0,  #1.0e-3,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-2,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 1,  # 10
}

fk_final_cost = {
    'type': CostFK,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_hand,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': bigman_env.get_state_info(name='distance_hand')['idx'],
    'op_point_name': hand_name,
    'op_point_offset': hand_offset,
    'joints_idx': bigman_env.get_state_info(name='link_position')['idx'],
    'joint_ids': bigman_params['joint_ids'][body_part_active],
    'robot_model': robot_model,
    'wp': np.array([1.0, 1.0, 1.0, 8.0, 10.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'evalnorm': evall1l2term,
    'l1': 0.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 1.0,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 10,
}

fk_l1_final_cost = {
    'type': CostFK,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_hand,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': bigman_env.get_state_info(name='distance_hand')['idx'],
    'op_point_name': hand_name,
    'op_point_offset': hand_offset,
    'joints_idx': bigman_env.get_state_info(name='link_position')['idx'],
    'joint_ids': bigman_params['joint_ids'][body_part_active],
    'robot_model': robot_model,
    'wp': np.array([1.0, 1.0, 1.0, 8.0, 10.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'evalnorm': evall1l2term,
    'l1': 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 0.0,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 10,
}

fk_l2_final_cost = {
    'type': CostFK,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_hand,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': bigman_env.get_state_info(name='distance_hand')['idx'],
    'op_point_name': hand_name,
    'op_point_offset': hand_offset,
    'joints_idx': bigman_env.get_state_info(name='link_position')['idx'],
    'joint_ids': bigman_params['joint_ids'][body_part_active],
    'robot_model': robot_model,
    'wp': np.array([1.0, 1.0, 1.0, 8.0, 10.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'evalnorm': evall1l2term,
    'l1': 0.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 1.0,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 10,
}

cost_sum = {
    'type': CostSum,
    # 'costs': [act_cost, state_cost_distance],
    # 'weights': [1.0e-2, 1.0e-0],
    # 'costs': [act_cost, LAfk_cost, RAfk_cost, state_cost],
    # 'weights': [1.0e-2, 1.0e-0, 1.0e-0, 5.0e-1],
    #'costs': [act_cost, LAfk_cost, LAfk_final_cost],
    #'weights': [1.0e-1, 1.0e-0, 1.0e-0],
    'costs': [act_cost, fk_l1_cost, fk_l2_cost, fk_l1_final_cost, fk_l2_final_cost],
    'weights': [1.0e-1, 0.0e-0, 1.0e-0, 1.0e-0, 1.0e-0],
    # 'costs': [act_cost, state_cost],#, LAfk_cost, RAfk_cost],
    # 'weights': [0.1, 5.0],
}


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
drill_pose0 = drill_relative_pose.copy()
condition0 = create_bigman_drill_condition(q0, drill_pose0, bigman_env.get_state_info(),
                                         joint_idxs=bigman_params['joint_ids'][body_part_sensed])
bigman_env.add_condition(condition0)
reset_condition_bigman_drill_gazebo_fcn.add_reset_poses(drill_pose0)

# #q1 = np.zeros(31)
# q1 = q0.copy()
# q1[15] = np.deg2rad(25)
# q1[16] = np.deg2rad(40)
# q1[18] = np.deg2rad(-45)
# q1[20] = np.deg2rad(-5)
# q1[24] = np.deg2rad(25)
# q1[25] = np.deg2rad(-40)
# q1[27] = np.deg2rad(-45)
# q1[29] = np.deg2rad(-5)
# drill_pose1 = create_drill_relative_pose(drill_x=drill_x+0.02, drill_y=drill_y+0.02, drill_z=drill_z, drill_yaw=drill_yaw+5)
# condition1 = create_bigman_drill_condition(q1, drill_pose1, bigman_env.get_state_info(),
#                                          joint_idxs=bigman_params['joint_ids'][body_part_sensed])
# bigman_env.add_condition(condition1)
# reset_condition_bigman_drill_gazebo_fcn.add_reset_poses(drill_pose1)

# q2 = q0.copy()
# q2[15] = np.deg2rad(25)
# q2[16] = np.deg2rad(30)
# q2[18] = np.deg2rad(-50)
# q2[21] = np.deg2rad(-45)
# q2[24] = np.deg2rad(25)
# q2[25] = np.deg2rad(-30)
# q2[27] = np.deg2rad(-50)
# q2[30] = np.deg2rad(-45)
# drill_pose2 = create_drill_relative_pose(drill_x=drill_x-0.02, drill_y=drill_y-0.02, drill_z=drill_z, drill_yaw=drill_yaw-5)
# condition2 = create_bigman_drill_condition(q2, drill_pose2, bigman_env.get_state_info(),
#                                          joint_idxs=bigman_params['joint_ids'][body_part_sensed])
# bigman_env.add_condition(condition2)
# reset_condition_bigman_drill_gazebo_fcn.add_reset_poses(drill_pose2)

# q3 = q0.copy()
# q3[15] = np.deg2rad(10)
# q3[16] = np.deg2rad(10)
# q3[18] = np.deg2rad(-35)
# q3[24] = np.deg2rad(10)
# q3[25] = np.deg2rad(-10)
# q3[27] = np.deg2rad(-35)
# drill_pose3 = create_drill_relative_pose(drill_x=drill_x-0.06, drill_y=drill_y, drill_z=drill_z, drill_yaw=drill_yaw+10)
# condition3 = create_bigman_drill_condition(q3, drill_pose3, bigman_env.get_state_info(),
#                                          joint_idxs=bigman_params['joint_ids'][body_part_sensed])
# bigman_env.add_condition(condition3)
# reset_condition_bigman_drill_gazebo_fcn.add_reset_poses(drill_pose3)

# q4 = q0.copy()
# drill_pose4 = create_drill_relative_pose(drill_x=drill_x, drill_y=drill_y, drill_z=drill_z, drill_yaw=drill_yaw-5)
# condition4 = create_bigman_drill_condition(q4, drill_pose4, bigman_env.get_state_info(),
#                                          joint_idxs=bigman_params['joint_ids'][body_part_sensed])
# bigman_env.add_condition(condition4)
# reset_condition_bigman_drill_gazebo_fcn.add_reset_poses(drill_pose4)






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
            'drill_relative_pose': drill_relative_pose,
            'drill_size': drill_size,
            'final_drill_height': final_drill_height,
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
resume_training_itr = None  # Resume from previous training iteration
data_files_dir = None  # 'GPS_2017-08-04_09:40:59'  # In case we want to resume from previous training

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
                'covariance_damping': 2.0,  # If greater than zero, covariance is computed as a multiple of the old
                                            # covariance. Multiplier is taken to the power (1 / covariance_damping).
                                            # If greater than one, slows down convergence and keeps exploration noise
                                            # high for more iterations.
                'min_temperature': 0.001,  # Minimum bound of the temperature optimization for the soft-max
                                           # probabilities of the policy samples.
                'use_sumexp': False,
                'pi2_use_dgd_eta': False,
                'pi2_cons_per_step': True,
                }

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
                       #'init_var': np.ones(len(bigman_params['joint_ids'][body_part_active]))*0.3e-1,  # Initial variance (Default:10)
                       'init_var': np.array([3.0e-1, 3.0e-1, 3.0e-1, 3.0e-1, 1.0e-1, 1.0e-1, 1.0e-1])*1.0e-00,
                       #'init_var': np.ones(len(bigman_params['joint_ids'][body_part_active])),  # Initial variance (Default:10)
                       # 'init_var': np.array([3.0e-1, 3.0e-1, 3.0e-1, 3.0e-1, 1.0e-1, 1.0e-1, 1.0e-1,
                       #                       3.0e-1, 3.0e-1, 3.0e-1, 3.0e-1, 1.0e-1, 1.0e-1, 1.0e-1])*1.0,  # Initial variance (Default:10)
                       'pos_gains': 0.001,  #np.array([1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 5.0e-2, 5.0e-2, 5.0e-2])*1.0e+1,  # 0.001,  # Position gains (Default:10)
                       'vel_gains_mult': 0.01,  # Velocity gains multiplier on pos_gains
                       'init_action_offset': None,
                       'dJoints': len(bigman_params['joint_ids'][body_part_sensed]),  # Total joints in state
                       'state_to_pd': 'joints',  # Joints
                       'dDistance': 6,
                       }
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

pi2_hyperparams = {'fit_dynamics': False,  # Dynamics fitting is not required for PI2.
                   }


if learning_algorithm.upper() == 'MDGPS':
    gps_algo_hyperparams = mdgps_hyperparams
    traj_opt_method = traj_opt_lqr

elif learning_algorithm.upper() == 'PIGPS':
    mdgps_hyperparams.update(pigps_hyperparams)
    gps_algo_hyperparams = mdgps_hyperparams
    traj_opt_method = traj_opt_pi2

elif learning_algorithm.upper() == 'ILQR':
    gps_algo_hyperparams = ilqr_hyperparams
    traj_opt_method = traj_opt_lqr

elif learning_algorithm.upper() == 'PI2':
    gps_algo_hyperparams = pi2_hyperparams
    traj_opt_method = traj_opt_pi2
else:
    raise AttributeError("Wrong learning algorithm %s" % learning_algorithm.upper())


gps_hyperparams = {
    'T': int(EndTime/Ts),  # Total points
    'dt': Ts,
    'iterations': 22,  # 100  # 2000  # GPS episodes, "inner iterations" --> K iterations
    'test_after_iter': False,  # If test the learned policy after an iteration in the RL algorithm
    'test_samples': 2,  # Samples from learned policy after an iteration PER CONDITION (only if 'test_after_iter':True)
    # Samples
    'num_samples': 2,  # 20  # Samples for exploration trajs --> N samples
    'noisy_samples': True,
    'sample_on_policy': False,  # Whether generate on-policy samples or off-policy samples
    #'noise_var_scale': np.array([5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2]),  # Scale to Gaussian noise: N(0,1)*sqrt(noise_var_scale)
    #'noise_var_scale': np.array([1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1])*10,  # Scale to Gaussian noise: N(0,1)*sqrt(noise_var_scale)
    'smooth_noise': True,  # Apply Gaussian filter to noise generated
    'smooth_noise_var': 5.0e+0,  # Variance to apply to Gaussian Filter
    'smooth_noise_renormalize': True,  # Renormalize smooth noise to have variance=1
    'noise_var_scale': np.ones(len(bigman_params['joint_ids'][body_part_active])),  # Scale to Gaussian noise: N(0, 1)*sqrt(noise_var_scale), only if smooth_noise_renormalize
    'cost': cost_sum,
    # Conditions
    'conditions': len(bigman_env.get_conditions()),  # Total number of initial conditions
    'train_conditions': range(len(bigman_env.get_conditions())),  # Indexes of conditions used for training
    'test_conditions': range(len(bigman_env.get_conditions())),  # Indexes of conditions used for testing
    # KL step (epsilon)
    'kl_step': 0.2,  # Kullback-Leibler step (base_step)
    'min_step_mult': 0.01,  # Min possible value of step multiplier (multiplies kl_step in LQR)
    'max_step_mult': 1.0, #3 # 10.0,  # Max possible value of step multiplier (multiplies kl_step in LQR)
    # Others
    'gps_algo_hyperparams': gps_algo_hyperparams,
    'init_traj_distr': init_traj_distr,
    'fit_dynamics': True,
    'dynamics': learned_dynamics,
    'initial_state_var': 1e-2,# 1e-6,  # Max value for x0sigma in trajectories  # TODO: CHECK THIS VALUE, maybe it is too low
    'traj_opt': traj_opt_method,
    'max_ent_traj': 0.0,  # Weight of maximum entropy term in trajectory optimization
    'data_files_dir': data_files_dir,
}


if learning_algorithm.upper() == 'MDGPS':
    learn_algo = MDGPS(agent=bigman_agent, env=bigman_env, **gps_hyperparams)

elif learning_algorithm.upper() == 'PIGPS':
    learn_algo = PIGPS(agent=bigman_agent, env=bigman_env, **gps_hyperparams)

elif learning_algorithm.upper() == 'ILQR':
    learn_algo = ILQR(agent=bigman_agent, env=bigman_env, **gps_hyperparams)

elif learning_algorithm.upper() == 'PI2':
    learn_algo = PI2(agent=bigman_agent, env=bigman_env, **gps_hyperparams)

print("Learning algorithm: %s OK\n" % type(learn_algo))

# import numpy as np
# dX = bigman_env.get_state_dim()
# dU = bigman_env.get_action_dim()
# dO = bigman_env.get_obs_dim()
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
    #sampler = Sampler(bigman_agent.policy, bigman_env, **sampler_hyperparams)
    sampler = Sampler(learn_algo.cur[0].traj_distr, bigman_env, **sampler_hyperparams)
    print("Sampling from final policy!!!")
    sample_lists = list()
    for cond_idx in conditions_to_sample:
        raw_input("\nSampling %d times from condition %d and with policy:%s (noisy:%s). \n Press a key to continue..." %
              (n_samples, cond_idx, type(bigman_agent.policy), noisy))
        sample_list = sampler.take_samples(n_samples, cond=cond_idx, noisy=noisy)
        # costs = learn_algo._eval_conditions_sample_list_cost([sample_list])
        # # print(costs)
        # # raw_input('pppp')
        # sample_lists.append(sample_list)

    bigman_env.reset(time=1, cond=0)




print("The script has finished!")
os._exit(0)

