from __future__ import print_function

import os
import random
import signal

import numpy as np
from robolearn.old_utils.sampler import Sampler

from robolearn.old_agents import GPSAgent
from robolearn.old_algos.gps.mdgps import MDGPS
from robolearn.old_algos.gps.pigps import PIGPS
from robolearn.old_algos.trajopt.dreps import DREPS
from robolearn.old_algos.trajopt.ilqr import ILQR
from robolearn.old_algos.trajopt.mdreps import MDREPS
from robolearn.old_algos.trajopt.pi2 import PI2
from robolearn.old_costs.cost_action import CostAction
from robolearn.old_costs.cost_fk import CostFK
from robolearn.old_costs.cost_state import CostState
from robolearn.old_costs.cost_sum import CostSum
from robolearn.old_costs.cost_utils import RAMP_FINAL_ONLY, RAMP_CONSTANT
from robolearn.old_costs.cost_utils import evall1l2term
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
from robolearn.old_utils.tasks.bigman.reach_drill_utils import Reset_condition_bigman_drill_gazebo
from robolearn.old_utils.tasks.bigman.reach_drill_utils import create_bigman_drill_condition
from robolearn.old_utils.tasks.bigman.reach_drill_utils import create_drill_relative_pose
from robolearn.old_utils.tasks.bigman.reach_drill_utils import create_hand_relative_pose
from robolearn.old_utils.tasks.bigman.reach_drill_utils import spawn_drill_gazebo
from robolearn.old_utils.tasks.bigman.reach_drill_utils import task_space_torque_control_demos, \
    load_task_space_torque_control_demos
from robolearn.old_utils.tasks.bigman.reach_drill_utils import task_space_torque_control_dual_demos, \
    load_task_space_torque_control_dual_demos
from robolearn.old_utils.traj_opt.traj_opt_dreps import TrajOptDREPS
from robolearn.old_utils.traj_opt.traj_opt_lqr import TrajOptLQR
from robolearn.old_utils.traj_opt.traj_opt_mdreps import TrajOptMDREPS
from robolearn.old_utils.traj_opt.traj_opt_pi2 import TrajOptPI2

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
# Task parameters
Ts = 0.01
Treach = 8
Tlift = 0  # 3.8
Tinter = 0  # 0.5
Tend = 0  # 0.7
# EndTime = 4  # Using final time to define the horizon
EndTime = Treach + Tinter + Tlift + Tend  # Using final time to define the horizon
init_with_demos = False
generate_dual_sets = True
demos_dir = None  # 'TASKSPACE_TORQUE_CTRL_DEMO_2017-07-21_16:32:39'
dual_dir = 'DUAL_DEMOS_2017-08-23_07:10:35'
#seed = 6  previous 04/09/17 17:30 pm
seed = 0

random.seed(seed)
np.random.seed(seed)

# BOX
drill_x = 0.70
drill_y = 0.00
drill_z = -0.1327
drill_yaw = 0  # Degrees
#drill_size = [0.1, 0.1, 0.3]
drill_size = [0.11, 0.11, 0.3]  # Beer
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
    hand_y = -drill_size[1]/2-0.02
    hand_z = drill_size[2]/2+0.02
    hand_name = RH_name
    hand_offset = r_soft_hand_offset
else:
    hand_y = drill_size[1]/2+0.02
    hand_z = drill_size[2]/2+0.02
    hand_name = LH_name
    hand_offset = l_soft_hand_offset

hand_rel_pose = create_hand_relative_pose([0, 0, 0, 1, 0, 0, 0], hand_x=0.0, hand_y=hand_y, hand_z=hand_z, hand_yaw=0)


object_name = 'drill'
object_rel_pose = create_hand_relative_pose([0, 0, 0, 1, 0, 0, 0], hand_x=0.0, hand_y=hand_y, hand_z=hand_z, hand_yaw=0)



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

                      {'name': 'distance_object',
                       'type': 'object_pose',
                       'body_name': object_name,
                       'target_rel_pose': drill_relative_pose,
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

                {'name': 'distance_object',
                 'type': 'object_pose',
                 'body_name': object_name,
                 'target_rel_pose': drill_relative_pose,
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
# target_distance_hand[-2] = -0.02  # Yoffset
# target_distance_hand[-1] = 0.1  # Zoffset

target_distance_object = np.zeros(6)
state_cost_distance = {
    'type': CostState,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'evalnorm': evall1l2term,  # TODO: ALWAYS USE evall1l2term
    'l1': 1.0,  # Weight for l1 norm
    'l2': 0.0,  # Weight for l2 norm
    'alpha': 1e-2,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
    'data_types': {
        'distance_object': {
            # 'wp': np.ones_like(target_state),  # State weights - must be set.
            'wp': np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]),  # State weights - must be set.
            'target_state': target_distance_object,  # Target state - must be set.
            'average': None,  # (12, 3),
            'data_idx': bigman_env.get_state_info(name='distance_object')['idx']
        },
    },
}
state_final_cost_distance = {
    'type': CostState,
    'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'evalnorm': evall1l2term,  # TODO: ALWAYS USE evall1l2term
    'l1': 1.0,  # Weight for l1 norm
    'l2': 0.0,  # Weight for l2 norm
    'alpha': 1e-2,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 10.0,  # Weight multiplier on final time step.
    'data_types': {
        'distance_object': {
            # 'wp': np.ones_like(target_state),  # State weights - must be set.
            'wp': np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]),  # State weights - must be set.
            'target_state': target_distance_object,  # Target state - must be set.
            'average': None,  # (12, 3),
            'data_idx': bigman_env.get_state_info(name='distance_object')['idx']
        },
    },
}

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
    #'evalnorm': evallogl2term,
    'l1': 1.0,  # 1.0,  # 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 1.0,  # 1.0,  #1.0e-3,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
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
    #'evalnorm': evallogl2term,
    'l1': 1.0,  # 1.0,  # 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 0.0,  # 1.0,  #1.0e-3,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
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
    #'evalnorm': evallogl2term,
    'l1': 0.0,  # 1.0,  # 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 1.0,  # 1.0,  #1.0e-3,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 1,  # 10
}

fk_final_cost = {
    'type': CostFK,
    'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_hand,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': bigman_env.get_state_info(name='distance_hand')['idx'],
    'op_point_name': hand_name,
    'op_point_offset': hand_offset,
    'joints_idx': bigman_env.get_state_info(name='link_position')['idx'],
    'joint_ids': bigman_params['joint_ids'][body_part_active],
    'robot_model': robot_model,
    'wp': np.array([1.0, 1.0, 1.0, 10.0, 10.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'evalnorm': evall1l2term,
    #'evalnorm': evallogl2term,
    'l1': 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 1.0,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 50,
}

fk_l1_final_cost = {
    'type': CostFK,
    'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_hand,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': bigman_env.get_state_info(name='distance_hand')['idx'],
    'op_point_name': hand_name,
    'op_point_offset': hand_offset,
    'joints_idx': bigman_env.get_state_info(name='link_position')['idx'],
    'joint_ids': bigman_params['joint_ids'][body_part_active],
    'robot_model': robot_model,
    'wp': np.array([1.0, 1.0, 1.0, 10.0, 10.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'evalnorm': evall1l2term,
    #'evalnorm': evallogl2term,
    'l1': 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 0.0,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 50,
}

fk_l2_final_cost = {
    'type': CostFK,
    'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_hand,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': bigman_env.get_state_info(name='distance_hand')['idx'],
    'op_point_name': hand_name,
    'op_point_offset': hand_offset,
    'joints_idx': bigman_env.get_state_info(name='link_position')['idx'],
    'joint_ids': bigman_params['joint_ids'][body_part_active],
    'robot_model': robot_model,
    'wp': np.array([1.0, 1.0, 1.0, 10.0, 10.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'evalnorm': evall1l2term,
    #'evalnorm': evallogl2term,
    'l1': 0.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 1.0,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 50,
}

cost_sum = {
    'type': CostSum,
    # 'costs': [act_cost, state_cost_distance],
    # 'weights': [1.0e-2, 1.0e-0],
    # 'costs': [act_cost, LAfk_cost, RAfk_cost, state_cost],
    # 'weights': [1.0e-2, 1.0e-0, 1.0e-0, 5.0e-1],
    #'costs': [act_cost, LAfk_cost, LAfk_final_cost],
    #'weights': [1.0e-1, 1.0e-0, 1.0e-0],
    'costs': [act_cost, fk_l1_cost, fk_l2_cost, fk_l1_final_cost, fk_l2_final_cost, state_cost_distance, state_final_cost_distance],
    'weights': [1.0e-1, 1.5e-1, 1.0e-0, 1.5e-1, 1.0e-0, 5.0e+0, 1.0e+1],
    # 'costs': [act_cost, state_cost],#, LAfk_cost, RAfk_cost],
    # 'weights': [0.1, 5.0],
}


# ########## #
# ########## #
# Conditions #
# ########## #
# ########## #
drill_relative_poses = []  # Used only in dual demos

# q0 = np.zeros(31)
# q0[15] = np.deg2rad(25)
# q0[16] = np.deg2rad(40)
# q0[18] = np.deg2rad(-75)
# #q0[15:15+7] = [0.0568,  0.2386, -0.2337, -1.6803,  0.2226,  0.0107,  0.5633]
# q0[24] = np.deg2rad(25)
# q0[25] = np.deg2rad(-40)
# q0[27] = np.deg2rad(-75)
# #q0[24:24+7] = [0.0568,  -0.2386, 0.2337, -1.6803,  -0.2226,  0.0107,  -0.5633]
# drill_pose0 = drill_relative_pose.copy()
# condition0 = create_bigman_drill_condition(q0, drill_pose0, bigman_env.get_state_info(),
#                                          joint_idxs=bigman_params['joint_ids'][body_part_sensed])
# bigman_env.add_condition(condition0)
# reset_condition_bigman_drill_gazebo_fcn.add_reset_poses(drill_pose0)
# drill_relative_poses.append(drill_pose0)

# # q1 = q0.copy()
# q1 = np.zeros(31)
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
# drill_relative_poses.append(drill_pose1)

# # q2 = q0.copy()
# q2 = np.zeros(31)
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
#                                            joint_idxs=bigman_params['joint_ids'][body_part_sensed])
# bigman_env.add_condition(condition2)
# reset_condition_bigman_drill_gazebo_fcn.add_reset_poses(drill_pose2)
# drill_relative_poses.append(drill_pose2)

# q3 = q0.copy()
q3 = np.zeros(31)
q3[15] = np.deg2rad(10)
q3[16] = np.deg2rad(10)
q3[18] = np.deg2rad(-35)
# q3[24] = np.deg2rad(10)
# q3[25] = np.deg2rad(-10)
# q3[27] = np.deg2rad(-35)
# q3[24] = np.deg2rad(-10)
# #q3[25] = np.deg2rad(-20)
# #q3[25] = np.deg2rad(-10)
# q3[25] = np.deg2rad(-30)
# q3[26] = np.deg2rad(0)
# q3[27] = np.deg2rad(-85)
# q3[28] = np.deg2rad(0)
# q3[29] = np.deg2rad(0)
# q3[30] = np.deg2rad(0)
q3[24] = np.deg2rad(20)
q3[25] = np.deg2rad(-55)
q3[26] = np.deg2rad(0)
q3[27] = np.deg2rad(-95)
q3[28] = np.deg2rad(0)
q3[29] = np.deg2rad(0)
q3[30] = np.deg2rad(0)
#drill_pose3 = create_drill_relative_pose(drill_x=drill_x-0.06, drill_y=drill_y, drill_z=drill_z, drill_yaw=drill_yaw+10)
drill_pose3 = create_drill_relative_pose(drill_x=drill_x+0.05, drill_y=drill_y-0.3, drill_z=drill_z, drill_yaw=drill_yaw+10)
condition3 = create_bigman_drill_condition(q3, drill_pose3, bigman_env.get_state_info(),
                                           joint_idxs=bigman_params['joint_ids'][body_part_sensed])
bigman_env.add_condition(condition3)
reset_condition_bigman_drill_gazebo_fcn.add_reset_poses(drill_pose3)
drill_relative_poses.append(drill_pose3)


# # q4 = q0.copy()
# q4 = np.zeros(31)
# drill_pose4 = create_drill_relative_pose(drill_x=drill_x, drill_y=drill_y, drill_z=drill_z, drill_yaw=drill_yaw-5)
# condition4 = create_bigman_drill_condition(q4, drill_pose4, bigman_env.get_state_info(),
#                                          joint_idxs=bigman_params['joint_ids'][body_part_sensed])
# bigman_env.add_condition(condition4)
# reset_condition_bigman_drill_gazebo_fcn.add_reset_poses(drill_pose4)
# drill_relative_poses.append(drill_pose4)



# #################### #
# #################### #
# ## DEMONSTRATIONS ## #
# #################### #
# #################### #
if init_with_demos is True:
    print("")
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

# DUAL SAMPLES
if generate_dual_sets is True:
    print("")
    change_print_color.change('GREEN')
    if dual_dir is None:
        task_space_torque_control_dual_params = {
            'active_joints': 'RA',
            'n_good_samples': 5,
            'n_bad_samples': 5,
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
            'drill_relative_poses': drill_relative_poses,  # THIS
            'drill_relative_pose_cond_id': range(-8, -1),  # OR THIS
            'drill_size': drill_size,
            'final_drill_height': final_drill_height,
            # offsets [roll, pitch, yaw, x, y, z]
            #'good_offsets': [[0, 0, 0, 0.25, -0.25, drill_size[2]/2+0.1],
            'good_offsets': [[-45,      0,      0,      0,      -0.13,      0.17],
                             [0,      0,      0,      0,      -0.12,      0.17],
                             [30,      0,      0,      0,      -0.13,      0.15],
                             [3,      0,      0,      0,      -0.14,      0.15],
                             [-8,      0,      0,      0,      -0.14,      0.14],
                             ],
            'bad_offsets': [[-10,      0,      0,      0.05,      0.1,      0.17],
                            [2,      0,      0,      0,      0.1,      0.10],
                            [25,      0,      -5,      0,      0.0,      0.20],
                            [1,      10,      2,      -0.1,      0.14,      0.21],
                            [3,      10,      40,      0.05,      0.05,      0.18],
                            ],
        }

        good_trajs, bad_trajs = task_space_torque_control_dual_demos(**task_space_torque_control_dual_params)
        bigman_env.reset(time=2, cond=0)

    else:
        good_trajs, bad_trajs = load_task_space_torque_control_dual_demos(dual_dir)
        print('Good/bad dual samples has been obtained from directory %s' % dual_dir)

else:
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
data_files_dir = None  # 'GPS_2017-09-05_13:07:23'  # None  # In case we want to resume from previous training

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
                       #'init_var': np.array([3.0e-1, 3.0e-1, 3.0e-1, 3.0e-1, 2.0e-1, 2.0e-1, 2.0e-1])*1.0e+0,
                       #'init_var': np.ones(7)*0.5,
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
                   'good_const': False,  # Use good constraints
                   'bad_const': False,  # Use bad constraints
                   'del0': 1e-4,  # Eta updates for non-SPD Q-function (non-SPD correction step).
                   'del0_good': 1e-4,  # Omega updates for non-SPD Q-function (non-SPD correction step).
                   'del0_bad': 1e-8,  # Nu updates for non-SPD Q-function (non-SPD correction step).
                   # 'eta_error_threshold': 1e16, # TODO: REMOVE, it is not used
                   'min_eta': 1e-8,  # At min_eta, kl_div > kl_step
                   'max_eta': 1e16,  # At max_eta, kl_div < kl_step
                   'min_omega': 1e-8,  # At min_omega, kl_div > kl_step
                   'max_omega': 1e16,  # At max_omega, kl_div < kl_step
                   'min_nu': 1e-8,  # At min_nu, kl_div > kl_step
                   'max_nu': 2.0e1,  # At max_nu, kl_div < kl_step,
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
                      'base_kl_bad': 2.5,  # (chi) to be used with multiplier | kl_div_b >= kl_bad
                      'base_kl_good': 1.0,  # (xi) to be used with multiplier | kl_div_g <= kl_good
                      'bad_traj_selection_type': 'always',  # 'always', 'only_traj'
                      'good_traj_selection_type': 'always',  # 'always', 'only_traj'
                      'init_eta': 4.62,
                      'init_nu': 0.5,
                      'init_omega': 1.0,
                      'min_bad_mult': 0.01,  # Min possible value of step multiplier (multiplies base_kl_bad in LQR)
                      'max_bad_mult': 20.0,  # Max possible value of step multiplier (multiplies base_kl_bad in LQR)
                      'min_good_mult': 0.01,  # Min possible value of step multiplier (multiplies base_kl_good in LQR)
                      'max_good_mult': 20.0,  # Max possible value of step multiplier (multiplies base_kl_good in LQR)
                      'min_bad_var': np.array([3.0e-1, 3.0e-1, 3.0e-1, 3.0e-1, 1.0e-1, 1.0e-1, 1.0e-1])*1.0e-00,
                      'min_good_var': np.array([3.0e-1, 3.0e-1, 3.0e-1, 3.0e-1, 1.0e-1, 1.0e-1, 1.0e-1])*1.0e-00,
                      'init_pol_wt': 0.01,  # TODO: remove need for init_pol_wt in MDGPS (It should not work with MDGPS)
                      'policy_sample_mode': 'add',
                      'step_rule': 'laplace',  # Whether to use 'laplace' or 'mc' cost in step adjustment
                      'policy_prior': {'type': ConstantPolicyPrior,
                                       'strength': 1e-4,
                                       },
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
    sample_on_policy = False
    test_after_iter = False
    use_global_policy = False
    #use_global_policy = False
else:
    raise AttributeError("Wrong learning algorithm %s" % learning_algorithm.upper())


gps_hyperparams = {
    'T': int(EndTime/Ts),  # Total points
    'dt': Ts,
    'iterations': 100,  # 100  # 2000  # GPS episodes, "inner iterations" --> K iterations
    'test_after_iter': test_after_iter,  # If test the learned policy after an iteration in the RL algorithm
    'test_samples': 3,  # Samples from learned policy after an iteration PER CONDITION (only if 'test_after_iter':True)
    # Samples
    'num_samples': 5,  # 20  # Samples for exploration trajs --> N samples
    'noisy_samples': True,
    'sample_on_policy': sample_on_policy,  # Whether generate on-policy samples or off-policy samples
    #'noise_var_scale': np.array([5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2]),  # Scale to Gaussian noise: N(0,1)*sqrt(noise_var_scale)
    #'noise_var_scale': np.array([1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1])*10,  # Scale to Gaussian noise: N(0,1)*sqrt(noise_var_scale)
    'smooth_noise': True,  # Apply Gaussian filter to noise generated
    #'smooth_noise_var': 5.0e+0,  # np.power(2*Ts, 2), # Variance to apply to Gaussian Filter. In Kumar (2016) paper, it is the std dev of 2 Ts
    'smooth_noise_var': 8.0e+0,  # np.power(2*Ts, 2), # Variance to apply to Gaussian Filter. In Kumar (2016) paper, it is the std dev of 2 Ts
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
    learn_algo = MDGPS(agent=bigman_agent, env=bigman_env, **gps_hyperparams)

elif learning_algorithm.upper() == 'PIGPS':
    learn_algo = PIGPS(agent=bigman_agent, env=bigman_env, **gps_hyperparams)

elif learning_algorithm.upper() == 'ILQR':
    learn_algo = ILQR(agent=bigman_agent, env=bigman_env, **gps_hyperparams)

elif learning_algorithm.upper() == 'PI2':
    learn_algo = PI2(agent=bigman_agent, env=bigman_env, **gps_hyperparams)

elif learning_algorithm.upper() == 'DREPS':
    learn_algo = DREPS(agent=bigman_agent, env=bigman_env, **gps_hyperparams)

elif learning_algorithm.upper() == 'MDREPS':
    learn_algo = MDREPS(agent=bigman_agent, env=bigman_env, **gps_hyperparams)

else:
    raise AttributeError("Wrong learning algorithm %s" % learning_algorithm.upper())

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

