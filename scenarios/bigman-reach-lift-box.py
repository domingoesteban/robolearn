from __future__ import print_function
from builtins import input

import sys
import os
import signal
import numpy as np
import matplotlib.pyplot as plt

from robolearn.utils.iit.iit_robots_params import bigman_params
from robolearn.envs import BigmanEnv
from robolearn.agents import GPSAgent

from robolearn.policies.policy_opt.policy_opt_tf import PolicyOptTf
from robolearn.policies.policy_opt.tf_model_example import tf_network

from robolearn.utils.sample import Sample
from robolearn.utils.sample_list import SampleList

from robolearn.costs.cost_action import CostAction
from robolearn.costs.cost_state import CostState
from robolearn.costs.cost_fk import CostFK
from robolearn.costs.cost_sum import CostSum
from robolearn.costs.cost_utils import RAMP_QUADRATIC

from robolearn.utils.dynamics.dynamics_lr_prior import DynamicsLRPrior
from robolearn.utils.dynamics.dynamics_prior_gmm import DynamicsPriorGMM

from robolearn.algos.gps.gps import GPS
from robolearn.policies.lin_gauss_init import init_lqr, init_pd
from robolearn.policies.policy_prior import PolicyPrior  # For MDGPS

from robolearn.utils.sampler import Sampler
from robolearn.policies.traj_reprod_policy import TrajectoryReproducerPolicy
from robolearn.utils.joint_space_control_sampler import JointSpaceControlSampler
from robolearn.policies.computed_torque_policy import ComputedTorquePolicy

from robolearn.utils.lift_box_utils import create_box_relative_pose
from robolearn.utils.lift_box_utils import reset_condition_bigman_box_gazebo
from robolearn.utils.lift_box_utils import spawn_box_gazebo
from robolearn.utils.lift_box_utils import create_bigman_box_condition
from robolearn.utils.lift_box_utils import create_ee_relative_pose
from robolearn.utils.lift_box_utils import generate_reach_joints_trajectories
from robolearn.utils.lift_box_utils import generate_lift_joints_trajectories

from robolearn.utils.robot_model import RobotModel
from robolearn.utils.algos_utils import IterationData
from robolearn.utils.algos_utils import TrajectoryInfo
from robolearn.utils.print_utils import change_print_color
from robolearn.utils.plot_utils import plot_joint_info

import time

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
Tlift = 0
# EndTime = 4  # Using final time to define the horizon
EndTime = Treach + Tlift  # Using final time to define the horizon

# BOX
box_x = 0.75-0.05
box_y = 0.00
box_z = 0.0184
box_yaw = 0  # Degrees
box_size = [0.4, 0.5, 0.3]
box_relative_pose = create_box_relative_pose(box_x=box_x, box_y=box_y, box_z=box_z, box_yaw=box_yaw)

# Robot Model (It is used to calculate the IK cost)
robot_urdf_file = os.environ["ROBOTOLOGY_ROOT"]+'/configs/ADVR_shared/bigman/urdf/bigman.urdf'
robot_model = RobotModel(robot_urdf_file)
LH_name = 'LWrMot3'
RH_name = 'RWrMot3'
l_soft_hand_offset = np.array([0.000, -0.030, -0.210])
r_soft_hand_offset = np.array([0.000, 0.030, -0.210])


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
command_type = 'effort'
file_save_restore = "models/bigman_agent_vars.ckpt"

observation_active = [{'name': 'joint_state',
                       'type': 'joint_state',
                       'ros_topic': '/xbotcore/bigman/joint_states',
                       'fields': ['link_position', 'link_velocity', 'effort'],
                       'joints': bigman_params['joint_ids']['UB']},

                      {'name': 'ft_left_arm',
                       'type': 'ft_sensor',
                       'ros_topic': '/xbotcore/bigman/ft/l_arm_ft',
                       'fields': ['force', 'torque']},

                      {'name': 'ft_right_arm',
                       'type': 'ft_sensor',
                       'ros_topic': '/xbotcore/bigman/ft/r_arm_ft',
                       'fields': ['force', 'torque']},

                      # {'name': 'ft_left_leg',
                      #  'type': 'ft_sensor',
                      #  'ros_topic': '/xbotcore/bigman/ft/l_leg_ft',
                      #  'fields': ['force', 'torque']},

                      # {'name': 'ft_right_leg',
                      #  'type': 'ft_sensor',
                      #  'ros_topic': '/xbotcore/bigman/ft/r_leg_ft',
                      #  'fields': ['force', 'torque']},

                      # {'name': 'imu1',
                      #  'type': 'imu',
                      #  'ros_topic': '/xbotcore/bigman/imu/imu_link',
                      #  'fields': ['orientation', 'angular_velocity', 'linear_acceleration']},

                      {'name': 'optitrack',
                       'type': 'optitrack',
                       'ros_topic': '/optitrack/relative_poses',
                       'fields': ['position', 'orientation'],
                       'bodies': ['LSoftHand', 'RSoftHand', 'box']},
                      ]

state_active = [{'name': 'joint_state',
                 'type': 'joint_state',
                 'fields': ['link_position', 'link_velocity'],
                 'joints': bigman_params['joint_ids']['BA']},

                {'name': 'optitrack',
                 'type': 'optitrack',
                 'fields': ['position', 'orientation'],
                 'bodies': ['box']}]  # check if it is better relative position with EE(EEs)


# Spawn Box first because it is simulation
spawn_box_gazebo(box_relative_pose, box_size=box_size)

# Create a Bigman robot ROS EnvInterface
bigman_env = BigmanEnv(interface=interface, mode='simulation',
                       body_part_active=body_part_active, command_type=command_type,
                       observation_active=observation_active,
                       state_active=state_active,
                       cmd_freq=int(1/Ts),
                       reset_simulation_fcn=reset_condition_bigman_box_gazebo)

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
    'iterations': 1000,  # Inner iteration (Default:5000). Recommended: 1000?
    'network_params': {
        'n_layers': 1,  # Hidden layers??
        'dim_hidden': [40],  # List of size per n_layers
        'obs_names': bigman_env.get_obs_info()['names'],
        'obs_dof': bigman_env.get_obs_info()['dimensions'],  # DoF for observation data tensor
        'batch_size': 15,  # TODO: Check if this value is OK (same than n_samples?)
        # 'num_filters': [5, 10],
        # 'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, RGB_IMAGE],  # Deprecated from original GPS code
        # 'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES],  # Deprecated from original GPS code
        # 'obs_image_data': [RGB_IMAGE],  # Deprecated from original GPS code
        # 'sensor_dims': SENSOR_DIMS,  # Deprecated from original GPS code
        # 'image_width': IMAGE_WIDTH (80),  # For multi_modal_network
        # 'image_height': IMAGE_HEIGHT (64),  # For multi_modal_network
        # 'image_channels': IMAGE_CHANNELS (3),  # For multi_modal_network
    }
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
    # 'l1': 1e-3,
    # 'alpha': 1e-2,
    'target': None,   # Target action value
}

# State Cost
target_state = box_relative_pose
state_cost = {
    'type': CostState,
    'ramp_option': RAMP_QUADRATIC,  # How target cost ramps over time. RAMP_* :CONSTANT,LINEAR, QUADRATIC, FINAL_ONLY
    'l1': 0.0,
    'l2': 1.0,
    'wp_final_multiplier': 5.0,  # Weight multiplier on final time step.
    'data_types': {
        'optitrack': {
            'wp': np.ones_like(target_state),  # State weights - must be set.
            'target_state': target_state,  # Target state - must be set.
            'average': None,  # (12, 3),
            'data_idx': bigman_env.get_state_info(name='optitrack')['idx']
        },
        # 'link_position': {
        #     'wp': np.ones_like(target_pos),  # State weights - must be set.
        #     'target_state': target_pos,  # Target state - must be set.
        #     'average': None,  #(12, 3),
        #     'data_idx': bigman_env.get_state_info(name='link_position')['idx']
        # },
        # 'link_velocity': {
        #     'wp': np.ones_like(target_vel),  # State weights - must be set.
        #     'target_state': target_vel,  # Target state - must be set.
        #     'average': None,  #(12, 3),
        #     'data_idx': bigman_env.get_state_info(name='link_velocity')['idx']
        # },
    },
}

left_ee_pose = create_ee_relative_pose(box_relative_pose, ee_x=0, ee_y=box_size[1]/2-0.02, ee_z=0, ee_yaw=0)
LAfk_cost = {
    'type': CostFK,
    'ramp_option': RAMP_QUADRATIC,  # How target cost ramps over time. RAMP_* :CONSTANT,LINEAR, QUADRATIC, FINAL_ONLY
    'target_end_effector': left_ee_pose,
    'end_effector_name': LH_name,
    'end_effector_offset': l_soft_hand_offset,
    'joint_ids': bigman_params['joint_ids']['BA'],
    'robot_model': robot_model,
    'wp': np.array([1.2, 0, 0.8, 1, 1.2, 0.8]),  # one less because 'quat' error | 1)orient 2)pos
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'state_idx': bigman_env.get_state_info(name='link_position')['idx']
}

right_ee_pose = create_ee_relative_pose(box_relative_pose, ee_x=0, ee_y=-box_size[1]/2+0.02, ee_z=0, ee_yaw=0)
RAfk_cost = {
    'type': CostFK,
    'ramp_option': RAMP_QUADRATIC,  # How target cost ramps over time. RAMP_* :CONSTANT,LINEAR, QUADRATIC, FINAL_ONLY
    'target_end_effector': right_ee_pose,
    'end_effector_name': RH_name,
    'end_effector_offset': r_soft_hand_offset,
    'joint_ids': bigman_params['joint_ids']['BA'],
    'robot_model': robot_model,
    'wp': np.array([1.2, 0, 0.8, 1, 1.2, 0.8]),  # one less because 'quat' error | 1)orient 2)pos
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'state_idx': bigman_env.get_state_info(name='link_position')['idx']
}

cost_sum = {
    'type': CostSum,
    'costs': [act_cost, state_cost, LAfk_cost, RAfk_cost],
    'weights': [0.1, 5.0, 8.0, 8.0],
    # 'costs': [act_cost, state_cost],#, LAfk_cost, RAfk_cost],
    # 'weights': [0.1, 5.0],
}


# ########## #
# ########## #
# Conditions #
# ########## #
# ########## #
q0 = np.zeros(31)
condition0 = create_bigman_box_condition(q0, box_relative_pose, joint_idxs=bigman_params['joint_ids']['BA'])
bigman_env.add_condition(condition0)

# q1 = q0.copy()
# q1[16] = np.deg2rad(50)
# q1[25] = np.deg2rad(-50)
# condition1 = create_bigman_box_condition(q1, box_relative_pose, joint_idxs=bigman_params['joint_ids']['BA'])
# bigman_env.add_condition(condition1)


# ################################ #
# ################################ #
# ## SAMPLE FROM DEMONSTRATIONS ## #
# ################################ #
# ################################ #
change_print_color.change('GREEN')
n_samples = 4
noisy = False
sampler_hyperparams = {
    'noisy': noisy,
    'noise_var_scale': 0.001,  # It can be a np.array() with dim=dU
    'smooth_noise': False,  # Whether or not to perform smoothing of noise
    'smooth_noise_var': 0.0001,  # If smooth=True, applies a Gaussian filter with this variance. E.g. 0.01
    'smooth_noise_renormalize': False,  # If smooth=True, renormalizes data to have variance 1 after smoothing.
    'T': int(EndTime/Ts),  # Total points
    'dt': Ts
    }

# # Generate joints trajectories
# print("Generating joints trajectories..")
# # Expand conditions:
# init_cond = bigman_env.get_conditions()
# joints_trajectories = list()
# joint_idx = bigman_params['joint_ids']['BA']
# state_info = bigman_env.get_state_info()
# q0 = np.zeros(robot_model.q_size)
# for cond in init_cond:
#     if Treach > 0:
#         arms_des = cond[state_info['idx'][state_info['names'].index('link_position')]]
#         q0[joint_idx] = arms_des
#         qs_reach, qdots_reach, qddots_reach = generate_reach_joints_trajectories(box_relative_pose, box_size, Treach,
#                                                                                  q0, option=0, dt=Ts)
#         #joints_to_plot = bigman_params['joint_ids']['LA']
#         #joint_names = [bigman_params['joints_names'][idx] for idx in joints_to_plot]
#         #plot_joint_info(joints_to_plot, qs_reach, joint_names, data='position', block=False)
#         #plot_joint_info(joints_to_plot, qdots_reach, joint_names, data='velocity', block=False)
#         #plot_joint_info(joints_to_plot, qddots_reach, joint_names, data='acceleration', block=False)
#         #input("Plotting reaching... Press a key to continue")
#     if Tlift > 0:
#         qs_lift, qdots_lift, qddots_lift = generate_lift_joints_trajectories(box_relative_pose, box_size, Tlift, q0,
#                                                                              option=0, dt=Ts)
#         #joints_to_plot = bigman_params['joint_ids']['LA']
#         #joint_names = [bigman_params['joints_names'][idx] for idx in joints_to_plot]
#         #plot_joint_info(joints_to_plot, qs_lift, joint_names, data='position', block=False)
#         #plot_joint_info(joints_to_plot, qdots_lift, joint_names, data='velocity', block=False)
#         #plot_joint_info(joints_to_plot, qddots_lift, joint_names, data='acceleration', block=False)
#         #input("Plotting lifting... Press a key to continue")
#     # Concatenate reach and lift trajectories
#     if Treach > 0 and Tlift > 0:
#         qs = np.r_[qs_reach, qs_lift]
#         qdot_s = np.r_[qdots_reach, qdots_lift]
#         qddot_s = np.r_[qddots_reach, qddots_lift]
#     elif Treach > 0 and not Tlift > 0:
#         qs = qs_reach
#         qdot_s = qdots_reach
#         qddot_s = qddots_reach
#     elif Treach > 0 and not Tlift > 0:
#         qs = qs_lift
#         qdot_s = qdots_lift
#         qddot_s = qddots_lift
#     else:
#         raise ValueError("Both Treach and Tlift not defined!!")
#     joints_trajectories.append([qs, qdot_s, qddot_s])
# computed_torque_policy = ComputedTorquePolicy(robot_model.model)
# sampler_hyperparams['act_idx'] = bigman_params['joint_ids']['BA']
# sampler_hyperparams['joints_idx'] = bigman_params['joint_ids']['BA']
# sampler_hyperparams['state_pos_idx'] = bigman_env.get_state_info(name='link_position')['idx']
# sampler_hyperparams['state_vel_idx'] = bigman_env.get_state_info(name='link_velocity')['idx']
# sampler_hyperparams['q_size'] = robot_model.q_size
# sampler_hyperparams['qdot_size'] = robot_model.qdot_size
# sampler_hyperparams['joints_trajectories'] = joints_trajectories
# sampler = JointSpaceControlSampler(computed_torque_policy, bigman_env, **sampler_hyperparams)
# #input("Press a key for sampling from Sampler")
# for cond_idx, _ in enumerate(init_cond):
#     print("\nSampling %d times from condition%d and with policy:%s (noisy:%s)" % (n_samples, cond_idx,
#                                                                                   type(computed_torque_policy), noisy))
#     sampler.take_samples(n_samples, cond=cond_idx, noisy=noisy, save=False)


# #traj_files = ['trajectories/traj1'+'_x'+str(box_x)+'_y'+str(box_y)+'_Y'+str(box_yaw)+'_m0_reach.npy',
# #              'trajectories/traj1'+'_x'+str(box_x)+'_y'+str(box_y)+'_Y'+str(box_yaw)+'_m1_lift.npy']
# traj_files = ['trajectories/traj1'+'_x'+str(box_x)+'_y'+str(box_y)+'_Y'+str(box_yaw)+'_m0_reach.npy']
# traj_rep_policy = TrajectoryReproducerPolicy(traj_files, act_idx=bigman_params['joint_ids']['BA'])
# sampler = Sampler(traj_rep_policy, bigman_env, **sampler_hyperparams)


# ######################## #
# ######################## #
# ## LEARNING ALGORITHM ## #
# ######################## #
# ######################## #
change_print_color.change('YELLOW')
print("\nConfiguring learning algorithm...\n")

# Learning params
total_episodes = 2000
num_samples = 20  # Samples for exploration trajs
resume_training_itr = None  # 10 - 1  # Resume from previous training iteration
data_files_dir = None  # './GPS_2017-06-15_14:56:13'  # In case we want to resume from previous training
T = int(EndTime/Ts)  # Total points
conditions = len(bigman_env.get_conditions())  # Total number of initial conditions
train_conditions = range(conditions)  # Indexes of conditions used for training
test_conditions = train_conditions  # Indexes of conditions used for testing
sample_on_policy = False  # Whether generate on-policy samples or off-policy samples
test_policy_after_iter = True  # If test the learned policy after an iteration in the RL algorithm
kl_step = 0.2  # Kullback-Leibler step

# init_traj_distr values can be lists if they are different for each condition
init_traj_distr = {'type': init_lqr,
                   'init_var': 1.0,
                   'stiffness': 1.0,
                   'stiffness_vel': 0.5,
                   'final_weight': 10.0,  # Multiplies cost at T
                   # Parameters for guessing dynamics
                   'init_acc': np.zeros(action_dim),  # dU vector(np.array) of accelerations, default zeros.
                   'init_gains': 1*np.ones(action_dim),  # dU vector(np.array) of gains, default ones.
                   }
# init_traj_distr = [{'type': init_pd,
#                     'init_var': 0.00001,  # initial variance (Default:10)
#                     'pos_gains': 0.001,  # position gains (Default:10)
#                     'vel_gains_mult': 0.01,  # velocity gains multiplier on pos_gains
#                     'init_action_offset': None,
#                    }]

learned_dynamics = {'type': DynamicsLRPrior,
                    'regularization': 1e-6,
                    'prior': {
                        'type': DynamicsPriorGMM,
                        'max_clusters': 20,
                        'min_samples_per_cluster': 40,
                        'max_samples': 20,
                        },
                    }

# gps_algo = 'pigps'
# gps_algo_hyperparams = {'init_pol_wt': 0.01,
#                         'policy_sample_mode': 'add'
#                         }
gps_algo = 'mdgps'
gps_algo_hyperparams = {'init_pol_wt': 0.01,
                        'policy_sample_mode': 'add',
                        'step_rule': 'laplace',  # Whether to use 'laplace' or 'mc' cost in step adjusment
                        'policy_prior': {'type': PolicyPrior},
                        }
learn_algo = GPS(agent=bigman_agent, env=bigman_env,
                 iterations=total_episodes, num_samples=num_samples,
                 T=T, dt=Ts,
                 cost=cost_sum,
                 conditions=conditions,
                 gps_algo=gps_algo,
                 gps_algo_hyperparams=gps_algo_hyperparams,
                 train_conditions=train_conditions,
                 test_conditions=test_conditions,
                 sample_on_policy=sample_on_policy,
                 test_after_iter=test_policy_after_iter,
                 init_traj_distr=init_traj_distr,
                 dynamics=learned_dynamics,
                 kl_step=kl_step,
                 data_files_dir=data_files_dir
                 )
print("Learning algorithm: %s OK\n" % type(learn_algo))

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
    conditions_to_sample = [0]
    change_print_color.change('GREEN')
    n_samples = 4
    noisy = False
    sampler_hyperparams = {
        'noisy': noisy,
        'noise_var_scale': 0.0001,  # It can be a np.array() with dim=dU
        'smooth_noise': False,  # Whether or not to perform smoothing of noise
        'smooth_noise_var': 0.01,   # If smooth=True, applies a Gaussian filter with this variance. E.g. 0.01
        'smooth_noise_renormalize': False,  # If smooth=True, renormalizes data to have variance 1 after smoothing.
        'T': int(EndTime/Ts),  # Total points
        'dt': Ts
        }
    sampler = Sampler(bigman_agent.policy, bigman_env, **sampler_hyperparams)
    print("Sampling from final policy!!!")
    for cond_idx in conditions_to_sample:
        input("\nSampling %d times from condition%d and with policy:%s (noisy:%s). \n Press a key to continue..." %
              (n_samples, cond_idx, type(bigman_agent.policy), noisy))
        sampler.take_samples(n_samples, cond=cond_idx, noisy=noisy)

print("The script has finished!")
os._exit(0)

