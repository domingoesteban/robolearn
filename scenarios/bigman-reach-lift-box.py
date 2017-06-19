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

from robolearn.utils.lift_box_utils import create_box_relative_pose
from robolearn.utils.lift_box_utils import reset_condition_bigman_box_gazebo
from robolearn.utils.lift_box_utils import spawn_box_gazebo
from robolearn.utils.lift_box_utils import create_bigman_box_condition
from robolearn.utils.lift_box_utils import create_ee_relative_pose

from robolearn.utils.robot_model import RobotModel
from robolearn.utils.algos_utils import IterationData
from robolearn.utils.algos_utils import TrajectoryInfo
from robolearn.utils.print_utils import change_print_color

import time


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
# update_frequency = 5
Ts = 0.01
# EndTime = 4  # Using final time to define the horizon
EndTime = 4  # Using final time to define the horizon

# BOX
box_x = 0.75-0.05
box_y = 0.00
box_z = 0.0184
box_yaw = 0  # Degrees
box_size = [0.4, 0.5, 0.3]
box_relative_pose = create_box_relative_pose(box_x=box_x, box_y=box_y, box_z=box_z, box_yaw=box_yaw)

# Robot Conditions


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

                      #{'name': 'ft_left_leg',
                      # 'type': 'ft_sensor',
                      # 'ros_topic': '/xbotcore/bigman/ft/l_leg_ft',
                      # 'fields': ['force', 'torque']},

                      #{'name': 'ft_right_leg',
                      # 'type': 'ft_sensor',
                      # 'ros_topic': '/xbotcore/bigman/ft/r_leg_ft',
                      # 'fields': ['force', 'torque']},

                      #{'name': 'imu1',
                      # 'type': 'imu',
                      # 'ros_topic': '/xbotcore/bigman/imu/imu_link',
                      # 'fields': ['orientation', 'angular_velocity', 'linear_acceleration']},

                      {'name': 'optitrack',
                       'type': 'optitrack',
                       'ros_topic': '/optitrack/relative_poses',
                       'fields': ['position', 'orientation'],
                       'bodies': ['LSoftHand', 'RSoftHand', 'box']},
                      ]

#observation_active = [{'name': 'imu1',
#                       'type': 'imu',
#                       'ros_topic': '/xbotcore/bigman/imu/imu_link',
#                       'fields': ['orientation', 'angular_velocity', 'linear_acceleration']}]

#state_active = [{'name': 'joint_state',
#                 'type': 'joint_state',
#                 'fields': ['link_position', 'link_velocity'],
#                 'joints': bigman_params['joint_ids']['LA']}]  # Value that can be gotten from robot_params['joints_ids']['LA']

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

print("Bigman Environment OK. body_part_active:%s (action_dim=%d). Command_type:%s" % (body_part_active, action_dim, command_type))

#print(bigman_env.get_state_info())
#raw_input("SSS")


# ################# #
# ################# #
# ##### AGENT ##### #
# ################# #
# ################# #
change_print_color.change('CYAN')
print("\nCreating Bigman Agent...")

# Create an Agent
# Agent option
#policy_params = {
#    'network_params': {
#        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES],
#        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES],
#        'sensor_dims': SENSOR_DIMS,
#    },
#    'network_model': tf_network,
#    'iterations': 1000,
#    'weights_file_prefix': EXP_DIR + 'policy',
#}
policy_params = {
    'network_model': tf_network,  # tf_network, multi_modal_network, multi_modal_network_fp
    'iterations': 100,  # Inner iteration (Default:5000). Reccomended: 1000?
    'network_params': {
        'n_layers': 1,  # Hidden layers??
        'dim_hidden': [40],  # Dictionary of size per n_layers
        'obs_names': bigman_env.get_obs_info()['names'],
        'obs_dof': bigman_env.get_obs_info()['dimensions'],  # DoF for observation data tensor
        'batch_size': 15,  # TODO: Check if this value is OK (same than name_samples)
        #'num_filters': [5, 10],
        #'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, RGB_IMAGE],  # Deprecated from original GPS code
        #'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES],  # Deprecated from original GPS code
        #'obs_image_data': [RGB_IMAGE],  # Deprecated from original GPS code
        #'sensor_dims': SENSOR_DIMS,  # Deprecated from original GPS code
        #'image_width': IMAGE_WIDTH (80),  # For multi_modal_network
        #'image_height': IMAGE_HEIGHT (64),  # For multi_modal_network
        #'image_channels': IMAGE_CHANNELS (3),  # For multi_modal_network
    }
}

policy_opt = {
    'type': PolicyOptTf,
    'hyperparams': policy_params
    }
#policy = None
bigman_agent = GPSAgent(act_dim=action_dim, obs_dim=observation_dim, state_dim=state_dim, policy_opt=policy_opt)
# Load previous learned variables
#bigman_agent.load(file_save_restore)
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
    #'l1': 1e-3,
    #'alpha': 1e-2,
    'target': None,   # Target action value
}

# State Cost

#target_state = left_ee_pose + box_relative_pose
target_state = box_relative_pose
# 'B' pose
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
            'average': None,  #(12, 3),
            'data_idx': bigman_env.get_state_info(name='optitrack')['idx']
        }
    },
}
#state_cost = {
#    'type': CostState,
#    'ramp_option': RAMP_QUADRATIC,  # How target cost ramps over time. RAMP_* :CONSTANT,LINEAR, QUADRATIC, FINAL_ONLY
#    'l1': 0.0,
#    'l2': 1.0,
#    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
#    'data_types': {
#        'link_position': {
#            'wp': np.ones_like(target_pos),  # State weights - must be set.
#            'target_state': target_pos,  # Target state - must be set.
#            'average': None,  #(12, 3),
#            'data_idx': bigman_env.get_state_info(name='link_position')['idx']
#        },
#        'link_velocity': {
#            'wp': np.ones_like(target_vel),  # State weights - must be set.
#            'target_state': target_vel,  # Target state - must be set.
#            'average': None,  #(12, 3),
#            'data_idx': bigman_env.get_state_info(name='link_velocity')['idx']
#        },
#    },
#}

# Robot Model
robot_urdf = os.environ["ROBOTOLOGY_ROOT"]+'/configs/ADVR_shared/bigman/urdf/bigman.urdf'
robot_model = RobotModel(robot_urdf)
LH_name = 'LWrMot3'
RH_name = 'RWrMot3'
l_soft_hand_offset = np.array([0.000, -0.030, -0.210])
r_soft_hand_offset = np.array([0.000, 0.030, -0.210])

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

# Sum of costs
#cost_sum = {
#    'type': CostSum,
#    'costs': [act_cost, state_cost],#, LAfk_cost, RAfk_cost],
#    'weights': [0.1, 5.0],
#}
cost_sum = {
    'type': CostSum,
    'costs': [act_cost, state_cost, LAfk_cost, RAfk_cost],
    'weights': [0.1, 5.0, 8.0, 8.0],
}

# ################################ #
# ################################ #
# ## SAMPLE FROM DEMONSTRATIONS ## #
# ################################ #
# ################################ #
#change_print_color.change('GREEN')
#n_samples = 4
#noisy = True
#sampler_hyperparams = {
#    'noisy': noisy,
#    'noise_var_scale': 0.0001,  # It can be a np.array() with dim=dU
#    'smooth_noise': False,
#    'smooth_noise_var': 0.01,#01
#    'smooth_noise_renormalize': False,
#    'T': int(EndTime/Ts),  # Total points
#    'dt': Ts
#    }
##traj_files = ['trajectories/traj1'+'_x'+str(box_x)+'_y'+str(box_y)+'_Y'+str(box_yaw)+'_m0_reach.npy',
##              'trajectories/traj1'+'_x'+str(box_x)+'_y'+str(box_y)+'_Y'+str(box_yaw)+'_m1_lift.npy']
#traj_files = ['trajectories/traj1'+'_x'+str(box_x)+'_y'+str(box_y)+'_Y'+str(box_yaw)+'_m0_reach.npy']
#traj_rep_policy = TrajectoryReproducerPolicy(traj_files, act_idx=bigman_params['joint_ids']['BA'])
#sampler = Sampler(traj_rep_policy, bigman_env, **sampler_hyperparams)
#condition = create_bigman_box_condition(traj_rep_policy.eval(t=0), box_relative_pose)
#cond_idx = bigman_env.add_condition(condition)
#print(cond_idx)
##raw_input("Press a key for sampling from Sampler")
#print("\nSampling %d times from condition%d and with policy:%s (noisy:%s)" % (n_samples, cond_idx,
#                                                                              type(traj_rep_policy), noisy))
#sampler.take_samples(n_samples, cond=cond_idx, noisy=noisy)

# ########## #
# Conditions #
# ########## #
q0 = np.zeros(31)
condition0 = create_bigman_box_condition(q0, box_relative_pose, joint_idxs=bigman_params['joint_ids']['BA'])
bigman_env.add_condition(condition0)

#q1 = q0.copy()
#q1[16] = np.deg2rad(50)
#q1[25] = np.deg2rad(-50)
#condition1 = create_bigman_box_condition(q1, box_relative_pose, joint_idxs=bigman_params['joint_ids']['BA'])
#bigman_env.add_condition(condition1)


# ######################## #
# ######################## #
# ## LEARNING ALGORITHM ## #
# ######################## #
# ######################## #
change_print_color.change('YELLOW')
print("\nConfiguring learning algorithm...\n")

# Learning params
total_episodes = 10
num_samples = 4  # Samples for exploration trajs
resume_training_itr = None  #10 - 1  # Resume from previous training iteration
data_files_dir = None  #'./GPS_2017-06-15_14:56:13'
T = int(EndTime/Ts)  # Total points
conditions = len(bigman_env.get_conditions())  # Total number of initial conditions
train_conditions = range(conditions)  # Indexes of conditions used for training
test_conditions = train_conditions  # Indexes of conditions used for testing
sample_on_policy = False
test_policy_after_iter = False
kl_step = 0.2

# init_traj_distr is a list of dict
init_traj_distr = {'type': init_lqr,
                   'init_var': 1.0,
                   'stiffness': 1.0,
                   'stiffness_vel': 0.5,
                   'final_weight': 10.0,  # Multiplies cost at T
                   # Parameters for guessing dynamics
                   'init_acc': np.zeros(action_dim),  # dU vector(np.array) of accelerations, default zeros.
                   'init_gains': 1*np.ones(action_dim),  # dU vector(np.array) of gains, default ones.
                  }
#init_traj_distr = [{'type': init_pd,
#                    'init_var': 0.00001,  # initial variance (Default:10)
#                    'pos_gains': 0.001,  # position gains (Default:10)
#                    'vel_gains_mult': 0.01,  # velocity gains multiplier on pos_gains
#                    'init_action_offset': None,
#                   }]

learned_dynamics = {'type': DynamicsLRPrior,
                    'regularization': 1e-6,
                    'prior': {
                        'type': DynamicsPriorGMM,
                        'max_clusters': 20,
                        'min_samples_per_cluster': 40,
                        'max_samples': 20,
                    },
                    }

#gps_algo = 'pigps'
## PIGPS hyperparams
#gps_algo_hyperparams = {'init_pol_wt': 0.01,
#                        'policy_sample_mode': 'add'
#                        }
gps_algo = 'mdgps'
# MDGPS hyperparams
gps_algo_hyperparams = {'init_pol_wt': 0.01,
                        'policy_sample_mode': 'add',
                        # Whether to use 'laplace' or 'mc' cost in step adjusment
                        'step_rule': 'laplace',
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

# Learn using learning algorithm
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
        'smooth_noise': False,
        'smooth_noise_var': 0.01,#01
        'smooth_noise_renormalize': False,
        'T': int(EndTime/Ts),  # Total points
        'dt': Ts
        }
    sampler = Sampler(bigman_agent.policy, bigman_env, **sampler_hyperparams)
    print("Sampling from final policy!!!")
    for cond_idx in conditions_to_sample:
        raw_input("\nSampling %d times from condition%d and with policy:%s (noisy:%s). \n Press a key to continue..." %
                  (n_samples, cond_idx, type(bigman_agent.policy), noisy))
        sampler.take_samples(n_samples, cond=cond_idx, noisy=noisy)

print("The script has finished!")
os._exit(0)

