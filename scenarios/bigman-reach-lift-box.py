from __future__ import print_function

import sys
import numpy as np
import matplotlib.pyplot as plt

from robolearn.utils.iit.iit_robots_params import *
from robolearn.envs import BigmanEnv
from robolearn.agents import GPSAgent

from robolearn.policies.policy_opt.policy_opt_tf import PolicyOptTf
from robolearn.policies.policy_opt.tf_model_example import tf_network

from robolearn.utils.sample import Sample
from robolearn.utils.sample_list import SampleList

from robolearn.costs.cost_action import CostAction
from robolearn.costs.cost_state import CostState
from robolearn.costs.cost_sum import CostSum
from robolearn.costs.cost_utils import RAMP_QUADRATIC

from robolearn.utils.algos_utils import IterationData
from robolearn.utils.algos_utils import TrajectoryInfo
from robolearn.algos.gps.gps import GPS
from robolearn.policies.lin_gauss_init import init_lqr, init_pd
from robolearn.policies.policy_prior import PolicyPrior  # For MDGPS

from robolearn.utils.sampler import Sampler
from robolearn.policies.traj_reprod_policy import TrajectoryReproducerPolicy

import tf
from robolearn.utils.transformations import homogeneous_matrix

import rospy
from robolearn.utils.print_utils import *

import time


# ################## #
# ################## #
# ### PARAMETERS ### #
# ################## #
# ################## #
# Task parameters
#update_frequency = 5
Ts = 0.01
EndTime = 4  # Using final time to define the horizon

# BOX
box_position = np.array([0.75-0.05,
                         0.00,
                         0.0184])
box_size = [0.4, 0.5, 0.3]
box_yaw = 0  # Degrees
box_orient = tf.transformations.rotation_matrix(np.deg2rad(box_yaw), [0, 0, 1])
box_matrix = homogeneous_matrix(rot=box_orient, pos=box_position)


# ################### #
# ################### #
# ### ENVIRONMENT ### #
# ################### #
# ################### #

print("\nCreating Bigman environment...")

# Robot configuration
interface = 'ros'
body_part_active = 'UB'
command_type = 'position'
file_save_restore = "models/bigman_agent_vars.ckpt"


observation_active = [{'name': 'joint_state',
                       'type': 'joint_state',
                       'ros_topic': '/xbotcore/bigman/joint_states',
                       'fields': ['link_position', 'link_velocity', 'effort'],
                       'joints': bigman_params['joint_ids']['UB']},  # Value that can be gotten from robot_params['joints_names']['UB']

                      {'name': 'ft_left_arm',
                       'type': 'ft_sensor',
                       'ros_topic': '/xbotcore/bigman/ft/l_arm_ft',
                       'fields': ['force', 'torque']},

                      {'name': 'ft_right_arm',
                       'type': 'ft_sensor',
                       'ros_topic': '/xbotcore/bigman/ft/r_arm_ft',
                       'fields': ['force', 'torque']},

                      {'name': 'ft_left_leg',
                       'type': 'ft_sensor',
                       'ros_topic': '/xbotcore/bigman/ft/l_leg_ft',
                       'fields': ['force', 'torque']},

                      {'name': 'ft_right_leg',
                       'type': 'ft_sensor',
                       'ros_topic': '/xbotcore/bigman/ft/r_leg_ft',
                       'fields': ['force', 'torque']},

                      {'name': 'imu1',
                       'type': 'imu',
                       'ros_topic': '/xbotcore/bigman/imu/imu_link',
                       'fields': ['orientation', 'angular_velocity', 'linear_acceleration']},

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




# Create a Bigman robot ROS EnvInterface
bigman_env = BigmanEnv(interface=interface, mode='simulation',
                       body_part_active=body_part_active, command_type=command_type,
                       observation_active=observation_active,
                       state_active=state_active,
                       cmd_freq=int(1/Ts))

# TODO: DOMINGOOOO
# TODO: Temporally using current state to set one initial condition
current_state = bigman_env.get_state()
bigman_env.set_initial_conditions([current_state])

action_dim = bigman_env.get_action_dim()
state_dim = bigman_env.get_state_dim()
observation_dim = bigman_env.get_obs_dim()

print("Bigman Environment OK. body_part_active:%s (action_dim=%d). Command_type:%s" % (body_part_active, action_dim, command_type))


# ################# #
# ################# #
# ##### AGENT ##### #
# ################# #
# ################# #
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
    'iterations': 500,  # Inner iteration (Default:5000). Reccomended: 1000?
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
policy = PolicyOptTf(policy_params, observation_dim, action_dim)
#policy = None
bigman_agent = GPSAgent(act_dim=action_dim, obs_dim=observation_dim, state_dim=state_dim, policy=policy)
# Load previous learned variables
#bigman_agent.load(file_save_restore)
print("Bigman Agent:%s OK\n" % type(bigman_agent))


# ################# #
# ################# #
# ##### COSTS ##### #
# ################# #
# ################# #

# Action Cost  #TODO: I think it doesn't have sense if the control is joint position
act_cost = {
    'type': CostAction,
    'wu': np.ones(action_dim) * 1e-4,
    #'l1': 1e-3,
    #'alpha': 1e-2,
    'target': None,   # Target action value
}

# State Cost
box_pose = [-0.7500,  # pos x
            0.0000,   # pos y
            0.0184,   # pos z
            0.0000,   # orient x
            0.0000,   # orient y
            0.0000,   # orient z
            1.0000]   # orient w

box_size = [0.4, 0.5, 0.3]

left_ee_pose = box_pose
left_ee_pose[0] += box_size[0]/2 - 0.05 #

target_state = left_ee_pose + box_pose
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

# Sum of costs
cost_sum = {
    'type': CostSum,
    'costs': [act_cost, state_cost],
    'weights': [0.1, 5.0],
}

# ################################ #
# ################################ #
# ## SAMPLE FROM DEMONSTRATIONS ## #
# ################################ #
# ################################ #
n_samples = 5
noisy = True
sampler_hyperparams = {
    'noisy': noisy,
    'noise_var_scale': 0.0001,  # It can be a np.array() with dim=dU
    'smooth_noise': False,
    'smooth_noise_var': 0.01,#01
    'smooth_noise_renormalize': False,
    'T': int(EndTime/Ts),  # Total points
    'dt': Ts
    }
traj_files = ['trajectories/traj1'+'_x'+str(box_position[0])+'_y'+str(box_position[1])+'_Y'+str(box_yaw)+'_m0_reach.npy',
              'trajectories/traj1'+'_x'+str(box_position[0])+'_y'+str(box_position[1])+'_Y'+str(box_yaw)+'_m1_lift.npy']
traj_rep_policy = TrajectoryReproducerPolicy(traj_files, act_idx=bigman_params['joint_ids']['UB'])
sampler = Sampler(traj_rep_policy, bigman_env, **sampler_hyperparams)
cond = bigman_env.add_q0(traj_rep_policy.traj_rep.get_data(0))
#raw_input("Press a key for sampling from Sampler")
sampler.take_samples(n_samples, cond=cond, noisy=noisy)


# ######################## #
# ######################## #
# ## LEARNING ALGORITHM ## #
# ######################## #
# ######################## #

# Learning params
total_episodes = 5
num_samples = 5  # Samples for exploration trajs
resume_training_itr = None  # Resume from previous training iteration
T = int(EndTime/Ts)  # Total points
conditions = 1  # Number of initial conditions
sample_on_policy = False
test_policy_after_iter = False
kl_step = 0.2
# init_traj_distr is a list of dict
init_traj_distr = {'type': init_lqr,
                    'init_var': 1.0,
                    'stiffness': 1.0,
                    'stiffness_vel': 0.5,
                    'final_weight': 1.0,
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
                 sample_on_policy=sample_on_policy,
                 test_after_iter=test_policy_after_iter,
                 init_traj_distr=init_traj_distr,
                 kl_step=kl_step,
                 gps_algo=gps_algo,
                 gps_algo_hyperparams=gps_algo_hyperparams
                 )
print("Learning algorithm: %s OK\n" % type(learn_algo))

# Learn using learning algorithm
print("Running Learning Algorithm!!!")
learn_algo.run(resume_training_itr)
print("Learning Algorithm has finished!")
sys.exit()

