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

import rospy

import time


# ################## #
# ################## #
# ### PARAMETERS ### #
# ################## #
# ################## #
# Task parameters
#update_frequency = 5
Ts = 0.05
EndTime = 5  # Using final time to define the horizon


# ################### #
# ################### #
# ### ENVIRONMENT ### #
# ################### #
# ################### #

print("\nCreating Bigman environment...")

# Robot configuration
interface = 'ros'
body_part_active = 'LA'
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
                       'fields': ['orientation', 'angular_velocity', 'linear_acceleration']}]

#observation_active = [{'name': 'imu1',
#                       'type': 'imu',
#                       'ros_topic': '/xbotcore/bigman/imu/imu_link',
#                       'fields': ['orientation', 'angular_velocity', 'linear_acceleration']}]

state_active = [{'name': 'joint_state',
                 'type': 'joint_state',
                 'fields': ['link_position', 'link_velocity'],
                 'joints': bigman_params['joint_ids']['LA']}]  # Value that can be gotten from robot_params['joints_ids']['LA']




# Create a Bigman robot ROS EnvInterface
bigman_env = BigmanEnv(interface=interface, mode='simulation',
                       body_part_active=body_part_active, command_type=command_type,
                       observation_active=observation_active,
                       state_active=state_active)

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
    'l1': 1e-3,
    'alpha': 1e-2,
    'target': None,   # Target action value
}

# State Cost
target_pos = np.zeros(len(state_active[0]['joints']))
target_vel = np.zeros(len(state_active[0]['joints']))
# 'B' pose
target_pos[:] = [np.deg2rad(0), np.deg2rad(90), np.deg2rad(0), np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]
state_cost = {
    'type': CostState,
    'ramp_option': RAMP_QUADRATIC,  # How target cost ramps over time. RAMP_* :CONSTANT,LINEAR, QUADRATIC, FINAL_ONLY
    'l1': 0.0,
    'l2': 1.0,
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
    'data_types': {
        'link_position': {
            'wp': np.ones_like(target_pos),  # State weights - must be set.
            'target_state': target_pos,  # Target state - must be set.
            'average': None,  #(12, 3),
            'data_idx': bigman_env.get_state_info(name='link_position')['idx']
        },
        'link_velocity': {
            'wp': np.ones_like(target_vel),  # State weights - must be set.
            'target_state': target_vel,  # Target state - must be set.
            'average': None,  #(12, 3),
            'data_idx': bigman_env.get_state_info(name='link_velocity')['idx']
        },
    },
}

# Sum of costs
cost_sum = {
    'type': CostSum,
    'costs': [act_cost, state_cost],
    'weights': [0.1, 5.0],
}


# ######################## #
# ######################## #
# ## LEARNING ALGORITHM ## #
# ######################## #
# ######################## #

# Learning params
#gps_algo = 'pigps'
gps_algo = 'mdgps'
total_episodes = 5
num_samples = 5  # Samples for exploration trajs
resume_training_itr = None  # Resume from previous training iteration
conditions = 1  # Number of initial conditions
T = int(EndTime/Ts)  # Total points
sample_on_policy = False
#init_traj_distr = {'type': init_lqr,  # init_lqr, init_pd
#                   'init_var': 1.0,
#                   'stiffness': 1.0,
#                   'stiffness_vel': 0.5,
#                   'final_weight': 1.0,
#                   # Parameters for guessing dynamics
#                   'init_acc': [],  # dU vector of accelerations, default zeros.
#                   'init_gains': [],  # dU vector of gains, default ones.
#                   }
init_traj_distr = {'type': init_pd,
                   'init_var': 0.00001,  # initial variance (Default:10)
                   'pos_gains': 0.001,  # position gains (Default:10)
                   'vel_gains_mult': 0.01,  # velocity gains multiplier on pos_gains
                   'init_action_offset': None,
                   }
test_policy_after_iter = False
learn_algo = GPS(agent=bigman_agent, env=bigman_env,
                 iterations=total_episodes, num_samples=num_samples,
                 T=T, dt=Ts,
                 cost=cost_sum,
                 conditions=conditions,
                 gps_algo=gps_algo,
                 sample_on_policy=sample_on_policy,
                 test_after_iter=test_policy_after_iter,
                 init_traj_distr=init_traj_distr
                 )
print("Learning algorithm: %s OK\n" % type(learn_algo))

# Learn using learning algorithm
print("Running Learning Algorithm!!!")
learn_algo.run(resume_training_itr)
print("Learning Algorithm has finished!")
sys.exit()

# ######################### #
# EXAMPLE OF AN EXPLORATION #
# ######################### #
ros_rate = rospy.Rate(int(1/Ts))  # hz
try:
    episode = 0
    sample_list = SampleList()

    print("Starting Training...")
    # Learn First
    for episode in range(total_episodes):
        print("")
        print("#"*15)
        print("Episode %d/%d" % (episode+1, total_episodes))
        print("#"*15)

        for n_sample in range(num_samples):
            print("")
            print("New Sample: Sample %d/%d" % (n_sample+1, num_samples))
            i = 0

            # Create a sample class
            sample = Sample(bigman_env, T)
            history = [None] * T
            obs_hist = [None] * T

            # Collect history
            for i in range(T):
                obs = bigman_env.get_observation()
                state = bigman_env.get_state()
                action = bigman_agent.act(obs=obs)
                bigman_env.send_action(action)
                print("Episode %d/%d | Sample:%d/%d | t=%d/%d" % (episode+1, total_episodes,
                                                                  n_sample+1, num_samples,
                                                                  i+1, T))
                obs_hist[i] = (obs, action)
                history[i] = (state, action)
                #print(obs)
                #print("..")
                #print(state)
                #print("--")
                #print("obs_shape:(%s)" % obs.shape)
                #print("state_shape:(%s)" % state.shape)
                #print("obs active names: %s" % bigman_env.get_obs_info()['names'])
                #print("obs active dims: %s" % bigman_env.get_obs_info()['dimensions'])
                #print("state active names: %s" % bigman_env.get_state_info()['names'])
                #print("state active dims: %s" % bigman_env.get_state_info()['dimensions'])
                #print("")

                #sample.set_acts(action, t=i)  # Set action One by one
                #sample.set_obs(obs[:42], obs_name='joint_state', t=i)  # Set action One by one
                #sample.set_states(state[:7], state_name='link_position', t=i)  # Set action One by one

                ros_rate.sleep()

            all_actions = np.array([hist[1] for hist in history])
            all_states = np.array([hist[0] for hist in history])
            all_obs = np.array([hist[0] for hist in obs_hist])
            sample.set_acts(all_actions)  # Set all actions at the same time
            sample.set_obs(all_obs)  # Set all obs at the same time
            sample.set_states(all_states)  # Set all states at the same time

            # Add sample to sample list
            print("Sample added to sample_list!")
            sample_list.add_sample(sample)

            print("Resetting environment!")
            bigman_env.reset(time=1)
            #rospy.sleep(5)  # Because I need to find a good way to reset

        print("")
        print("Exploration finished. %d samples were generated" % sample_list.num_samples())

        print("")
        print("Evaluating samples' costs...")
        #Evaluate costs for all samples for a condition.
        # Important classes
        #cost = act_cost['type'](act_cost)
        #cost = state_cost['type'](state_cost)
        cost = cost_sum['type'](cost_sum)
        iteration_data = IterationData()
        iteration_data.traj_info = TrajectoryInfo()  # Cast it directly in gps algo, with M variable
        # Constants.
        N_samples = len(sample_list)

        # Compute cost.
        cs = np.zeros((N_samples, T))  # Sample costs of the current iteration.
        cc = np.zeros((N_samples, T))  # Cost estimate constant term.
        cv = np.zeros((N_samples, T, state_dim+action_dim))  # Cost estimate vector term.
        Cm = np.zeros((N_samples, T, state_dim+action_dim, state_dim+action_dim))  # Cost estimate matrix term.

        for n in range(N_samples):
            sample = sample_list[n]
            # Get costs.
            l, lx, lu, lxx, luu, lux = cost.eval(sample)
            cc[n, :] = l
            cs[n, :] = l

            # Assemble matrix and vector.
            cv[n, :, :] = np.c_[lx, lu]
            Cm[n, :, :, :] = np.concatenate(
                (np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]),
                axis=1
            )

            #TODO: Check this part better, and understand it
            # Adjust for expanding cost around a sample.
            X = sample.get_states()
            U = sample.get_acts()
            yhat = np.c_[X, U]
            rdiff = -yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] += cv_update

        # Fill in cost estimate.
        iteration_data.traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        iteration_data.traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        iteration_data.traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        iteration_data.cs = cs  # True value of cost.
        print("Mean cost for iteration %d: %f" % (episode+1, np.sum(np.mean(cs, 0))))

        print("The episode has finished!")


        #print("Training the agent...")
        #bigman_agent.train(history=history)
        #bigman_agent.save(file_save_restore)
        #print("Training ready!")



    #all_samples_obs = sample_list.get_obs(idx=range(2, 4), obs_name='joint_state')
    #print(all_samples_obs.shape)

    #for samp in all_samples_obs:
    #    plt.plot(samp[:, 0])
    #plt.show()
    #plt.plot(sample.get_acts()[:, 0], 'k')
    #plt.plot(sample.get_obs('joint_state')[:, 0], 'b')
    #plt.plot(sample.get_states('link_position')[:, 0], 'r')
    #plt.show()

    print("Training finished!")
    sys.exit()

except KeyboardInterrupt:
    print('Training interrupted!')

