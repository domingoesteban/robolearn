from __future__ import print_function

import sys
import numpy as np
import matplotlib.pyplot as plt

from robolearn.utils.iit_robots_params import *
from robolearn.envs import BigmanEnv
from robolearn.agents import GPSAgent

from robolearn.utils.sample import Sample
from robolearn.utils.sample_list import SampleList

from robolearn.costs.cost_action import CostAction
from robolearn.costs.cost_state import CostState
from robolearn.costs.cost_sum import CostSum
from robolearn.costs.cost_utils import RAMP_QUADRATIC

from robolearn.utils.algos_utils import IterationData
from robolearn.utils.algos_utils import TrajectoryInfo
from robolearn.algos.gps import GPS

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
EndTime = 2  # Using final time to define the horizon


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
agent_behavior = 'learner'  # 'learner', 'actor'
bigman_agent = GPSAgent(act_dim=action_dim, obs_dim=observation_dim, state_dim=state_dim)
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
target_pos[0] = 0.7854
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

cost_sum = {
    'type': CostSum,
    'costs': [act_cost, state_cost],
    'weights': [0.1, 5.0],
}




# Reset to initial position
print("Resetting Bigman...")
bigman_env.reset(time=1)
#time.sleep(5)  # TODO: This is temporal, because after reset the robot usually moves
##bigman_ros_interface.reset()
print("Bigman reset OK\n")


# Learning params
total_episodes = 1
num_samples = 2  # Samples for exploration trajs
resume_training_itr = None  # Resume from previous training iteration
conditions = 1  # Number of initial conditions
T = int(EndTime/Ts)  # Total points
#learn_algo = GPS(agent=bigman_agent, env=bigman_env, iterations=total_episodes, num_samples=num_samples,
#                 T=T,
#                 conditions=conditions)
#
## Learn using learning algorithm
#learn_algo.run(resume_training_itr)



# ROS
ros_rate = rospy.Rate(int(1/Ts))  # hz

counter = 0

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

    while True:
        if agent_behavior == 'actor':
            obs = bigman_env.read_observation()
            obs = np.reshape(obs, [1, -1])
            action = bigman_agent.act(obs=obs)
            action = np.reshape(action, [-1, 1])
            bigman_env.send_action(action)
            ros_rate.sleep()

        else:
            # Old training example
            if counter == 0:
                print("Starting new rollout...")

            obs = bigman_env.read_observation()
            #R = R + bigman_ros_interface.get_reward()  # Accumulating reward
            if command_type == 'torque':
                action = np.random.normal(scale=80, size=action_dim)  # It should be a function of state
            else:
                action = np.random.normal(scale=0.4, size=action_dim)  # It should be a function of state
            bigman_env.send_action(action)
            #print("observation: %s , then action: %f" % (obs.transpose(), action))
            counter += 1
            #time.sleep(Ts)
            ros_rate.sleep()

            if counter % (T/Ts) == 0:
                print("The rollout has finished!")
                print("Accumulated Reward is: %f\n" % R)
                print("Training the agent...")
                bigman_agent.train(history=history)
                print("Training ready!")

                print("Resetting environment!")
                bigman_env.reset()
                #time.sleep(5)  # Because I need to find a good way to reset
                rospy.sleep(5)  # Because I need to find a good way to reset
                R = 0
                counter = 0

except KeyboardInterrupt:
    print('Training interrupted!')

