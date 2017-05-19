from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from robolearn.utils.iit_robots_params import *
from robolearn.envs import BigmanEnv
from robolearn.agents import GPSAgent
from robolearn.utils.sample import Sample
from robolearn.utils.sample_list import SampleList
from robolearn.algos.gps import GPS

import rospy

import time

# Agent option
agent_behavior = 'learner'  # 'learner', 'actor'
update_frequency = 5
Ts = 0.05
EndTime = 10  # Using final time to define the horizon



# Robot configuration
interface = 'ros'
body_part_active = 'LA'
command_type = 'position'
file_save_restore = "models/bigman_agent_vars.ckpt"


observation_active = [{'name': 'joint_state',
                       'type': 'joint_state',
                       'ros_topic': '/xbotcore/bigman/joint_states',
                       'fields': ['link_position', 'link_velocity', 'effort'],
                       'joints': range(15, 22) + range(24, 31)},  # Value that can be gotten from robot_params['joints_names']['UB']

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
                 'joints': range(15, 22)}]  # Value that can be gotten from robot_params['joints_names']['LA']


# Create a Bigman robot ROS EnvInterface
bigman_env = BigmanEnv(interface=interface, mode='simulation',
                       body_part_active=body_part_active, command_type=command_type,
                       observation_active=observation_active,
                       state_active=state_active)

action_dim = bigman_env.get_action_dim()
state_dim = bigman_env.get_state_dim()
observation_dim = bigman_env.get_obs_dim()

print("\nBigman body_part_active:%s (action_dim=%d). Command_type:%s" % (body_part_active, action_dim, command_type))

# Create an Agent
bigman_agent = GPSAgent(act_dim=action_dim, obs_dim=observation_dim, state_dim=state_dim)


# Load previous learned variables
#bigman_agent.load(file_save_restore)
print("Bigman agent OK\n")

## Reset to initial position
#print("Resetting Bigman...")
#bigman_env.reset()
#time.sleep(5)  # TODO: This is temporal, because after reset the robot usually moves
##bigman_ros_interface.reset()
#print("Bigman reset OK\n")


# Learning params
total_episodes = 10
num_samples = 5  # Samples for exploration trajs
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
R = 0  # Finite horizon return
try:
    episode = 0
    sample_list = SampleList()

    print("Starting Training...")
    # Learn First
    while episode < total_episodes:
        print("Episode %d/%d" % (episode+1, total_episodes))
        i = 0

        # Create a sample class
        sample = Sample(bigman_env, T)
        history = [None] * T
        state_hist = [None] * T

        # Collect history
        for i in range(T):
            #action = bigman_agent.act(obs=obs).reshape([-1, 1])
            action = np.zeros(bigman_agent.act_dim)#.reshape([-1, 1])
            action[0] = 0.5 * np.sin(rospy.Time.now().to_sec())
            bigman_env.send_action(action)
            print("i=%d/%d" % (i, T))
            obs = bigman_env.get_observation()
            state = bigman_env.get_state()
            history[i] = (obs, action)
            state_hist[i] = state
            #print(obs)
            #print("..")
            #print(state)
            print("--")
            print("obs_shape:(%s)" % obs.shape)
            print("state_shape:(%s)" % state.shape)
            print("obs active names: %s" % bigman_env.get_obs_info()['names'])
            print("obs active dims: %s" % bigman_env.get_obs_info()['dimensions'])
            print("state active names: %s" % bigman_env.get_state_info()['names'])
            print("state active dims: %s" % bigman_env.get_state_info()['dimensions'])
            print("")
            #sample.set_acts(action, t=i)  # Set action One by one
            #sample.set_obs(obs[:42], obs_name='joint_state', t=i)  # Set action One by one
            #sample.set_states(state[:7], state_name='link_position', t=i)  # Set action One by one

            ros_rate.sleep()

        print("The episode has finished!")
        print("Accumulated Reward is: %f\n" % R)
        all_actions = np.array([hist[1] for hist in history])
        all_obs = np.array([hist[0] for hist in history])
        all_states = np.array([hist for hist in state_hist])
        sample.set_acts(all_actions)  # Set all actions at the same time
        sample.set_obs(all_obs)  # Set all obs at the same time
        sample.set_states(all_states)  # Set all states at the same time

        #plt.plot(sample.get_acts()[:, 0], 'k')
        #plt.plot(sample.get_obs('joint_state')[:, 0], 'b')
        #plt.plot(sample.get_states('link_position')[:, 0], 'r')
        #plt.show()

        print("Training the agent...")
        #bigman_agent.train(history=history)
        #bigman_agent.save(file_save_restore)
        print("Training ready!")

        print("Resetting environment!")
        #bigman_env.reset()
        #rospy.sleep(5)  # Because I need to find a good way to reset

        episode += 1

    print("Training finished!")

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

