from __future__ import print_function

import numpy as np

from robolearn.envs import CentauroEnv
from robolearn.agents import GPSAgent
from robolearn.algos.gps import GPS

import rospy

import time

# Agent option
agent_behavior = 'learner'  # 'learner', 'actor'
update_frequency = 5
Ts = 0.1
EndTime = 10  # Using final time to define the horizon



# Robot configuration
interface = 'ros'
joints_active = 'left_arm'
command_type = 'position'
file_save_restore = "models/centauro_agent_vars.ckpt"

# Create a ROS EnvInterface
centauro_env = CentauroEnv(interface=interface, mode='simulation', joints_active=joints_active, command_type=command_type)

action_dim = centauro_env.get_action_dim()
state_dim = centauro_env.get_state_dim()
observation_dim = centauro_env.get_obs_dim()

print("\nCentauro joints_active:%s (action_dim=%d). Command_type:%s" % (joints_active, action_dim, command_type))

# Create an Agent
centauro_agent = GPSAgent(act_dim=action_dim, obs_dim=observation_dim, state_dim=state_dim)


# Load previous learned variables
#centauro_agent.load(file_save_restore)
print("Centauro agent OK\n")

## Reset to initial position
#print("Resetting Centauro...")
#centauro_env.reset()
#time.sleep(5)  # TODO: This is temporal, because after reset the robot usually moves
##centauro_ros_interface.reset()
#print("Centauro reset OK\n")

# Learning params
total_episodes = 10
num_samples = 5  # Samples for exploration trajs
resume_training_itr = None  # Resume from previous training iteration
conditions = 1  # Number of initial conditions
T = EndTime/Ts  # Total points
learn_algo = GPS(agent=centauro_agent, env=centauro_env, iterations=total_episodes, num_samples=num_samples,
                 T=T,
                 conditions=conditions)

# Learn using learning algorithm
learn_algo.run(resume_training_itr)

## ROS
#ros_rate = rospy.Rate(1/Ts)  # hz
#
#counter = 0
#R = 0  # Finite horizon return
#T = int(EndTime/Ts)  # Horizon
#try:
#    episode = 0
#    history = [None] * total_episodes * T
#
#    # Learn First
#    while episode < total_episodes:
#        print("Episode %d/%d" % (episode+1, total_episodes))
#        i = 0
#
#        # Collect history
#        for i in xrange(T):
#            obs = centauro_env.read_observation().reshape([1, -1])
#            r = centauro_env.get_reward()
#            action = centauro_agent.act(obs=obs).reshape([-1, 1])
#            centauro_env.send_action(action)
#            history[episode*T+i] = (obs, r, action)
#            R = R + r
#
#            ros_rate.sleep()
#
#        print("The episode has finished!")
#        print("Accumulated Reward is: %f\n" % R)
#
#        print("Training the agent...")
#        centauro_agent.train(history=history)
#        centauro_agent.save(file_save_restore)
#        print("Training ready!")
#
#        print("Resetting environment!")
#        centauro_env.reset()
#        rospy.sleep(5)  # Because I need to find a good way to reset
#        R = 0
#
#        episode += 1
#
#    print("Training finished!")
#
#    while True:
#        if agent_behavior == 'actor':
#            obs = centauro_env.read_observation()
#            obs = np.reshape(obs, [1, -1])
#            action = centauro_agent.act(obs=obs)
#            action = np.reshape(action, [-1, 1])
#            centauro_env.send_action(action)
#            ros_rate.sleep()
#
#        else:
#            # Old training example
#            if counter == 0:
#                print("Starting new rollout...")
#
#            obs = centauro_env.read_observation()
#            #R = R + centauro_ros_interface.get_reward()  # Accumulating reward
#            if command_type == 'torque':
#                action = np.random.normal(scale=80, size=action_dim)  # It should be a function of state
#            else:
#                action = np.random.normal(scale=0.4, size=action_dim)  # It should be a function of state
#            centauro_env.send_action(action)
#            #print("observation: %s , then action: %f" % (obs.transpose(), action))
#            counter += 1
#            #time.sleep(Ts)
#            ros_rate.sleep()
#
#            if counter % (T/Ts) == 0:
#                print("The rollout has finished!")
#                print("Accumulated Reward is: %f\n" % R)
#                print("Training the agent...")
#                centauro_agent.train(history=history)
#                print("Training ready!")
#
#                print("Resetting environment!")
#                centauro_env.reset()
#                #time.sleep(5)  # Because I need to find a good way to reset
#                rospy.sleep(5)  # Because I need to find a good way to reset
#                R = 0
#                counter = 0
#
#except KeyboardInterrupt:
#    print('Training interrupted!')

