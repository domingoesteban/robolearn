from __future__ import print_function

import numpy as np

from robolearn.envs import BigmanEnv
from robolearn.agents import LinearTFAgent
from robolearn.agents import MlpTFAgent
import rospy

import time

# Agent option
agent_behavior = 'learner' # 'learner'

# Learning params
total_episodes = 10
update_frequency = 5
EndTime = 10  # Using final time to define the horizon
Ts = 0.1


# Robot configuration
interface = 'ros'
joints_active = 'both_arms'
command_type = 'position'

# Create a ROS EnvInterface
bigman_env = BigmanEnv(interface=interface, mode='simulation', joints_active=joints_active, command_type=command_type)

action_dim = bigman_env.get_action_dim()
observation_dim = bigman_env.get_obs_dim()

print("\nBigman joints_active:%s (action_dim=%d). Command_type:%s" % (joints_active, action_dim, command_type))

# Create an Agent
#bigman_agent = LinearTFAgent(act_dim=action_dim, obs_dim=observation_dim)
bigman_agent = MlpTFAgent(act_dim=action_dim, obs_dim=observation_dim, hidden_units=[250, 250])
print("Bigman agent OK\n")

# Reset to initial position
bigman_env.reset()
time.sleep(5)  # TODO: This is temporal, because after reset the robot usually moves
#bigman_ros_interface.reset()
print("Bigman reset OK\n")

# ROS
ros_rate = rospy.Rate(1/Ts)  # hz

counter = 0
R = 0  # Finite horizon return
T = int(EndTime/Ts)
try:
    episode = 0
    history = [None] * total_episodes * T

    # Learn First
    while episode < total_episodes:
        print("Episode %d/%d" % (episode+1, total_episodes))
        i = 0

        # Collect history
        for i in xrange(T):
            obs = bigman_env.read_observation().reshape([1, -1])
            r = bigman_env.get_reward()
            action = bigman_agent.act(obs=obs).reshape([-1, 1])
            bigman_env.send_action(action)
            history[episode*T+i] = (obs, r, action)
            R = R + r

            ros_rate.sleep()

        print("The episode has finished!")
        print("Accumulated Reward is: %f\n" % R)

        print("Training the agent...")
        bigman_agent.train(history=history)
        print("Training ready!")

        print("Resetting environment!")
        bigman_env.reset()
        rospy.sleep(5)  # Because I need to find a good way to reset
        R = 0

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

