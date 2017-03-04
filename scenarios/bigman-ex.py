from __future__ import print_function

import numpy as np

from robolearn.envs import BigmanEnv
from robolearn.agents import LinearTFAgent
import rospy

import time
np.zeros_like()
# Agent option
agent_behavior = 'actor' # 'learner'

# Learning params
T = 10  # Using final time to define the horizon
Ts = 0.1

# Robot configuration
interface = 'ros'
joints_active = 'both_arms'
command_type = 'torque'

# Create a ROS EnvInterface
bigman_env = BigmanEnv(interface=interface, mode='simulation', joints_active=joints_active, command_type=command_type)

action_dim = bigman_env.get_action_dim()
observation_dim = bigman_env.get_obs_dim()

print("\nBigman joints_active:%s (action_dim=%d). Command_type:%s" % (joints_active, action_dim, command_type))

# Create a Linear Agent
bigman_agent = LinearTFAgent(act_dim=action_dim, obs_dim=observation_dim)
print("Bigman agent OK\n")

# Reset to initial position
bigman_env.reset()
time.sleep(5)  # TODO: This is temporal, because after reset the robot usually moves
#bigman_ros_interface.reset()
print("Bigman reset OK\n")



counter = 0
R = 0  # Finite horizon return
try:
    while True:
        if agent_behavior == 'actor':
            obs = bigman_env.read_observation()
            obs = np.reshape(obs, [1, -1])
            action = 10*bigman_agent.act(obs=obs)
            action = np.reshape(action, [-1, 1])
            bigman_env.send_action(action)
            rospy.sleep(Ts)  # Because I need to find a good way to reset
        elif agent_behavior == 'learner':
            pass
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
            rospy.sleep(Ts)

            if counter % (T/Ts) == 0:
                print("The rollout has finished!")
                print("Accumulated Reward is: %f\n" % R)
                bigman_env.reset()
                #time.sleep(5)  # Because I need to find a good way to reset
                rospy.sleep(5)  # Because I need to find a good way to reset
                R = 0
                counter = 0

except KeyboardInterrupt:
    print('Training interrupted!')

