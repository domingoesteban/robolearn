from __future__ import print_function

import numpy as np

from robolearn.envs import AcrobotROSEnvInterface

import time

# Learning params
T = 10  # Using final time to define the horizon
Ts = 0.01

# Create a ROS EnvInterface
ros_interface = AcrobotROSEnvInterface('simulation')

# ros_interface.run()

# Reset to initial position
ros_interface.reset()
time.sleep(2)  # TODO: This is temporal, because after reset the robot usually moves
ros_interface.reset()


counter = 0
R = 0  # Finite horizon return
try:
    while True:
        if counter == 0:
            print("Starting new rollout...")

        obs = ros_interface.get_observation()
        R = R + ros_interface.get_reward()  # Accumulating reward
        action = np.random.normal(scale=2, size=1)  # It should be a function of state
        ros_interface.send_action(action)
        print("observation: %s , then action: %f" % (obs.transpose(), action))
        counter += 1
        time.sleep(Ts)

        if counter % (T/Ts) == 0:
            print("The rollout has finished!")
            print("Accumulated Reward is: %f\n" % R)
            ros_interface.reset()
            ros_interface.reset()
            time.sleep(3)  # Because I need to find a good way to reset
            R = 0
            counter = 0

except KeyboardInterrupt:
    print('Training interrupted!')

