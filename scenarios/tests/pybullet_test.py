"""
import pybullet as p
import pybullet_data
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
#physicsClient = p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
#physicsClient = p.connect(p.UDP, 'localhost', 1234)  # or p.DIRECT for non-graphical version
#physicsClient = p.connect(p.TCP, 'localhost', 6667)  # or p.DIRECT for non-graphical version
#physicsClient = p.connect(p.SHARED_MEMORY, 1234)  # or p.DIRECT for non-graphical version

p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [1, 1, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)
boxId = p.load("r2d2.urdf", cubeStartPos, cubeStartOrientation)
manipulatorStartPos = [0, 0, 0.5]
manipulatorStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
manipulatorId = p.loadURDF("./manipulator2d.urdf", manipulatorStartPos, manipulatorStartOrientation)

p.stepSimulation()
#p.setRealTimeSimulation(1)

cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos, cubeOrn)
input('key to finish')
p.disconnect()
"""


import math
import numpy as np
from pybullet_envs.bullet import minitaur_gym_env

n_envs = 2

envs = [None for _ in range(n_envs)]

for ii in range(n_envs):
    envs[ii] = minitaur_gym_env.MinitaurBulletEnv(
        render=True,
        leg_model_enabled=False,
        motor_velocity_limit=np.inf,
        motor_overheat_protection=True,
        accurate_motor_model_enabled=True,
        motor_kp=1.20,
        motor_kd=0.02,
        on_rack=False)
    input('press key')


steps = 1000
amplitude = 0.5
speed = 3

actions_and_observations = []

for ii in range(n_envs):
    for step_counter in range(steps):
        # Matches the internal timestep.
        time_step = 0.01
        t = step_counter * time_step
        current_row = [t]

        action = [math.sin(speed * t) * amplitude + math.pi / 2] * 8
        current_row.extend(action)

        observation, _, _, _ = envs[ii].step(action)
        current_row.extend(observation.tolist())
        actions_and_observations.append(current_row)

    envs[ii].reset()

input('press to finish')
