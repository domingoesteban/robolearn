import os
import numpy as np
import matplotlib.pyplot as plt

from robolearn.costs.cost_action import CostAction
from robolearn.costs.cost_state import CostState
from robolearn.costs.cost_fk_relative import CostFKRelative
from robolearn.costs.cost_sum import CostSum
from robolearn.costs.cost_utils import RAMP_QUADRATIC, RAMP_CONSTANT, RAMP_LINEAR, RAMP_FINAL_ONLY

from robolearn.utils.robot_model import RobotModel
from robolearn.utils.sample import Sample
from robolearn.utils.iit.iit_robots_params import bigman_params
from robolearn.utils.lift_box_utils import create_box_relative_pose
from robolearn.utils.lift_box_utils import create_hand_relative_pose


np.set_printoptions(precision=8, suppress=True, linewidth=1000)

# ########## #
# Parameters #
# ########## #

T = 100
dX = 35
dU = 14
dO = 35
# BOX
box_x = 0.75-0.05
box_y = 0.00
box_z = 0.0184
box_yaw = 0  # Degrees
box_size = [0.4, 0.5, 0.3]
box_relative_pose = create_box_relative_pose(box_x=box_x, box_y=box_y, box_z=box_z, box_yaw=box_yaw)


# ########### #
# Environment #
# ########### #
class FakeEnv(object):
    def __init__(self, dim_o, dim_x, dim_u):
        self.dO = dim_o
        self.dX = dim_x
        self.dU = dim_u

    def get_state_dim(self):
        return self.dX

    def get_action_dim(self):
        return self.dU

    def get_obs_dim(self):
        return self.dO

    @staticmethod
    def get_env_info():
        return ''

bigman_env = FakeEnv(dO, dX, dU)


# ########### #
# Robot Model #
# ########### #

# Robot Model (It is used to calculate the IK cost)
robot_urdf_file = os.environ["ROBOTOLOGY_ROOT"]+'/configs/ADVR_shared/bigman/urdf/bigman.urdf'
# robot_urdf_file = os.environ["ROBOTOLOGY_ROOT"]+'/robots/iit-bigman-ros-pkg/bigman_urdf/urdf/bigman.urdf'
robot_model = RobotModel(robot_urdf_file)
LH_name = 'LWrMot3'
RH_name = 'RWrMot3'
l_soft_hand_offset = np.array([0.000, -0.030, -0.210])
r_soft_hand_offset = np.array([0.000, 0.030, -0.210])


# ###### #
# Sample #
# ###### #
sample = Sample(bigman_env, T)

# Robot Configuration
# qLA = np.array([0.0568,  0.2386, -0.2337, -1.6803,  0.2226,  0.0107,  0.5633])
qLA = np.deg2rad([0, 50, 0, -75, 0, 0, 0])
# qRA = np.array([0.0568,  -0.2386, 0.2337, -1.6803,  -0.2226,  0.0107,  -0.5633])
qRA = np.deg2rad([0, -50, 0, -75, 0, 0, 0])

all_actions = np.ones((T, dU))*1
all_obs = np.zeros((T, dO))
all_states = np.zeros((T, dX))
all_states[:, :7] = np.tile(qLA, (T, 1))
all_states[:, 7:14] = np.tile(qRA, (T, 1))
all_states[:, -7:] = np.tile(box_relative_pose[[3, 4, 5, 6, 0, 1, 2]], (T, 1))

sample.set_acts(all_actions)  # Set all actions at the same time
sample.set_obs(all_obs)  # Set all obs at the same time
sample.set_states(all_states)  # Set all states at the same time

# #### #
# Cost #
# #### #
act_cost = {
    'type': CostAction,
    'wu': np.ones(dU) * 1e-4,
    'target': None,   # Target action value
}

# State Cost
target_state = box_relative_pose[[3, 4, 5, 6, 0, 1, 2]]
state_cost = {
    'type': CostState,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'l1': 0.0,  # Weight for l1 norm
    'l2': 1.0,  # Weight for l2 norm
    'alpha': 1e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 5.0,  # Weight multiplier on final time step.
    'data_types': {
        'optitrack': {
            'wp': np.ones_like(target_state),  # State weights - must be set.
            'target_state': target_state,  # Target state - must be set.
            'average': None,  # (12, 3),
            'data_idx': range(28, 35)
        },
    },
}

# FK Cost
left_hand_rel_pose = create_hand_relative_pose([0, 0, 0, 0, 0, 0, 1],
                                               hand_x=0.0, hand_y=box_size[1]/2-0.02, hand_z=0.0, hand_yaw=0)
left_hand_rel_pose[:] = left_hand_rel_pose[[3, 4, 5, 6, 0, 1, 2]]  # Changing from 'pos+orient' to 'orient+pos'
LAfk_cost = {
    'type': CostFKRelative,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_rel_pose': left_hand_rel_pose,
    'rel_data_type': 'state',  # 'state' or 'observation'
    #'rel_data_name': 'optitrack',  # Name of the state/observation
    'rel_idx': range(28, 35),
    'data_idx': range(0, 14),
    'end_effector_name': LH_name,
    'end_effector_offset': l_soft_hand_offset,
    'joint_ids': bigman_params['joint_ids']['BA'],
    'robot_model': robot_model,
    #'wp': np.array([1.2, 0, 0.8, 1, 1.2, 0.8]),  # one dim less because 'quat' error | 1)orient 2)pos
    #'wp': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'wp': np.array([1.0, 1.0, 1.0, 0.7, 0.8, 0.6]),  # one dim less because 'quat' error | 1)orient 2)pos
    'l1': 0.0,  # Weight for l1 norm
    'l2': 1.0,  # Weight for l2 norm
    'alpha': 1e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 5,
}


right_hand_rel_pose = create_hand_relative_pose([0, 0, 0, 0, 0, 0, 1],
                                                hand_x=0.0, hand_y=-box_size[1]/2+0.02, hand_z=0.0, hand_yaw=0)
right_hand_rel_pose[:] = right_hand_rel_pose[[3, 4, 5, 6, 0, 1, 2]]  # Changing from 'pos+orient' to 'orient+pos'
RAfk_cost = {
    'type': CostFKRelative,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT,LINEAR, QUADRATIC, FINAL_ONLY
    'target_rel_pose': right_hand_rel_pose,
    'rel_data_type': 'state',  # 'state' or 'observation'
    #'rel_data_name': 'optitrack',  # Name of the state/observation
    'rel_idx': range(28, 35),
    'data_idx': range(0, 14),
    'end_effector_name': RH_name,
    'end_effector_offset': r_soft_hand_offset,
    'joint_ids': bigman_params['joint_ids']['BA'],
    'robot_model': robot_model,
    #'wp': np.array([1.2, 0, 0.8, 1, 1.2, 0.8]),  # one dim less because 'quat' error | 1)orient 2)pos
    #'wp': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'wp': np.array([1.0, 1.0, 1.0, 0.7, 0.8, 0.6]),  # one dim less because 'quat' error | 1)orient 2)pos
    'l1': 0.0,  # Weight for l1 norm
    'l2': 1.0,  # Weight for l2 norm
    'alpha': 1e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 5,
}

cost_sum = {
    'type': CostSum,
    'costs': [act_cost, state_cost, LAfk_cost, RAfk_cost],
    'weights': [0.1, 5.0, 8.0, 8.0],
    # 'costs': [act_cost, state_cost],#, LAfk_cost, RAfk_cost],
    # 'weights': [0.1, 5.0],
}

cost1 = LAfk_cost['type'](LAfk_cost)
cost2 = RAfk_cost['type'](RAfk_cost)
cost3 = act_cost['type'](act_cost)
cost4 = state_cost['type'](state_cost)
#cost = cost_sum['type'](cost_sum)


print("Evaluating sample's cost...")
l, lx, lu, lxx, luu, lux = cost1.eval(sample)
print('----')
print(l[1])
print(l[-1])
# print(lx[1, :])
# print(lx[-1, :])
# print(lu[1, :])
# print(lu[-1, :])

# print("l %s" % str(l.shape))
# print("lx %s" % str(lx.shape))
# print("lu %s" % str(lu.shape))
# print("lxx %s" % str(lxx.shape))
# print("luu %s" % str(luu.shape))
# print("lux %s" % str(lux.shape))


l, lx, lu, lxx, luu, lux = cost2.eval(sample)
print('%%%%')
print(l[1])
print(l[-1])
# print(lx[1, :])
# print(lx[-1, :])
# print(lu[1, :])
# print(lu[-1, :])

# l, lx, lu, lxx, luu, lux = cost3.eval(sample)
# print('%%%%')
# print(l[1])
# print(l[-1])
# print(lx[1, :])
# print(lx[-1, :])
# print(lu[1, :])
# print(lu[-1, :])

plt.plot(l)
#plt.plot(lx)
#plt.plot(lu)
#plt.plot(lxx)
#plt.plot(lux)
plt.show()

