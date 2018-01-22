import numpy as np
from robolearn.costs.cost_action import CostAction
from robolearn.costs.cost_fk import CostFK
from robolearn.costs.cost_state import CostState
from robolearn.costs.cost_sum import CostSum
from robolearn.costs.cost_utils import RAMP_FINAL_ONLY, RAMP_CONSTANT
from robolearn.costs.cost_utils import evall1l2term


# parameters
weight = 1e-4
target = None

# Get some variables from the environment
action_dim = env.get_action_dim()

# Action Cost
# Action Cost
act_cost = {
    'type': CostAction,
    'wu': np.ones(action_dim) * weight,
    'target': target,   # Target action value
}

# State Cost
target_distance_hand = np.zeros(6)
# target_distance_hand[-2] = -0.02  # Yoffset
# target_distance_hand[-1] = 0.1  # Zoffset

target_distance_object = np.zeros(6)
state_cost_distance = {
    'type': CostState,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'evalnorm': evall1l2term,  # TODO: ALWAYS USE evall1l2term
    'l1': 1.0,  # Weight for l1 norm
    'l2': 0.0,  # Weight for l2 norm
    'alpha': 1e-2,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
    'data_types': {
        'distance_object': {
            # 'wp': np.ones_like(target_state),  # State weights - must be set.
            'wp': np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]),  # State weights - must be set.
            'target_state': target_distance_object,  # Target state - must be set.
            'average': None,  # (12, 3),
            'data_idx': env.get_state_info(name='distance_object')['idx']
        },
    },
}
state_final_cost_distance = {
    'type': CostState,
    'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'evalnorm': evall1l2term,  # TODO: ALWAYS USE evall1l2term
    'l1': 1.0,  # Weight for l1 norm
    'l2': 0.0,  # Weight for l2 norm
    'alpha': 1e-2,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 10.0,  # Weight multiplier on final time step.
    'data_types': {
        'distance_object': {
            # 'wp': np.ones_like(target_state),  # State weights - must be set.
            'wp': np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]),  # State weights - must be set.
            'target_state': target_distance_object,  # Target state - must be set.
            'average': None,  # (12, 3),
            'data_idx': env.get_state_info(name='distance_object')['idx']
        },
    },
}

# FK cost

fk_cost = {
    'type': CostFK,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_hand,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': bigman_env.get_state_info(name='distance_hand')['idx'],
    'op_point_name': hand_name,
    'op_point_offset': hand_offset,
    'joints_idx': bigman_env.get_state_info(name='link_position')['idx'],
    'joint_ids': bigman_params['joint_ids'][body_part_active],
    'robot_model': robot_model,
    # 'wp': np.array([1.0, 1.0, 1.0, 0.7, 0.8, 0.6]),  # one dim less because 'quat' error | 1)orient 2)pos
    # 'wp': np.array([1.0, 1.0, 1.0, 6.0, 6.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'wp': np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'evalnorm': evall1l2term,
    #'evalnorm': evallogl2term,
    'l1': 1.0,  # 1.0,  # 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 1.0,  # 1.0,  #1.0e-3,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 1,  # 10
}

fk_l1_cost = {
    'type': CostFK,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_hand,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': env.get_state_info(name='distance_hand')['idx'],
    'op_point_name': hand_name,
    'op_point_offset': hand_offset,
    'joints_idx': env.get_state_info(name='link_position')['idx'],
    'joint_ids': bigman_params['joint_ids'][body_part_active],
    'robot_model': robot_model,
    # 'wp': np.array([1.0, 1.0, 1.0, 0.7, 0.8, 0.6]),  # one dim less because 'quat' error | 1)orient 2)pos
    #'wp': np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'wp': np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'evalnorm': evall1l2term,
    #'evalnorm': evallogl2term,
    'l1': 1.0,  # 1.0,  # 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 0.0,  # 1.0,  #1.0e-3,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 1,  # 10
}

fk_l2_cost = {
    'type': CostFK,
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_hand,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': env.get_state_info(name='distance_hand')['idx'],
    'op_point_name': hand_name,
    'op_point_offset': hand_offset,
    'joints_idx': env.get_state_info(name='link_position')['idx'],
    'joint_ids': bigman_params['joint_ids'][body_part_active],
    'robot_model': robot_model,
    # 'wp': np.array([1.0, 1.0, 1.0, 0.7, 0.8, 0.6]),  # one dim less because 'quat' error | 1)orient 2)pos
    #'wp': np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'wp': np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'evalnorm': evall1l2term,
    #'evalnorm': evallogl2term,
    'l1': 0.0,  # 1.0,  # 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 1.0,  # 1.0,  #1.0e-3,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 1,  # 10
}

# Final cost

fk_final_cost = {
    'type': CostFK,
    'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_hand,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': bigman_env.get_state_info(name='distance_hand')['idx'],
    'op_point_name': hand_name,
    'op_point_offset': hand_offset,
    'joints_idx': bigman_env.get_state_info(name='link_position')['idx'],
    'joint_ids': bigman_params['joint_ids'][body_part_active],
    'robot_model': robot_model,
    #'wp': np.array([1.0, 1.0, 1.0, 10.0, 10.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    #'wp': np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'wp': np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'evalnorm': evall1l2term,
    #'evalnorm': evallogl2term,
    'l1': 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 1.0,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 50,
}

fk_l1_final_cost = {
    'type': CostFK,
    'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_hand,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': bigman_env.get_state_info(name='distance_hand')['idx'],
    'op_point_name': hand_name,
    'op_point_offset': hand_offset,
    'joints_idx': bigman_env.get_state_info(name='link_position')['idx'],
    'joint_ids': bigman_params['joint_ids'][body_part_active],
    'robot_model': robot_model,
    #'wp': np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'wp': np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'evalnorm': evall1l2term,
    #'evalnorm': evallogl2term,
    'l1': 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 0.0,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 50,
}

fk_l2_final_cost = {
    'type': CostFK,
    'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
    'target_pose': target_distance_hand,
    'tgt_data_type': 'state',  # 'state' or 'observation'
    'tgt_idx': env.get_state_info(name='distance_hand')['idx'],
    'op_point_name': hand_name,
    'op_point_offset': hand_offset,
    'joints_idx': env.get_state_info(name='link_position')['idx'],
    'joint_ids': bigman_params['joint_ids'][body_part_active],
    'robot_model': robot_model,
    #'wp': np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'wp': np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
    'evalnorm': evall1l2term,
    #'evalnorm': evallogl2term,
    'l1': 0.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
    'l2': 1.0,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
    'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
    'wp_final_multiplier': 50,
}

# Sum costs
costs_and_weights = [(act_cost, 1.0e-1),
                     # (fk_cost, 1.0e-0),
                     (fk_l1_cost, 1.5e-1),
                     (fk_l2_cost, 1.0e-0),
                     # (fk_final_cost, 1.0e-0),
                     (fk_l1_final_cost, 1.5e-1),
                     (fk_l2_final_cost, 1.0e-0),
                     (state_cost_distance, 5.0e+0),
                     (state_final_cost_distance, 1.0e+1),
                     ]

cost_sum = {
    'type': CostSum,
    'costs': [cw[0] for cw in costs_and_weights],
    'weights': [cw[1] for cw in costs_and_weights],
}
