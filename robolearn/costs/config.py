""" Default configuration and hyperparameter values for costs. """
import numpy as np

from robolearn.costs.cost_utils import RAMP_CONSTANT, evallogl2term


# CostFK
COST_FK = {
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
    'wp': None,  # State weights - must be set.
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
    'env_target': True,  # TODO - This isn't used.
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-5,
    'target_end_effector': None,  # Target end-effector position.
    'evalnorm': evallogl2term,
}

# CostFKRelative
COST_FK_RELATIVE = {
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
    'wp': None,  # State weights - must be set.
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
    'env_target': True,  # TODO - This isn't used.
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-5,
    'target_rel_pose': None,
    'rel_data_type': None,  # 'state' or 'observation'
    'rel_data_name': None,  # Name of the state/observation
    'evalnorm': evallogl2term,
}


# CostState
COST_STATE = {
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-2,
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
    'data_types': {
        'link_position': {
            'average': None,  # From superball_gps
            'target_state': None,  # Target state - must be set.
            'wp': None,  # State weights - must be set.
            'data_idx': [],  # Indexes in the state vector
        },
    },
}

# CostBinaryRegion
COST_BINARY_REGION = {
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-2,
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
    'data_types': {
        'link_position': {
            'target_state': None,  # Target state - must be set.
            'wp': None,  # State weights - must be set.
        },
    },
    'max_distance': 0.1,
    'outside_cost': 1.0,
    'inside_cost': 0.0,
}

# CostBinaryRegion
COST_SAFE_DISTANCE = {
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
    'data_types': {
        'link_position': {
            'wp': None,  # State weights - must be set.
            'safe_distance': None,  # Target state - must be set.
            'outside_cost': 1.0,
            'inside_cost': 1.0,
            'data_idx': [],
        },
    },
}

# CostSum
COST_SUM = {
    'costs': [],  # A list of hyperparam dictionaries for each cost.
    'weights': [],  # Weight multipliers for each cost.
}


# CostAction
COST_ACTION = {
    'wu': np.array([]),  # Torque penalties, must be 1 x dU numpy array.
    'target': None,   # Target action value  # DOMINGO: From superball_gps
}


# CostLinWP
COST_LIN_WP = {
    'waypoint_time': np.array([1.0]),
    'ramp_option': RAMP_CONSTANT,
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-5,
    'logalpha': 1e-5,
    'log': 0.0,
}
