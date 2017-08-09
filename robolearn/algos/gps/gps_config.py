
import numpy as np

from robolearn.utils.dynamics.dynamics_lr_prior import DynamicsLRPrior
from robolearn.utils.dynamics.dynamics_prior_gmm import DynamicsPriorGMM

from robolearn.utils.traj_opt.traj_opt_lqr import TrajOptLQR
from robolearn.utils.traj_opt.traj_opt_pi2 import TrajOptPI2

from robolearn.policies.lin_gauss_init import init_lqr, init_pd
from robolearn.policies.policy_prior import ConstantPolicyPrior

# Algorithm
default_gps_hyperparams = {
    'conditions': 1,  # Number of initial conditions
    'iterations': 10,  # Default GPS iterations
    #'train_conditions': [0, 1, 2, 3], # ID? of conditions for training
    #'test_conditions': [0, 1, 2, 3],  # ID? of conditions for testing

    'noisy_samples': True,
    'smooth_noise': True,
    'smooth_noise_var': 1.0,
    'smooth_noise_renormalize': True,


    #'init_traj_distr': None,  # A list of initial LinearGaussianPolicy objects for each condition.
    'init_traj_distr': {
        'type': init_lqr,  # init_lqr, init_pd
        ##'init_gains':  1.0 / PR2_GAINS,
        ##'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
        #'init_var': 1.0,
        #'stiffness': 0.5,  # 1.0
        #'stiffness_vel': 0.25,  # 0.5
        #'final_weight': 50,
        ##'dt': agent['dt'],
        ##'T': agent['T']
    },


    # Indicates if the algorithm requires fitting of the dynamics.
    'fit_dynamics': True,
    # Dynamics hyperaparams.
    'dynamics': {
        'type': DynamicsLRPrior,
        'regularization': 1e-6,
        'prior': {
            'type': DynamicsPriorGMM,
            'max_clusters': 20,
            'min_samples_per_cluster': 40,
            'max_samples': 20,
        },
    },

    # DEFAULT FROM ORIGINAL CODE
    'inner_iterations': 1,  # Number of iterations??? TODO: Check if it is used

    'min_eta': 1e-5,  # Minimum initial lagrange multiplier in DGD for trajectory optimization.

    'kl_step': 0.2,  # TODO: Check default to 5.

    'min_step_mult': 0.01,
    'max_step_mult': 10.0,

    'min_mult': 0.1,
    'max_mult': 5.0,

    # TODO: CHECK IF THEY ARE USED (From mjc_peg_images algorithm)
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-2]),
    'policy_dual_rate': 0.2,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,

    # Trajectory settings.
    'initial_state_var': 1e-6,

    # Trajectory optimization.
    #'traj_opt': None,
    'traj_opt': {
        'type': TrajOptLQR,
        # 'type': TrajOptPI2,
    },
    #'traj_opt': {
    #    'type': TrajOptPI2,
    #    'kl_threshold': 2.0,
    #    'covariance_damping': 2.0,
    #    'min_temperature': 0.001,
    #},

    # Weight of maximum entropy term in trajectory optimization.
    'max_ent_traj': 0.0,

    # Costs.
    'cost': None,  # A list of Cost objects for each condition.

    # Whether or not to sample with agent policy (only for badmm/mdgps).
    'sample_on_policy': False,

}

# BADMM
default_badmm_hyperparams = {
    'inner_iterations': 4,
    'policy_dual_rate': 0.1,
    'policy_dual_rate_covar': 0.0,
    'fixed_lg_step': 0,
    'lg_step_schedule': 10.0,
    'ent_reg_schedule': 0.0,
    'init_pol_wt': 0.01,
    'policy_sample_mode': 'add',
    'exp_step_increase': 2.0,
    'exp_step_decrease': 0.5,
    'exp_step_upper': 0.5,
    'exp_step_lower': 1.0,
}

# MDGPS
default_mdgps_hyperparams = {
    # TODO: remove need for init_pol_wt in MDGPS
    'init_pol_wt': 0.01,
    'policy_sample_mode': 'add',
    # Whether to use 'laplace' or 'mc' cost in step adjustment
    'step_rule': 'laplace',
    'policy_prior': {'type': ConstantPolicyPrior,
                     'strenght': 1e-4},
}

# PIGPS
default_pigps_hyperparams = {
    'init_pol_wt': 0.01,
    'policy_sample_mode': 'add',
    # Dynamics fitting is not required for PIGPS.
    'fit_dynamics': False,
}

# PILQR
default_pilqr_hyperparams = {
    'init_pol_wt': 0.01,
    # Dynamics fitting is not required for PI2 but it is for LQR.
    'fit_dynamics': True,
    # Whether to use 'const' or 'res_percent' in step adjusment
    'step_rule': 'res_percent',
    'step_rule_res_ratio_dec': 0.2,
    'step_rule_res_ratio_inc': 0.05,
    # The following won't work for a different horizon, but that's up to the user.
    'kl_step': np.linspace(0.4, 0.2, 100),
    'max_step_mult': np.linspace(10.0, 5.0, 100),
    'min_step_mult': np.linspace(0.01, 0.5, 100),
    'max_mult': np.linspace(5.0, 2.0, 100),
    'min_mult': np.linspace(0.1, 0.5, 100),
}

# MDGPS-PILQR
default_msgps_pilqr_hyperparams = {
    # TODO: remove need for init_pol_wt in MDGPS
    'init_pol_wt': 0.01,
    'policy_sample_mode': 'add',
    # Dynamics fitting is not required for PI2 but it is for LQR.
    'fit_dynamics': True,
    # Whether to use 'const' or 'res_percent' in step adjusment
    'step_rule': 'res_percent',
    'step_rule_res_ratio_dec': 0.2,
    'step_rule_res_ratio_inc': 0.05,
    # The following won't work for a different horizon, but that's up to the user.
    'kl_step': np.linspace(0.4, 0.2, 100),
    'max_step_mult': np.linspace(10.0, 5.0, 100),
    'min_step_mult': np.linspace(0.01, 0.5, 100),
    'max_mult': np.linspace(5.0, 2.0, 100),
    'min_mult': np.linspace(0.1, 0.5, 100),
}

default_multi_mdgps_hyperparams = {
}
