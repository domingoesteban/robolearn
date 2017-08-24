"""
Default configuration for trajectory optimization.
Author: C. Finn et al. Code in https://github.com/cbfinn/gps
"""


# TrajOptLQR
default_traj_opt_lqr_hyperparams = {
    # Dual variable updates for non-PD Q-function.
    'del0': 1e-4,
    #'eta_error_threshold': 1e16,  # TODO: REMOVE, it is not used
    'min_eta': 1e-8,
    'max_eta': 1e16,
    'cons_per_step': False,  # Whether or not to enforce separate KL constraints at each time step.
    'use_prev_distr': False,  # Whether or not to measure expected KL under the previous traj distr.
    'update_in_bwd_pass': True,  # Whether or not to update the TVLG controller during the bwd pass.
}

# TrajOptPI2
default_traj_opt_pi2_hyperparams = {
    'kl_threshold': 1.0,   # KL-divergence threshold between old and new policies.
    'covariance_damping': 2.0,  # If greater than zero, covariance is computed as a multiple of the old covariance.
                                # Multiplier is taken to the power (1 / covariance_damping). If greater than one, slows
                                # down convergence and keeps exploration noise high for more iterations.
    'min_temperature': 0.001,  # Minimum bound of the temperature optimization for the soft-max probabilities of the
                               # policy samples.
    'use_sumexp': False,
    'pi2_use_dgd_eta': False,
    'pi2_cons_per_step': True,
    'min_eta': 1e-8,
    'max_eta': 1e16,
    'del0': 1e-4,
}

# TrajOptPILQR
default_traj_opt_pilqr_hyperparams = {
    'use_lqr_actions': True,
    'cons_per_step': True,
}

# TrajOptDREPS
default_traj_opt_dreps_hyperparams = {
    'epsilon': 1.0,   # KL-divergence threshold between old and new policies.
    'xi': 5.0,
    'chi': 2.0,
    'dreps_cons_per_step': True,
}


# TrajOptmDREPS
default_traj_opt_mdreps_hyperparams = {
    'epsilon': 1.0,   # KL-divergence threshold between old and new policies.
    'xi': 5.0,
    'chi': 2.0,
    'dreps_cons_per_step': True,
}
