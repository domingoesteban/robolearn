# ILQR
default_ilqr_hyperparams = {
    'inner_iterations': 1,
}

# PI2
default_pi2_hyperparams = {
    'inner_iterations': 1,
    # Dynamics fitting is not required for PI2.
    'fit_dynamics': False,
}

# MOTO
default_moto_hyperparams = {
    'inner_iterations': 1,
}

# DREPS
default_dreps_hyperparams = {
    'inner_iterations': 1,
    'epsilon': 0.1,
    'xi': 0.1,
    'chi': 0.1,
}

# mDREPS
default_mdreps_hyperparams = {
    'inner_iterations': 1,
    'epsilon': 0.1,
    'xi': 0.1,
    'chi': 0.1,
}
