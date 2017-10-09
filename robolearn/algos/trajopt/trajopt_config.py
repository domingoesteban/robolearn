# ILQR
DEFAULT_ILQR_HYPERPARAMS = {
    'inner_iterations': 1,
}

# PI2
DEFAULT_PI2_HYPERPARAMS = {
    'inner_iterations': 1,
    # Dynamics fitting is not required for PI2.
    'fit_dynamics': False,
}

# MOTO
DEFAULT_MOTO_HYPERPARAMS = {
    'inner_iterations': 1,
}

# DREPS
DEFAULT_DREPS_HYPERPARAMS = {
    'inner_iterations': 1,
    'epsilon': 0.1,
    'xi': 0.1,
    'chi': 0.1,
}

# mDREPS
DEFAULT_MDREPS_HYPERPARAMS = {
    'inner_iterations': 1,
    'epsilon': 0.1,
    'xi': 0.1,
    'chi': 0.1,
}
