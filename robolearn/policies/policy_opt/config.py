"""
Default configuration for policy optimization.
Authors: Finn et al
Modified by: robolearn collaborators
"""

# config options shared by both caffe and tf.
GENERIC_CONFIG = {
}


POLICY_OPT_TF = {
    # Initialization.
    'init_var': 0.1,  # Initial policy variance.
    'ent_reg': 0.0,  # Entropy regularizer.
    # Solver hyperparameters.
    'iterations': 5000,  # Number of iterations per inner iteration.
    'batch_size': 25,
    'lr': 0.001,  # Base learning rate (by default it's fixed).
    'lr_policy': 'fixed',  # Learning rate policy.
    'momentum': 0.9,  # Momentum.
    'weight_decay': 0.005,  # Weight decay.
    'solver_type': 'Adam',  # Solver type (e.g. 'SGD', 'Adam', etc.).
    # set gpu usage.
    'use_gpu': True,  # Whether or not to use the GPU for caffe training.
    'gpu_id': 0,
    'random_seed': 1,
    # Other hyperparameters.
    'copy_param_scope': '',
    'fc_only_iterations': 0,
    'gpu_mem_percentage': 0.4,
}


POLICY_OPT_RANDOM = {}
