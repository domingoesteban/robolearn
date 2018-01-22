from robolearn.algos.gps.multi_gps import MULTIGPS

gps_hyperparams = {
    'T': int(EndTime/Ts),  # Total points
    'dt': Ts,
    'iterations': 25,  # 100  # 2000  # GPS episodes, "inner iterations" --> K iterations
    'test_after_iter': test_after_iter,  # If test the learned policy after an iteration in the RL algorithm
    'test_samples': 2,  # Samples from learned policy after an iteration PER CONDITION (only if 'test_after_iter':True)
    # Samples
    'num_samples': 6,  # 20  # Samples for exploration trajs --> N samples
    'noisy_samples': True,
    'sample_on_policy': sample_on_policy,  # Whether generate on-policy samples or off-policy samples
    #'noise_var_scale': np.array([5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2, 5.0e-2]),  # Scale to Gaussian noise: N(0,1)*sqrt(noise_var_scale)
    #'noise_var_scale': np.array([1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1, 1.0e-1])*10,  # Scale to Gaussian noise: N(0,1)*sqrt(noise_var_scale)
    'smooth_noise': True,  # Apply Gaussian filter to noise generated
    #'smooth_noise_var': 5.0e+0,  # np.power(2*Ts, 2), # Variance to apply to Gaussian Filter. In Kumar (2016) paper, it is the std dev of 2 Ts
    'smooth_noise_var': 8.0e+0,  # np.power(2*Ts, 2), # Variance to apply to Gaussian Filter. In Kumar (2016) paper, it is the std dev of 2 Ts
    'smooth_noise_renormalize': True,  # Renormalize smooth noise to have variance=1
    'noise_var_scale': np.ones(len(bigman_params['joint_ids'][body_part_active])),  # Scale to Gaussian noise: N(0, 1)*sqrt(noise_var_scale), only if smooth_noise_renormalize
    'cost': cost_sum,
    # Conditions
    'conditions': len(bigman_env.get_conditions()),  # Total number of initial conditions
    'train_conditions': range(len(bigman_env.get_conditions())),  # Indexes of conditions used for training
    'test_conditions': range(len(bigman_env.get_conditions())),  # Indexes of conditions used for testing
    # KL step (epsilon)
    'kl_step': 0.2,  # Kullback-Leibler step (base_step)
    'min_step_mult': 0.01,  # Min possible value of step multiplier (multiplies kl_step in LQR)
    'max_step_mult': 10.0,  # Previous 23/08 -> 1.0 #3 # 10.0,  # Max possible value of step multiplier (multiplies kl_step in LQR)
    # Others
    'gps_algo_hyperparams': gps_algo_hyperparams,
    'init_traj_distr': init_traj_distr,
    'fit_dynamics': True,
    'dynamics': learned_dynamics,
    'initial_state_var': 1e-6,  #1e-2,# 1e-6,  # Max value for x0sigma in trajectories  # TODO: CHECK THIS VALUE, maybe it is too low
    'traj_opt': traj_opt_method,
    'max_ent_traj': 0.0,  # Weight of maximum entropy term in trajectory optimization  # CHECK THIS VALUE!!!, I AM USING ZERO!!
    'use_global_policy': use_global_policy,
    'data_files_dir': data_files_dir,
}

learn_algo = MULTIGPS(bigman_agents, bigman_env, **gps_hyperparams)
