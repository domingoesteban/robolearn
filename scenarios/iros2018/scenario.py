from __future__ import print_function

import os
import random
import signal
import yaml
from builtins import input

import numpy as np
from robolearn.v010.envs.pusher3dof import Pusher3DofBulletEnv
from robolearn.v010.utils.sample.sampler import Sampler

from robolearn.v010.agents import GPSAgent
from robolearn.v010.algos.gps.dual_gps import DualGPS
# Costs
from robolearn.v010.costs.cost_action import CostAction
# from robolearn.costs.cost_fk import CostFK
from robolearn.v010.costs.cost_state import CostState
from robolearn.v010.costs.cost_safe_distance import CostSafeDistance
from robolearn.v010.costs.cost_state_difference import CostStateDifference
from robolearn.v010.costs.cost_safe_state_difference import CostSafeStateDifference
from robolearn.v010.costs.cost_sum import CostSum
from robolearn.v010.costs.cost_utils import RAMP_FINAL_ONLY, RAMP_CONSTANT
from robolearn.v010.costs.cost_utils import evall1l2term
# from robolearn.envs import BigmanEnv
from robolearn.v010.policies.lin_gauss_init import init_pd, init_dual_demos, init_lqr, init_pd_tgt
from robolearn.v010.policies.policy_opt.policy_opt_tf import PolicyOptTf
from robolearn.v010.policies.policy_opt.tf_models import tf_network
from robolearn.v010.policies.policy_prior import ConstantPolicyPrior  # For MDGPS
from robolearn.v010.policies.lin_gauss_policy import LinearGaussianPolicy
from robolearn.v010.policies.dataset_policy import DataSetPolicy
from robolearn.v010.utils.dynamics.dynamics_lr_prior import DynamicsLRPrior
from robolearn.v010.utils.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from robolearn.v010.utils.print_utils import change_print_color
from robolearn.v010.utils.transformations_utils import create_quat_pose
from robolearn.v010.utils.traj_opt.dualist_traj_opt import DualistTrajOpt

np.set_printoptions(precision=4, suppress=True, linewidth=1000)


def kill_everything(_signal=None, _frame=None):
    print("\n\033[1;31mThe script has been killed by the user!!")
    os._exit(1)


signal.signal(signal.SIGINT, kill_everything)


class Scenario(object):
    """Defines a RL scenario (environment, agent and learning algorithm)

    """
    def __init__(self, hyperparams):

        self.hyperparams = hyperparams

        # Task Parameters
        yaml_path = os.path.dirname(__file__) + '/hyperparameters/' + \
                    self.hyperparams['scenario'] + '.yaml'
        assert(os.path.exists(yaml_path))
        with open(yaml_path, 'r') as f:
            self.task_params = yaml.load(f)

        Tend = self.task_params['Tend']
        Ts = self.task_params['Ts']
        self.task_params['T'] = int(Tend/Ts)

        if self.hyperparams['render']:
            self.task_params['render'] = self.hyperparams['render']

        self.task_params['seed'] = self.hyperparams['seed']

        # Numpy max
        os.environ['OMP_NUM_THREADS'] = str(self.task_params['np_threads'])

        # Environment
        self.env = self.create_environment()

        self.action_dim = self.env.action_dim
        self.state_dim = self.env.state_dim
        self.obs_dim = self.env.obs_dim

        # Agent
        self.agent = self.create_agent()

        # Costs
        self.cost = self.create_cost()

        # Initial Conditions
        self.init_cond = self.create_init_conditions()

        # Learning Algorithm
        self.learn_algo = self.create_learning_algo()

    def create_environment(self):
        """Instantiate an specific RL environment to interact with.

        Returns:
            RL environment

        """
        change_print_color.change('BLUE')
        print("\nCreating Environment...")

        # Environment parameters
        env_with_img = False
        rdn_tgt_pos = False
        render = self.task_params['render']
        obs_like_mjc = self.task_params['obs_like_mjc']
        ntargets = self.task_params['ntargets']
        tgt_weights = self.task_params['tgt_weights']
        tgt_positions = self.task_params['tgt_positions']
        tgt_types = self.task_params['tgt_types']
        sim_timestep = 0.001
        frame_skip = int(self.task_params['Ts']/sim_timestep)

        env = Pusher3DofBulletEnv(render=render, obs_with_img=env_with_img,
                                  obs_mjc_gym=obs_like_mjc, ntargets=ntargets,
                                  rdn_tgt_pos=rdn_tgt_pos, tgt_types=tgt_types,
                                  sim_timestep=sim_timestep,
                                  frame_skip=frame_skip,
                                  obs_distances=True)

        env.set_tgt_cost_weights(tgt_weights)
        env.set_tgt_pos(tgt_positions)

        print("Environment:%s OK!." % type(env).__name__)

        return env

    def create_agent(self):
        """Instantiate the RL agent who interacts with the environment.

        Returns:
            RL agent

        """
        change_print_color.change('CYAN')
        print("\nCreating Agent...")

        policy_params = {
            'network_model': tf_network,  # tf_network, multi_modal_network, multi_modal_network_fp
            'network_params': {
                'n_layers': 2,  # Number of Hidden layers
                'dim_hidden': [40, 40], #[32, 32], # List of size per n_layers
                'obs_names': self.env.get_obs_info()['names'],
                'obs_dof': self.env.get_obs_info()['dimensions'],  # DoF for observation data tensor
            },
            # Initialization.
            'init_var': 1.0,  #0.1,  # Initial policy variance.
            'ent_reg': 0.0,  # Entropy regularizer (Used to update policy variance)
            # Solver hyperparameters.
            'iterations': self.task_params['tf_iterations'],  # Number of iterations per inner iteration (Default:5000).
            'batch_size': 64, #32,  # 15
            'lr': 1e-5, #1e-4,  #1e-3 # Base learning rate (by default it's fixed).
            'lr_policy': 'fixed',  # Learning rate policy.
            'momentum': 0.9,  # Momentum.
            'weight_decay': 0.005,  # Weight decay to prevent overfitting.
            'solver_type': 'Adam',  # Solver type (e.g. 'SGD', 'Adam', 'RMSPROP', 'MOMENTUM', 'ADAGRAD').
            # GPU usage.
            'use_gpu': self.task_params['use_gpu'],  # Whether or not to use the GPU for training.
            'gpu_id': 0,
            'gpu_mem_percentage': self.task_params['gpu_mem_percentage'],
            # Training data.
            'fc_only_iterations': 0,  # Iterations of only FC before normal training
            # Others.
            'random_seed': self.hyperparams['seed'] \
                if self.task_params['tf_seed'] == -1 \
                else self.task_params['tf_seed'],  # TF random seed
            'log_dir': self.hyperparams['log_dir'],
            # 'weights_file_prefix': EXP_DIR + 'policy',
        }

        policy_opt = {
            'type': PolicyOptTf,
            'hyperparams': policy_params
        }

        agent = GPSAgent(act_dim=self.action_dim, obs_dim=self.obs_dim,
                         state_dim=self.state_dim, policy_opt=policy_opt,
                         agent_name='agent'+str('%02d' % self.hyperparams['run_num']))
        print("Agent:%s OK\n" % type(agent).__name__)

        return agent

    def create_cost(self):
        """Instantiate the cost that evaluates the RL agent performance.

        Returns:
            Cost Function

        """
        change_print_color.change('GRAY')
        print("\nCreating Costs...")

        # Action Cost
        weight = 1e0  # 1e-4
        target = None
        act_cost = {
            'type': CostAction,
            'wu': np.ones(self.action_dim) * weight,
            'target': target,   # Target action value
        }

        # # FK Cost
        # fk_l1_cost = {
        #     'type': CostFK,
        #     'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
        #     'target_pose': target_distance_hand,
        #     'tgt_data_type': 'state',  # 'state' or 'observation'
        #     'tgt_idx': bigman_env.get_state_info(name='distance_hand')['idx'],
        #     'op_point_name': hand_name,
        #     'op_point_offset': hand_offset,
        #     'joints_idx': bigman_env.get_state_info(name='link_position')['idx'],
        #     'joint_ids': bigman_params['joint_ids'][body_part_active],
        #     'robot_model': robot_model,
        #     'wp': np.array([3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
        #     'evalnorm': evall1l2term,
        #     #'evalnorm': evallogl2term,
        #     'l1': 1.0,  # 1.0,  # 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
        #     'l2': 0.0,  # 1.0,  #1.0e-3,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
        #     'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
        #     'wp_final_multiplier': 1,  # 10
        # }
        #
        # fk_l2_cost = {
        #     'type': CostFK,
        #     'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
        #     'target_pose': target_distance_hand,
        #     'tgt_data_type': 'state',  # 'state' or 'observation'
        #     'tgt_idx': bigman_env.get_state_info(name='distance_hand')['idx'],
        #     'op_point_name': hand_name,
        #     'op_point_offset': hand_offset,
        #     'joints_idx': bigman_env.get_state_info(name='link_position')['idx'],
        #     'joint_ids': bigman_params['joint_ids'][body_part_active],
        #     'robot_model': robot_model,
        #     # 'wp': np.array([1.0, 1.0, 1.0, 0.7, 0.8, 0.6]),  # one dim less because 'quat' error | 1)orient 2)pos
        #     #'wp': np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
        #     'wp': np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),  # one dim less because 'quat' error | 1)orient 2)pos
        #     'evalnorm': evall1l2term,
        #     #'evalnorm': evallogl2term,
        #     'l1': 0.0,  # 1.0,  # 1.0,  # Weight for l1 norm: log(d^2 + alpha) --> Lorentzian rho-function Precise placement at the target
        #     'l2': 1.0,  # 1.0,  #1.0e-3,  # Weight for l2 norm: d^2 --> Encourages to quickly get the object in the vicinity of the target
        #     'alpha': 1.0e-5,  # e-5,  # Constant added in square root in l1 norm
        #     'wp_final_multiplier': 1,  # 10
        # }

        state_diff_weights = self.task_params['state_diff_weights']
        l1_l2_weights = np.array(self.task_params['l1_l2'])
        inside_cost = self.task_params['inside_cost']


        # State costs
        target_distance_object = np.zeros(3)
        state_cost_distance = {
            'type': CostState,
            'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
            'evalnorm': evall1l2term,  # TODO: ALWAYS USE evall1l2term
            'l1': l1_l2_weights[0],  # Weight for l1 norm
            'l2': l1_l2_weights[1],  # Weight for l2 norm
            'alpha': 1e-10,  # Constant added in square root in l1 norm
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'tgt0': {
                    'wp': np.array(state_diff_weights),  # State weights - must be set.
                    'target_state': target_distance_object,  # Target state - must be set.
                    'average': None,
                    'data_idx': self.env.get_state_info(name='tgt0')['idx']
                },
            },
        }

        state_final_cost_distance = {
            'type': CostState,
            'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
            'evalnorm': evall1l2term,  # TODO: ALWAYS USE evall1l2term
            'l1': l1_l2_weights[0],  # Weight for l1 norm
            'l2': l1_l2_weights[1],  # Weight for l2 norm
            'alpha': 1e-10,  # Constant added in square root in l1 norm
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'tgt0': {
                    'wp': np.array(state_diff_weights),  # State weights - must be set.
                    'target_state': target_distance_object,  # Target state - must be set.
                    'average': None,
                    'data_idx': self.env.get_state_info(name='tgt0')['idx']
                },
            },
        }

        safe_radius = 0.15
        cost_safe_distance = {
            'type': CostSafeDistance,
            'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'tgt1': {
                    'wp': np.array([1.0, 1.0]),  # State weights - must be set.
                    'idx_to_use': [0, 1],  # Only X and Y
                    'data_idx': self.env.get_state_info(name='tgt1')['idx'][:2],
                    'safe_distance': np.sqrt([safe_radius**2/2, safe_radius**2/2]),
                    'outside_cost': np.array([0.0, 0.0]),
                    'inside_cost': np.array([inside_cost, inside_cost]),
                },
            },
        }
        cost_final_safe_distance = {
            'type': CostSafeDistance,
            'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time.
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'tgt1': {
                    'wp': np.array([1.0, 1.0]),  # State weights - must be set.
                    'idx_to_use': [0, 1],  # Only X and Y
                    'data_idx': self.env.get_state_info(name='tgt1')['idx'][:2],
                    'safe_distance': np.sqrt([safe_radius**2/2, safe_radius**2/2]),
                    'outside_cost': np.array([0.0, 0.0]),
                    'inside_cost': np.array([inside_cost, inside_cost]),
                },
            },
        }

        """
        cost_state_difference = {
            'type': CostStateDifference,
            'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
            'evalnorm': evall1l2term,  # TODO: ALWAYS USE evall1l2term
            'l1': l1_l2_weights[0],  # Weight for l1 norm
            'l2': l1_l2_weights[1],  # Weight for l2 norm
            'alpha': 1e-10,  # Constant added in square root in l1 norm
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'ee': {
                    'data_idx': self.env.get_state_info(name='ee')['idx'],
                    'idx_to_use': [0, 1, 2],  # All: X, Y, theta
                    'wp': np.array(state_diff_weights),  # State weights - must be set.
                    'average': None,
                    'target_state': 'tgt0',  # Target state - must be set.
                    'tgt_idx': self.env.get_state_info(name='tgt0')['idx'],
                },
            },
        }

        cost_final_state_difference = {
            'type': CostStateDifference,
            'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
            'evalnorm': evall1l2term,  # TODO: ALWAYS USE evall1l2term
            'l1': l1_l2_weights[0],  # Weight for l1 norm
            'l2': l1_l2_weights[1],  # Weight for l2 norm
            'alpha': 1e-10,  # Constant added in square root in l1 norm
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'ee': {
                    'data_idx': self.env.get_state_info(name='ee')['idx'],
                    'idx_to_use': [0, 1, 2],  # All: X, Y, theta
                    'wp': np.array(state_diff_weights),  # State weights - must be set.
                    'average': None,
                    'target_state': 'tgt0',  # Target state - must be set.
                    'tgt_idx': self.env.get_state_info(name='tgt0')['idx'],
                },
            },
        }

        cost_safe_state_difference = {
            'type': CostSafeStateDifference,
            'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'ee': {
                    'data_idx': self.env.get_state_info(name='ee')['idx'][:2],
                    'idx_to_use': [0, 1],  # Only X and Y
                    'wp': np.array([1.0, 1.0]),  # State weights - must be set.
                    'target_state': 'tgt1',  # Target state - must be set.
                    'tgt_idx': self.env.get_state_info(name='tgt1')['idx'][:2],
                    'safe_distance': np.sqrt([safe_radius**2/2, safe_radius**2/2]),
                    'outside_cost': np.array([0.0, 0.0]),
                    'inside_cost': np.array([inside_cost, inside_cost]),
                },
            },
        }

        cost_final_safe_state_difference = {
            'type': CostSafeStateDifference,
            'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time.
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'ee': {
                    'data_idx': self.env.get_state_info(name='ee')['idx'][:2],
                    'idx_to_use': [0, 1],  # Only X and Y
                    'wp': np.array([1.0, 1.0]),  # State weights - must be set.
                    'target_state': 'tgt1',  # Target state - must be set.
                    'tgt_idx': self.env.get_state_info(name='tgt1')['idx'][:2],
                    'safe_distance': np.sqrt([safe_radius**2/2, safe_radius**2/2]),
                    'outside_cost': np.array([0.0, 0.0]),
                    'inside_cost': np.array([inside_cost, inside_cost]),
                },
            },
        }
        """


        # Sum costs
        # costs_and_weights = [(act_cost, 1.0e-1),
        des_weights = self.task_params['cost_weights']
        print('Costs weights:', des_weights)
        costs_and_weights = [(act_cost, des_weights[0]),
                             # (cost_state_difference, des_weights[1]),
                             # (cost_final_state_difference, des_weights[2]),
                             # (cost_safe_state_difference, des_weights[3]),
                             # (cost_final_safe_state_difference, des_weights[4]),
                             (state_cost_distance, des_weights[1]),
                             (state_final_cost_distance, des_weights[2]),
                             (cost_safe_distance, des_weights[3]),
                             (cost_final_safe_distance, des_weights[4]),
                             # WORKING:
                             # (cost_safe_distance, 1.0e+1),
                             # (state_cost_distance, 5.0e-0),
                             # (state_final_cost_distance, 1.0e+3),
                             ]

        cost_sum = {
            'type': CostSum,
            'costs': [cw[0] for cw in costs_and_weights],
            'weights': [cw[1] for cw in costs_and_weights],
        }

        return cost_sum

    def create_init_conditions(self):
        """Defines the initial conditions for the environment.

        Returns:
            Environment' initial conditions.

        """
        change_print_color.change('MAGENTA')
        print("\nCreating Initial Conditions...")

        if bool(self.task_params['fix_init_conds']):
            initial_cond = self.task_params['init_cond']

            ddof = 3  # Data dof (file): x, y, theta
            pdof = 3  # Pose dof (env): x, y, theta
            ntgt = self.task_params['ntargets']

            for cc, cond in enumerate(initial_cond):
                env_condition = np.zeros(self.env.obs_dim)
                env_condition[:self.env.action_dim] = np.deg2rad(cond[:3])
                cond_idx = 2*self.env.action_dim
                data_idx = self.env.action_dim # We now this is 3
                for tt in range(self.task_params['ntargets']):
                    tgt_data = cond[data_idx:data_idx+ddof]
                    # tgt_pose = create_quat_pose(pos_x=tgt_data[0],
                    #                             pos_y=tgt_data[1],
                    #                             pos_z=z_fix,
                    #                             rot_yaw=np.deg2rad(tgt_data[2]))
                    # env_condition[cond_idx:cond_idx+pdof] = tgt_pose
                    tgt_data[2] = np.deg2rad(tgt_data[2])
                    env_condition[cond_idx:cond_idx+pdof] = tgt_data
                    cond_idx += pdof
                    data_idx += ddof

                self.env.add_custom_init_cond(env_condition)
        else:
            cond_mean = np.array(self.task_params['mean_init_cond'])
            cond_std = np.array(self.task_params['std_init_cond'])

            tgt_idx = [6, 7, 8]
            obst_idx = [9, 10]  # Only X-Y is random

            # Set the np seed
            np.random.seed(self.task_params['seed'])

            all_init_conds = np.zeros((int(self.task_params['n_init_cond']), 9))

            for cc in range(int(self.task_params['n_init_cond'])):
                rand_data = np.random.rand(len(cond_mean))
                init_cond = cond_std*rand_data + cond_mean
                joint_pos = np.deg2rad(init_cond[:3])

                env_condition = np.zeros(self.env.obs_dim)
                env_condition[:self.env.action_dim] = joint_pos
                # env_condition[obst_idx] = init_cond[3:]

                # Temporally hack for getting ee _object
                self.env.add_custom_init_cond(env_condition)
                self.env.reset(condition=-1)
                # obs = self.env.get_observation()
                des_tgt = self.env.get_ee_pose()
                self.env.clear_custom_init_cond(-1)

                env_condition[:3] = np.deg2rad(self.task_params['init_joint_pos'])
                env_condition[tgt_idx] = des_tgt
                env_condition[obst_idx] = init_cond[3:]

                # Save the values so we can print them on file
                all_init_conds[cc, :3] = np.rad2deg(env_condition[:3])
                all_init_conds[cc, 3:5] = env_condition[6:8]
                all_init_conds[cc, 5] = np.rad2deg(env_condition[8])
                all_init_conds[cc, 6:8] = env_condition[9:11]
                all_init_conds[cc, 8] = np.rad2deg(env_condition[11])

                # Now add the target properly
                print('INIT COND', env_condition)
                self.env.add_custom_init_cond(env_condition)

            init_conds_file = os.path.dirname(__file__) + '/init_conds.csv'
            np.set_printoptions(threshold=np.inf,
                                linewidth=np.inf)  # turn off summarization, line-wrapping
            with open(init_conds_file, 'w') as f:
                f.write(np.array2string(all_init_conds, separator=', '))

        return self.env.get_conditions()

    def create_learning_algo(self):
        """Instantiates the RL algorithm

        Returns:
            Learning algorithm

        """
        change_print_color.change('YELLOW')
        print("\nConfiguring learning algorithm...\n")

        # Dynamics
        learned_dynamics = {'type': DynamicsLRPrior,
                            'regularization': 1e-6,
                            'prior': {
                                'type': DynamicsPriorGMM,
                                'max_clusters': 20,  # Maximum number of clusters to fit.
                                'min_samples_per_cluster': 40,  # Minimum samples per cluster.
                                'max_samples': 20,  # Max. number of trajectories to use for fitting the GMM at any given time.
                                'strength': 1.0,  # Adjusts the strength of the prior.
                            },
                            }
        # init_traj_distr = {'type': init_pd_tgt,
        #                    'init_var': np.array([1., 1., 1.])*1.0,
        #                    # 'pos_gains': 0.001,  # float or array
        #                    # 'vel_gains_mult': 0.01,  # Velocity gains multiplier on pos_gains
        #                    'pos_gains': np.array([1, 0.1, 0.1])*1e-1, # float or array
        #                    'vel_gains_mult': 0.01,  # Velocity gains multiplier on pos_gains
        #                    'init_action_offset': None,
        #                    'dJoints': self.env.get_total_joints(),  # Total joints in state
        #                    'state_to_pd': 'joints',  # Joints
        #                    'idx_tgt': np.array([9, 10]),
        #                    'idx_ee': np.array([6, 7]),
        #                    }

        init_traj_distr = {'type': init_pd,
                           'init_var': np.array([1., 1., 1.])*1.0,
                           # 'pos_gains': 0.001,  # float or array
                           # 'vel_gains_mult': 0.01,  # Velocity gains multiplier on pos_gains
                           'pos_gains': 0.01,  # float or array
                           'vel_gains_mult': 0.01,  # Velocity gains multiplier on pos_gains
                           'init_action_offset': None,
                           'dJoints': self.env.get_total_joints(),  # Total joints in state
                           'state_to_pd': 'joints',  # Joints
                           'dDistance': 6,
                           }
        # init_traj_distr = {
        #     'type': init_lqr,
        #     'init_gains': np.array([1., 1., 1.])*1.0,
        #     'init_acc': np.zeros(3),
        #     'init_var': 1.0,
        #     'stiffness': 1.0,
        #     'stiffness_vel': 0.5,
        #     'final_weight': 1, #50.0,
        # }

        # Trajectory Optimization Method
        traj_opt_method = {
            'type': DualistTrajOpt,
            'bad_const': self.task_params['consider_bad'],  # Use bad constraints
            'good_const': self.task_params['consider_good'],  # Use good constraints
            'del0': 1e-4,  # Eta updates for non-SPD Q-function (non-SPD correction step).
            'del0_bad': 1e-8,  # Nu updates for non-SPD Q-function (non-SPD correction step).
            'del0_good': 1e-4,  # Omega updates for non-SPD Q-function (non-SPD correction step).
            'min_eta': 1e-8,  # At min_eta, kl_div > kl_step
            'max_eta': 1e16,  # At max_eta, kl_div < kl_step
            'min_nu': 1e-8,  # At min_nu, kl_div > kl_step
            'max_nu': self.task_params['max_nu'],  # At max_nu, kl_div < kl_step,
            'min_omega': 1e-8,  # At min_omega, kl_div > kl_step
            'max_omega': self.task_params['max_omega'],  #1e16,  # At max_omega, kl_div < kl_step
            'step_tol': 0.1,
            'bad_tol': 0.1,
            'good_tol': 0.1,
            'cons_per_step': False,  # Whether or not to enforce separate KL constraints at each time step. #TODO: IF TRUE, MAYBE IT DOES WORK WITH MDGPS because it doesn't consider dual vars
            'use_prev_distr': False,  # Whether or not to measure expected KL under the previous traj distr.
            'update_in_bwd_pass': True,  # Whether or not to update the TVLG controller during the bwd pass.
            'adam_alpha': 0.5,
            'adam_max_iter': 500,
            'weight_bad': self.task_params['weight_bad'],
            'weight_good': self.task_params['weight_good'],
            }

        good_trajs = None
        bad_trajs = None
        dmdgps_hyperparams = {
            'inner_iterations': self.task_params['inner_iterations'],  # Times the trajectories are updated
            # G/B samples selection
            'good_samples': good_trajs,  # Good samples demos
            'bad_samples': bad_trajs,  # Bad samples demos
            'n_good_samples': self.task_params['n_good_samples'],  # Number of good samples per each trajectory
            'n_bad_samples': self.task_params['n_bad_samples'],  # Number of bad samples per each trajectory
            'n_good_buffer': self.task_params['n_good_buffer'],  # Number of good samples in the buffer
            'n_bad_buffer': self.task_params['n_bad_buffer'],  # Number of bad samples in the buffer
            'good_traj_selection_type': self.task_params['good_traj_selection_type'],  # 'always', 'only_traj'
            'bad_traj_selection_type': self.task_params['bad_traj_selection_type'],  # 'always', 'only_traj'
            'bad_costs': self.task_params['bad_costs'],
            # G/B samples fitting
            'duality_dynamics_type': 'duality',  # Samples to use to update the dynamics 'duality', 'iteration'
            # Initial dual variables
            'init_eta': 0.1,#4.62,
            'init_nu': 1e-8,
            'init_omega': 1e-8,
            # KL step (epsilon)
            'step_rule': 'laplace',  # Whether to use 'laplace' or 'mc' cost in step adjustment
            'kl_step': self.task_params['kl_step'],  # Kullback-Leibler step (base_step)
            'min_step_mult': 0.5,  # Min possible value of step multiplier (multiplies kl_step)
            'max_step_mult': 3.0,  # Max possible value of step multiplier (multiplies kl_step)
            # KL bad (xi)
            'kl_bad': self.task_params['kl_bad'], #4.2  # Xi KL base value | kl_div_b >= kl_bad
            'min_bad_mult': 0.1,  # Min possible value of step multiplier (multiplies base_kl_bad)
            'max_bad_mult': 3.0,  # Max possible value of step multiplier (multiplies base_kl_bad)
            # KL good (chi)
            'kl_good': self.task_params['kl_good'],  #2.0,  # Chi KL base value  | kl_div_g <= kl_good
            'min_good_mult': 0.1,  # Min possible value of step multiplier (multiplies base_kl_good)
            'max_good_mult': 10.0,  # Max possible value of step multiplier (multiplies base_kl_good)
            # LinearPolicy 'projection'
            'init_pol_wt': 0.01,  # TODO: remove need for init_pol_wt in MDGPS (It should not work with MDGPS)
            'policy_sample_mode': 'add',  # Mode to update dynamics prior (Not used in ConstantPolicyPrior)
            'policy_prior': {'type': ConstantPolicyPrior,
                             'strength': 1e-4,
                             },
            'min_bad_var': np.array([3.0, 3.0, 3.0])*1.0e-02,
            'min_good_var': np.array([3.0, 3.0, 3.0])*1.0e-02,
            # 'min_bad_var': np.array([3.0, 3.0, 3.0])*1.0e-02,
            # 'min_good_var': np.array([3.0, 3.0, 3.0])*1.0e-02,
            # SL step
            'forget_bad_samples': self.task_params['forget_bad_samples'],
            # TEMP Hyperparams
            'min_bad_rel_diff': self.task_params['min_bad_rel_diff'],
            'max_bad_rel_diff': self.task_params['max_bad_rel_diff'],
            'mult_bad_rel_diff': self.task_params['mult_bad_rel_diff'],
            'good_fix_rel_multi': self.task_params['good_fix_rel_multi'],
            }

        gps_hyperparams = {
            'T': self.task_params['T'],  # Total points
            'dt': self.task_params['Ts'],
            'iterations': self.task_params['iterations'],  # GPS episodes --> K iterations
            'test_after_iter': self.task_params['test_after_iter'],  # If test the learned policy after an iteration in the RL algorithm
            'test_samples': self.task_params['test_n_samples'],  # Samples from learned policy after an iteration PER CONDITION (only if 'test_after_iter':True)
            'sample_pol_first_itr': True,
            'sample_real_time': False,
            # Samples
            'num_samples': self.task_params['num_samples'],  # Samples for exploration trajs --> N samples
            'noisy_samples': True,
            'seed': self.task_params['seed'],
            'sample_on_policy': self.task_params['sample_on_policy'],  # Whether generate on-policy samples or off-policy samples
            'smooth_noise': True,  # Apply Gaussian filter to noise generated
            'smooth_noise_var': 5.0e+0,  # np.power(2*Ts, 2), # Variance to apply to Gaussian Filter. In Kumar (2016) paper, it is the std dev of 2 Ts
            'smooth_noise_renormalize': True,  # Renormalize smooth noise to have variance=1
            'noise_var_scale': 1.e-1*np.ones(self.action_dim),  # Scale to Gaussian noise: N(0, 1)*sqrt(noise_var_scale), only if smooth_noise_renormalize
            # Cost
            'cost': self.cost,
            # Conditions
            'conditions': len(self.init_cond),  # Total number of initial conditions
            'train_conditions': self.task_params['train_cond'],  # Indexes of conditions used for training
            'test_conditions': self.task_params['test_cond'],  # Indexes of conditions used for testing
            # TrajDist
            'init_traj_distr': init_traj_distr,
            'fit_dynamics': True,
            'dynamics': learned_dynamics,
            'initial_state_var': 1e-2,  # Max value for x0sigma in trajectories
            # TrajOpt
            'traj_opt': traj_opt_method,
            'max_ent_traj': 0.0,  # Weight of maximum entropy term in trajectory optimization #TODO: CHECK THIS VALUE
            # Others
            'algo_hyperparams': dmdgps_hyperparams,
            'data_files_dir': self.hyperparams['log_dir'],
        }

        return DualGPS(self.agent, self.env, **gps_hyperparams)

    def train(self, itr_load=None):
        """Train the RL agent with the learning algorithm.

        Args:
            itr_load: Iteration number with which to start

        Returns:
            bool: True for success, False otherwise.

        """
        change_print_color.change('WHITE')
        return self.learn_algo.run(itr_load)

    def test_policy(self, pol_type='global', condition=0, iteration=-1):
        """Test the RL agent using the policy learned in the specificied
        iteration in the specific condition.

        Args:
            pol_type: 'global' or 'local'
            condition: Condition number to test the agent
            iteration: Iteration to test the agent

        Returns:
            bool: True for success, False otherwise.

        """
        noise = np.zeros((self.task_params['T'], self.agent.act_dim))

        if iteration == -1:
            for rr in range(600):
                temp_path = self.hyperparams['log_dir'] + ('/itr_%02d' % rr)
                if os.path.exists(temp_path):
                    iteration += 1

        if iteration == -1:
            print("There is not itr_XX data in '%s'"
                  % self.hyperparams['log_dir'])
            return False

        dir_path = 'itr_%02d/' % iteration

        if pol_type == 'global':
            traj_opt_file = dir_path + 'policy_opt_itr_%02d.pkl' % iteration

            change_print_color.change('BLUE')
            print("\nLoading policy '%s'..." % traj_opt_file)

            prev_policy_opt = self.learn_algo.data_logger.unpickle(traj_opt_file)
            if prev_policy_opt is None:
                print("Error: cannot find '%s.'" % traj_opt_file)
                os._exit(1)
            else:
                self.agent.policy_opt.__dict__.update(prev_policy_opt.__dict__)

            self.agent.policy = self.agent.policy_opt.policy

            policy = None
        else:
            # itr_data_file = dir_path + 'iteration_data_itr_%02d.pkl' % iteration
            itr_data_file = dir_path + 'traj_distr_params_itr_%02d.pkl' % iteration

            change_print_color.change('BLUE')
            print("\nLoading iteration data '%s'..." % itr_data_file)

            itr_data = self.learn_algo.data_logger.unpickle(itr_data_file)
            # policy = itr_data[condition].traj_distr
            policy = LinearGaussianPolicy(itr_data[condition]['K'],
                                          itr_data[condition]['k'],
                                          itr_data[condition]['pol_covar'],
                                          itr_data[condition]['chol_pol_covar'],
                                          itr_data[condition]['inv_pol_covar'])

        stop = False
        while stop is False:
            self.env.reset(condition=condition)
            input('Press a key to start sampling...')
            sample = self.agent.sample(self.env, condition, self.task_params['T'],
                                       self.task_params['Ts'], noise, policy=policy,
                                       save=False)
            answer = input('Execute again. Write (n/N) to stop:')
            if answer.lower() in ['n']:
                stop = True

        return True

    def eval_dualism(self, step_by_step=False, only_global=False,
                     max_conds=None, max_iters=None, sample=-1):
        """Generate useful data from a log file.
        """
        log_path = 'state_data_' + self.hyperparams['log_dir']

        if max_iters is None:
            max_iters = self.task_params['iterations']

        if max_conds is None:
            # ASSUMING TEST_COND IS IN ORDER AND WE WANT ALL THE PREVIOUS
            max_conds = self.task_params['test_cond'][-1]

        all_noise = np.zeros((self.task_params['T'], self.agent.act_dim))
        all_noise = self.learn_algo.noise_data

        if step_by_step:
            stop_eval = False
            while stop_eval is False:

                iteration = int(input('Choose an iteration number:'))
                condition = int(input('Choose a condition number:'))

                if iteration == -1:
                    for rr in range(600):
                        temp_path = self.hyperparams['log_dir'] + ('/itr_%02d' % rr)
                        if os.path.exists(temp_path):
                            iteration += 1

                if iteration == -1:
                    print("There is not itr_XX data in '%s'"
                          % self.hyperparams['log_dir'])
                    return False

                if not only_global:
                    # PREV POLICY
                    itr_data_file = 'itr_%02d/traj_distr_params_itr_%02d.pkl' \
                                    % (iteration-1, iteration-1)
                    change_print_color.change('BLUE')
                    print("\nLoading traj_distr data '%s'..." % itr_data_file)
                    itr_data = self.learn_algo.data_logger.unpickle(itr_data_file)
                    # policy = itr_data[condition].traj_distr
                    prev_policy = LinearGaussianPolicy(itr_data[condition]['K'],
                                                       itr_data[condition]['k'],
                                                       itr_data[condition]['pol_covar'],
                                                       itr_data[condition]['chol_pol_covar'],
                                                       itr_data[condition]['inv_pol_covar'])

                    dir_path = 'itr_%02d/' % iteration

                    # NEW (UPDATED) POLICY
                    itr_data_file = dir_path + 'traj_distr_params_itr_%02d.pkl' % iteration
                    change_print_color.change('BLUE')
                    print("\nLoading traj_distr data '%s'..." % itr_data_file)
                    itr_data = self.learn_algo.data_logger.unpickle(itr_data_file)
                    # policy = itr_data[condition].traj_distr
                    new_policy = LinearGaussianPolicy(itr_data[condition]['K'],
                                                      itr_data[condition]['k'],
                                                      itr_data[condition]['pol_covar'],
                                                      itr_data[condition]['chol_pol_covar'],
                                                      itr_data[condition]['inv_pol_covar'])

                    # GOOD TRAJ
                    itr_data_file = dir_path + 'good_duality_info_itr_%02d.pkl' % iteration
                    change_print_color.change('BLUE')
                    print("\nLoading good duality data '%s'..." % itr_data_file)
                    itr_data = self.learn_algo.data_logger.unpickle(itr_data_file)
                    good_traj_distr = itr_data[condition].traj_dist

                    # BAD TRAJ
                    itr_data_file = dir_path + 'bad_duality_info_itr_%02d.pkl' % iteration
                    change_print_color.change('BLUE')
                    print("\nLoading bad duality data '%s'..." % itr_data_file)
                    itr_data = self.learn_algo.data_logger.unpickle(itr_data_file)
                    bad_traj_distr = itr_data[condition].traj_dist

                    # FROM DATA
                    itr_data_file = dir_path + 'traj_sample_itr_%02d.pkl' % iteration
                    print("\nLoading traj samples data '%s'..." % itr_data_file)
                    itr_data = self.learn_algo.data_logger.unpickle(itr_data_file)
                    n_sample = 0
                    sample_actions = itr_data[condition].get_actions()
                    sample_policy = DataSetPolicy(self.action_dim,
                                                  dataset=sample_actions[n_sample])
                    print('sample_dataset', sample_policy.dataset[-1, :])

                # GLOBAL POLICY
                dir_path = 'itr_%02d/' % iteration
                traj_opt_file = dir_path + 'policy_opt_itr_%02d.pkl' % iteration
                change_print_color.change('BLUE')
                print("\nLoading policy '%s'..." % traj_opt_file)
                prev_policy_opt = self.learn_algo.data_logger.unpickle(traj_opt_file)
                if prev_policy_opt is None:
                    print("Error: cannot find '%s.'" % traj_opt_file)
                    os._exit(1)
                else:
                    self.agent.policy_opt.__dict__.update(prev_policy_opt.__dict__)
                self.agent.policy = self.agent.policy_opt.policy

                # SAMPLING WITHOUT NOISE
                noise = np.zeros((self.task_params['T'], self.agent.act_dim))

                stop_selection = False
                while stop_selection is False:
                    self.env.reset(condition=condition)

                    option = input('Choose an option to sample from a key to '
                                   'start sampling from condition %02d and '
                                   'noise %02d (b,g,s,p,n,d,q):'
                                   % (condition, sample))
                    print(option)
                    if option.lower() == 'b':
                        print('Using bad policy...')
                        policy = bad_traj_distr
                    elif option.lower() == 'g':
                        print('Using good policy...')
                        policy = good_traj_distr
                    elif option.lower().startswith('s'):
                        # n_samp_opt = int(input('Choose a sample number:'))
                        n_samp_opt = int(option[-1])
                        sample_policy.dataset = sample_actions[n_samp_opt]
                        print('Using sample policy %02d ...' % n_samp_opt)
                        policy = sample_policy
                    elif option.lower() == 'p':
                        print('Using prev policy...')
                        policy = prev_policy
                    elif option.lower() == 'n':
                        print('Using new policy...')
                        policy = new_policy
                    elif option.lower() == 'd':
                        print('Using global policy (DNN) ...')
                        policy = None
                    elif option.lower() == 'q':
                        print('Loading new data...')
                        break
                    else:
                        print("Wrong option: '%s'" % option)
                        continue

                    stop_sample = False
                    while stop_sample is False:
                        input('Press a key to start sampling')
                        sample_data = self.agent.sample(self.env, condition,
                                                        self.task_params['T'],
                                                        self.task_params['Ts'],
                                                        noise,
                                                        policy=policy, save=False,
                                                        real_time=False)
                        # real_time=True)
                        input('Press a key to reset sampling')
                        # answer = input('Execute again. Write (n/N) to stop:')
                        # if answer.lower() in ['n']:
                        #     stop_sample = True
                        stop_sample = True

        else:
            for ii in range(1, max_iters):
                for cc in range(0, max_conds):
                    iteration = ii
                    condition = cc

                    # PREV POLICY
                    itr_data_file = 'itr_%02d/traj_distr_params_itr_%02d.pkl' \
                                    % (iteration-1, iteration-1)
                    change_print_color.change('BLUE')
                    print("\nLoading traj_distr data '%s'..." % itr_data_file)
                    itr_data = self.learn_algo.data_logger.unpickle(itr_data_file)
                    # policy = itr_data[condition].traj_distr
                    prev_policy = LinearGaussianPolicy(itr_data[condition]['K'],
                                                       itr_data[condition]['k'],
                                                       itr_data[condition]['pol_covar'],
                                                       itr_data[condition]['chol_pol_covar'],
                                                       itr_data[condition]['inv_pol_covar'])

                    dir_path = 'itr_%02d/' % iteration

                    # NEW (UPDATED) POLICY
                    itr_data_file = dir_path + 'traj_distr_params_itr_%02d.pkl' % iteration
                    change_print_color.change('BLUE')
                    print("\nLoading traj_distr data '%s'..." % itr_data_file)
                    itr_data = self.learn_algo.data_logger.unpickle(itr_data_file)
                    # policy = itr_data[condition].traj_distr
                    new_policy = LinearGaussianPolicy(itr_data[condition]['K'],
                                                      itr_data[condition]['k'],
                                                      itr_data[condition]['pol_covar'],
                                                      itr_data[condition]['chol_pol_covar'],
                                                      itr_data[condition]['inv_pol_covar'])

                    # GLOBAL POLICY
                    traj_opt_file = dir_path + 'policy_opt_itr_%02d.pkl' % iteration
                    change_print_color.change('BLUE')
                    print("\nLoading policy '%s'..." % traj_opt_file)
                    prev_policy_opt = self.learn_algo.data_logger.unpickle(traj_opt_file)
                    if prev_policy_opt is None:
                        print("Error: cannot find '%s.'" % traj_opt_file)
                        os._exit(1)
                    else:
                        self.agent.policy_opt.__dict__.update(prev_policy_opt.__dict__)
                    self.agent.policy = self.agent.policy_opt.policy

                    # GOOD TRAJ
                    itr_data_file = dir_path + 'good_duality_info_itr_%02d.pkl' % iteration
                    change_print_color.change('BLUE')
                    print("\nLoading good duality data '%s'..." % itr_data_file)
                    itr_data = self.learn_algo.data_logger.unpickle(itr_data_file)
                    good_traj_distr = itr_data[condition].traj_dist

                    # BAD TRAJ
                    itr_data_file = dir_path + 'bad_duality_info_itr_%02d.pkl' % iteration
                    change_print_color.change('BLUE')
                    print("\nLoading bad duality data '%s'..." % itr_data_file)
                    itr_data = self.learn_algo.data_logger.unpickle(itr_data_file)
                    bad_traj_distr = itr_data[condition].traj_dist

                    # FROM DATA
                    itr_data_file = dir_path + 'traj_sample_itr_%02d.pkl' % iteration
                    print("\nLoading traj samples data '%s'..." % itr_data_file)
                    itr_data = self.learn_algo.data_logger.unpickle(itr_data_file)
                    n_sample = 0
                    sample_actions = itr_data[condition].get_actions()
                    sample_policy = DataSetPolicy(self.action_dim,
                                                  dataset=sample_actions[n_sample])
                    print('sample_dataset', sample_policy.dataset[-1, :])

                    # TRAJ_INFO TO GET THE MODEL
                    noise = all_noise[iteration, condition, sample, :, :]
                    noise = np.zeros_like(noise)

                    self.env.reset(condition=condition)
                    policy = new_policy
                    new_sample = self.agent.sample(self.env, condition,
                                                   self.task_params['T'],
                                                   self.task_params['Ts'],
                                                   noise,
                                                   policy=policy, save=False,
                                                   real_time=False)

                    self.env.reset(condition=condition)
                    policy = prev_policy
                    prev_sample = self.agent.sample(self.env, condition,
                                                    self.task_params['T'],
                                                    self.task_params['Ts'],
                                                    noise,
                                                    policy=policy, save=False,
                                                    real_time=False)

                    self.env.reset(condition=condition)
                    policy = bad_traj_distr
                    bad_sample = self.agent.sample(self.env, condition,
                                                   self.task_params['T'],
                                                   self.task_params['Ts'],
                                                   noise,
                                                   policy=policy, save=False,
                                                   real_time=False)

                    self.env.reset(condition=condition)
                    policy = good_traj_distr
                    good_sample = self.agent.sample(self.env, condition,
                                                    self.task_params['T'],
                                                    self.task_params['Ts'],
                                                    noise,
                                                    policy=policy, save=False,
                                                    real_time=False)

                    self.env.reset(condition=condition)
                    policy = None
                    global_sample = self.agent.sample(self.env, condition,
                                                      self.task_params['T'],
                                                      self.task_params['Ts'],
                                                      noise,
                                                      policy=policy, save=False,
                                                      real_time=False)

                    all_x = dict()
                    all_x['bad'] = bad_sample.get_states()
                    all_x['good'] = good_sample.get_states()
                    all_x['prev'] = prev_sample.get_states()
                    all_x['new'] = new_sample.get_states()
                    all_x['global'] = global_sample.get_states()

                    self.learn_algo.data_logger.pickle(
                        ('all_x_itr_%02d_cond_%02d.pkl' % (iteration, condition)),
                        all_x,
                        dir_path=log_path
                    )

        return True
