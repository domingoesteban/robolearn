from __future__ import print_function

import os
import random
import signal
import yaml

import numpy as np
from robolearn.envs.pusher3dof import Pusher3DofBulletEnv
from robolearn.utils.sample.sampler import Sampler

from robolearn.agents import NoPolAgent
from robolearn.algos.trajopt.dual_trajopt import DualTrajOpt
# Costs
from robolearn.costs.cost_action import CostAction
# from robolearn.costs.cost_fk import CostFK
from robolearn.costs.cost_state import CostState
from robolearn.costs.cost_safe_distance import CostSafeDistance
from robolearn.costs.cost_state_difference import CostStateDifference
from robolearn.costs.cost_safe_state_difference import CostSafeStateDifference
from robolearn.costs.cost_sum import CostSum
from robolearn.costs.cost_utils import RAMP_FINAL_ONLY, RAMP_CONSTANT
from robolearn.costs.cost_utils import evall1l2term
# from robolearn.envs import BigmanEnv
from robolearn.policies.lin_gauss_init import init_pd, init_dual_demos
from robolearn.policies.policy_opt.policy_opt_tf import PolicyOptTf
from robolearn.policies.policy_opt.tf_models import tf_network
from robolearn.policies.policy_prior import ConstantPolicyPrior  # For MDGPS
from robolearn.utils.dynamics.dynamics_lr_prior import DynamicsLRPrior
from robolearn.utils.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from robolearn.utils.print_utils import change_print_color
from robolearn.utils.transformations_utils import create_quat_pose
from robolearn.utils.traj_opt.dualist_traj_opt import DualistTrajOpt

np.set_printoptions(precision=4, suppress=True, linewidth=1000)


def kill_everything(_signal=None, _frame=None):
    print("\n\033[1;31mThe script has been killed by the user!!")
    os._exit(1)


signal.signal(signal.SIGINT, kill_everything)


class Scenario(object):
    def __init__(self, hyperparams):

        self.hyperparams = hyperparams

        # Task Parameters
        yaml_path = os.path.dirname(__file__) + '/task_parameters.yaml'
        assert(os.path.exists(yaml_path))
        with open(yaml_path, 'r') as f:
            self.task_params = yaml.load(f)

        Tend = self.task_params['Tend']
        Ts = self.task_params['Ts']
        self.task_params['T'] = int(Tend/Ts)

        if self.hyperparams['render']:
            self.task_params['render'] = self.hyperparams['render']


        # Numpy max
        os.environ['OMP_NUM_THREADS'] = str(self.task_params['np_threads'])

        # Environment
        self.env = self.create_environment()

        self.action_dim = self.env.get_action_dim()
        self.state_dim = self.env.get_state_dim()
        self.obs_dim = self.env.get_obs_dim()

        # Agent
        self.agent = self.create_agent()

        # Costs
        self.cost = self.create_cost()

        # Initial Conditions
        self.init_cond = self.create_init_conditions()

        # Learning algo
        self.learn_algo = self.create_learning_algo()

    def create_environment(self):
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
        env = Pusher3DofBulletEnv(render=render, obs_with_img=env_with_img,
                                  obs_mjc_gym=obs_like_mjc, ntargets=ntargets,
                                  rdn_tgt_pos=rdn_tgt_pos, tgt_types=tgt_types)

        env.set_tgt_cost_weights(tgt_weights)
        env.set_tgt_pos(tgt_positions)

        print("Environment:%s OK!." % type(env).__name__)

        return env

    def create_agent(self):
        change_print_color.change('CYAN')
        print("\nCreating Agent...")

        policy_params = {}

        policy_opt = {
            'type': PolicyOptTf,
            'hyperparams': policy_params
        }

        agent = NoPolAgent(act_dim=self.action_dim, obs_dim=self.obs_dim,
                           state_dim=self.state_dim,
                           agent_name='agent'+str('%02d' %
                                                  self.hyperparams['run_num']))
        print("Agent:%s OK\n" % type(agent).__name__)

        return agent

    def create_cost(self):
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


        # State costs
        target_distance_object = np.zeros(2)
        # input(self.env.get_state_info())
        # input(self.env.get_state_info(name='tgt0')['idx'])
        state_cost_distance = {
            'type': CostState,
            'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
            'evalnorm': evall1l2term,  # TODO: ALWAYS USE evall1l2term
            'l1': 1.0,  # Weight for l1 norm
            'l2': 1.0,  # Weight for l2 norm
            'alpha': 1e-5,  # Constant added in square root in l1 norm
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'tgt0': {
                    'wp': np.array([1.0, 1.0]),  # State weights - must be set.
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
            'l1': 1.0,  # Weight for l1 norm
            'l2': 1.0,  # Weight for l2 norm
            'alpha': 1e-5,  # Constant added in square root in l1 norm
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'tgt0': {
                    'wp': np.array([1.0, 1.0]),  # State weights - must be set.
                    'target_state': target_distance_object,  # Target state - must be set.
                    'average': None,
                    'data_idx': self.env.get_state_info(name='tgt0')['idx']
                },
            },
        }

        cost_safe_distance = {
            'type': CostSafeDistance,
            'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'tgt1': {
                    'wp': np.array([1.0, 1.0]),  # State weights - must be set.
                    'safe_distance': np.array([0.15, 0.15]),
                    'outside_cost': np.array([0.0, 0.0]),
                    'inside_cost': np.array([1.0, 1.0]),
                    'data_idx': self.env.get_state_info(name='tgt1')['idx']
                },
            },
        }

        cost_state_difference = {
            'type': CostStateDifference,
            'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
            'evalnorm': evall1l2term,  # TODO: ALWAYS USE evall1l2term
            'l1': 1.e-0,  # Weight for l1 norm
            'l2': 5.e-2,  # Weight for l2 norm
            'alpha': 1e-10,  # Constant added in square root in l1 norm
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'ee': {
                    'wp': np.array([1.0, 1.0, 0.6]),  # State weights - must be set.
                    'target_state': 'tgt0',  # Target state - must be set.
                    'average': None,
                    'tgt_idx': self.env.get_state_info(name='tgt0')['idx'],
                    'data_idx': self.env.get_state_info(name='ee')['idx'],
                    'idx_to_use': [0, 1, 2],  # All: X, Y, theta
                },
            },
        }

        cost_final_state_difference = {
            'type': CostStateDifference,
            'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
            'evalnorm': evall1l2term,  # TODO: ALWAYS USE evall1l2term
            'l1': 1.e-0,  # Weight for l1 norm
            'l2': 5.e-2,  # Weight for l2 norm
            'alpha': 1e-10,  # Constant added in square root in l1 norm
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'ee': {
                    'wp': np.array([1.0, 1.0, 0.6]),  # State weights - must be set.
                    'target_state': 'tgt0',  # Target state - must be set.
                    'average': None,
                    'tgt_idx': self.env.get_state_info(name='tgt0')['idx'],
                    'data_idx': self.env.get_state_info(name='ee')['idx'],
                    'idx_to_use': [0, 1, 2],  # All: X, Y, theta
                },
            },
        }

        safe_radius = 0.15
        cost_safe_state_difference = {
            'type': CostSafeStateDifference,
            'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'ee': {
                    'wp': np.array([1.0, 1.0]),  # State weights - must be set.
                    'safe_distance': np.sqrt([safe_radius**2/2, safe_radius**2/2]),
                    'outside_cost': np.array([0.0, 0.0]),
                    'inside_cost': np.array([1.0, 1.0]),
                    'target_state': 'tgt1',  # Target state - must be set.
                    'tgt_idx': self.env.get_state_info(name='tgt1')['idx'][:2],
                    'data_idx': self.env.get_state_info(name='ee')['idx'][:2],
                    'idx_to_use': [0, 1],  # Only X and Y
                },
            },
        }

        cost_final_safe_state_difference = {
            'type': CostSafeStateDifference,
            'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time.
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'ee': {
                    'wp': np.array([1.0, 1.0]),  # State weights - must be set.
                    'safe_distance': np.sqrt([safe_radius**2/2, safe_radius**2/2]),
                    'outside_cost': np.array([0.0, 0.0]),
                    'inside_cost': np.array([1.0, 1.0]),
                    'target_state': 'tgt1',  # Target state - must be set.
                    'tgt_idx': self.env.get_state_info(name='tgt1')['idx'][:2],
                    'data_idx': self.env.get_state_info(name='ee')['idx'][:2],
                    'idx_to_use': [0, 1],  # Only X and Y
                },
            },
        }


        # Sum costs
        # costs_and_weights = [(act_cost, 1.0e-1),
        costs_and_weights = [(act_cost, 1.0e-5),
                             # # (fk_cost, 1.0e-0),
                             # (fk_l1_cost, 1.5e-1),
                             # (fk_l2_cost, 1.0e-0),
                             # # (fk_final_cost, 1.0e-0),
                             # (fk_l1_final_cost, 1.5e-1),
                             # (fk_l2_final_cost, 1.0e-0),
                             (cost_state_difference, 5.0e-0),
                             (cost_final_state_difference, 1.0e+3),
                             (cost_safe_state_difference, 5.0e+1),
                             (cost_final_safe_state_difference, 5.0e+4),
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
        change_print_color.change('MAGENTA')
        print("\nCreating Initial Conditions...")
        initial_cond = self.task_params['init_cond']

        ddof = 3  # Data dof (file): x, y, theta
        pdof = 3  # Pose dof (env): x, y, theta
        ntgt = self.task_params['ntargets']

        for cc, cond in enumerate(initial_cond):
            condition = np.zeros(self.env.get_obs_dim())
            condition[:self.env.get_action_dim()] = np.deg2rad(cond[:3])
            cond_idx = 2*self.env.get_action_dim() + pdof  # EE pose will be obtained from sim
            data_idx = self.env.get_action_dim()
            for tt in range(self.task_params['ntargets']):
                tgt_data = cond[data_idx:data_idx+ddof]
                # tgt_pose = create_quat_pose(pos_x=tgt_data[0],
                #                             pos_y=tgt_data[1],
                #                             pos_z=z_fix,
                #                             rot_yaw=np.deg2rad(tgt_data[2]))
                # condition[cond_idx:cond_idx+pdof] = tgt_pose
                tgt_data[2] = np.deg2rad(tgt_data[2])
                condition[cond_idx:cond_idx+pdof] = tgt_data
                cond_idx += pdof
                data_idx += ddof

            self.env.add_init_cond(condition)

        return self.env.get_conditions()

    def create_learning_algo(self):
        change_print_color.change('YELLOW')
        print("\nConfiguring learning algorithm...\n")

        # Learning params
        resume_training_itr = None  # Resume from previous training iteration

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

        init_traj_distr = {'type': init_pd,
                           'init_var': np.array([1.0, 1.0, 1.0])*5.0e-01,
                           'pos_gains': 0.001,  # float or array
                           'vel_gains_mult': 0.01,  # Velocity gains multiplier on pos_gains
                           'init_action_offset': None,
                           'dJoints': self.env.get_total_joints(),  # Total joints in state
                           'state_to_pd': 'joints',  # Joints
                           'dDistance': 6,
                           }

        # Trajectory Optimization Method
        traj_opt_method = {
            'type': DualistTrajOpt,
            'good_const': self.task_params['consider_good'],  # Use good constraints
            'bad_const': self.task_params['consider_bad'],  # Use bad constraints
            'del0': 1e-4,  # Eta updates for non-SPD Q-function (non-SPD correction step).
            'del0_good': 1e-4,  # Omega updates for non-SPD Q-function (non-SPD correction step).
            'del0_bad': 1e-8,  # Nu updates for non-SPD Q-function (non-SPD correction step).
            # 'eta_error_threshold': 1e16, # TODO: REMOVE, it is not used
            'min_eta': 1e-8,  # At min_eta, kl_div > kl_step
            'max_eta': 1e16,  # At max_eta, kl_div < kl_step
            'min_omega': 1e-8,  # At min_omega, kl_div > kl_step
            'max_omega': 1.0e4,  #1e16,  # At max_omega, kl_div < kl_step
            'min_nu': 1e-8,  # At min_nu, kl_div > kl_step
            'max_nu': 1.0e2,  # At max_nu, kl_div < kl_step,
            'step_tol': 0.1,
            'bad_tol': 0.1,
            'good_tol': 0.1,
            'cons_per_step': False,  # Whether or not to enforce separate KL constraints at each time step. #TODO: IF TRUE, MAYBE IT DOES WORK WITH MDGPS because it doesn't consider dual vars
            'use_prev_distr': False,  # Whether or not to measure expected KL under the previous traj distr.
            'update_in_bwd_pass': True,  # Whether or not to update the TVLG controller during the bwd pass.
            'adam_alpha': 0.005,
            'adam_max_iter': 200,
            }

        good_trajs = None
        bad_trajs = None
        dualtrajopt_hyperparams = {
            'inner_iterations': self.task_params['inner_iterations'],  # Times the trajectories are updated
            # G/B samples selection | Fitting
            'good_samples': good_trajs,  # Good samples demos
            'bad_samples': bad_trajs,  # Bad samples demos
            'n_good_buffer': self.task_params['n_good_buffer'],  # Number of good samples per each trajectory
            'n_bad_buffer': self.task_params['n_bad_buffer'],  # Number of bad samples per each trajectory
            'good_traj_selection_type': self.task_params['good_traj_selection_type'],  # 'always', 'only_traj'
            'bad_traj_selection_type': self.task_params['bad_traj_selection_type'],  # 'always', 'only_traj'
            'duality_dynamics_type': 'duality',  # Samples to use to update the dynamics 'duality', 'iteration'
            # Initial dual variables
            'init_eta': 0.1,#4.62,
            'init_nu': 0.1,
            'init_omega': 0.1,
            # KL step (epsilon)
            'step_rule': 'laplace',  # Whether to use 'laplace' or 'mc' cost in step adjustment
            'kl_step': 0.2,  # Kullback-Leibler step (base_step)
            'min_step_mult': 0.01,  # Min possible value of step multiplier (multiplies kl_step)
            'max_step_mult': 10.0,  # Max possible value of step multiplier (multiplies kl_step)
            # KL bad (xi)
            'kl_bad': 2, #4.2  # Xi KL base value | kl_div_b >= kl_bad
            'min_bad_mult': 0.01,  # Min possible value of step multiplier (multiplies base_kl_bad)
            'max_bad_mult': 10.0,  # Max possible value of step multiplier (multiplies base_kl_bad)
            # KL good (chi)
            'kl_good': 0.4,  #2.0,  # Chi KL base value  | kl_div_g <= kl_good
            'min_good_mult': 0.01,  # Min possible value of step multiplier (multiplies base_kl_good)
            'max_good_mult': 10.0,  # Max possible value of step multiplier (multiplies base_kl_good)
            # LinearPolicy 'projection'
            'init_pol_wt': 0.01,  # TODO: remove need for init_pol_wt in MDGPS (It should not work with MDGPS)
            'policy_sample_mode': 'add',  # Mode to update dynamics prior (Not used in ConstantPolicyPrior)
            'policy_prior': {'type': ConstantPolicyPrior,
                             'strength': 1e-4,
                             },
            'min_bad_var': np.array([3.0, 3.0, 3.0])*1.0e-02,
            'min_good_var': np.array([3.0, 3.0, 3.0])*1.0e-02,
            # TEMP Hyperparams
            'min_rel_diff': 0.01,
            'max_rel_diff': 2.0,
            'mult_rel_diff': 1,
            }

        gps_hyperparams = {
            'T': self.task_params['T'],  # Total points
            'dt': self.task_params['Ts'],
            'iterations': self.task_params['iterations'],  # GPS episodes --> K iterations
            # Samples
            'num_samples': self.task_params['num_samples'],  # Samples for exploration trajs --> N samples
            'noisy_samples': True,
            'smooth_noise': True,  # Apply Gaussian filter to noise generated
            'smooth_noise_var': 5.0e+0,  # np.power(2*Ts, 2), # Variance to apply to Gaussian Filter. In Kumar (2016) paper, it is the std dev of 2 Ts
            'smooth_noise_renormalize': True,  # Renormalize smooth noise to have variance=1
            'noise_var_scale': 1.e-1*np.ones(self.action_dim),  # Scale to Gaussian noise: N(0, 1)*sqrt(noise_var_scale), only if smooth_noise_renormalize
            # Cost
            'cost': self.cost,
            # Conditions
            'conditions': len(self.init_cond),  # Total number of initial conditions
            'train_conditions': self.task_params['train_cond'],  # Indexes of conditions used for training
            # TrajDist
            'init_traj_distr': init_traj_distr,
            'fit_dynamics': True,
            'dynamics': learned_dynamics,
            'initial_state_var': 1e-2,  # Max value for x0sigma in trajectories
            # TrajOpt
            'traj_opt': traj_opt_method,
            'max_ent_traj': 0.0,  # Weight of maximum entropy term in trajectory optimization #TODO: CHECK THIS VALUE
            # Others
            'algo_hyperparams': dualtrajopt_hyperparams,
            'data_files_dir': self.hyperparams['log_dir'],
        }

        return DualTrajOpt(self.agent, self.env, **gps_hyperparams)

    def train(self, itr_load=None):
        return self.learn_algo.run(itr_load)

    def test_policy(self, pol_type=None, condition=0, iteration=-1):
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

        itr_data_file = dir_path + 'iteration_data_itr_%02d.pkl' % iteration

        change_print_color.change('BLUE')
        print("\nLoading iteration data '%s'..." % itr_data_file)

        itr_data = self.learn_algo.data_logger.unpickle(itr_data_file)
        policy = itr_data[condition].traj_distr

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
