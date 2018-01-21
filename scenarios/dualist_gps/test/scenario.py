from __future__ import print_function

import os
import random
import signal
import yaml

import numpy as np
from robolearn.envs.pusher3dof import Pusher3DofBulletEnv
from robolearn.utils.sample.sampler import Sampler

from robolearn.agents import GPSAgent
from robolearn.algos.gps.dual_gps import DualGPS
from robolearn.costs.cost_action import CostAction
# from robolearn.costs.cost_fk import CostFK
from robolearn.costs.cost_state import CostState
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
# from robolearn.utils.iit.iit_robots_params import bigman_params
from robolearn.utils.print_utils import change_print_color
# from robolearn.utils.robot_model import RobotModel
# from robolearn.utils.tasks.bigman.reach_drill_utils import Reset_condition_bigman_drill_gazebo
# from robolearn.utils.tasks.bigman.reach_drill_utils import create_bigman_drill_condition
# from robolearn.utils.tasks.bigman.reach_drill_utils import create_drill_relative_pose
# from robolearn.utils.tasks.bigman.reach_drill_utils import create_hand_relative_pose
# from robolearn.utils.tasks.bigman.reach_drill_utils import spawn_drill_gazebo
# from robolearn.utils.tasks.bigman.reach_drill_utils import task_space_torque_control_dual_demos, \
#     load_task_space_torque_control_dual_demos
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
        env = Pusher3DofBulletEnv(render=render, obs_with_img=env_with_img,
                                  obs_mjc_gym=obs_like_mjc, ntargets=ntargets,
                                  rdn_tgt_pos=rdn_tgt_pos)

        env.set_tgt_cost_weights(tgt_weights)
        env.set_tgt_pos(tgt_positions)

        print("Environment:%s OK!." % type(env).__name__)

        return env

    def create_agent(self):
        change_print_color.change('CYAN')
        print("\nCreating Bigman Agent...")

        policy_params = {
            'network_model': tf_network,  # tf_network, multi_modal_network, multi_modal_network_fp
            'network_params': {
                'n_layers': 2,  # Number of Hidden layers
                'dim_hidden': [40, 40],  # List of size per n_layers
                'obs_names': self.env.get_obs_info()['names'],
                'obs_dof': self.env.get_obs_info()['dimensions'],  # DoF for observation data tensor
            },
            # Initialization.
            'init_var': 0.1,  # Initial policy variance.
            'ent_reg': 0.0,  # Entropy regularizer (Used to update policy variance)
            # Solver hyperparameters.
            'iterations': self.task_params['tf_iterations'],  # Number of iterations per inner iteration (Default:5000). Recommended: 1000?
            'batch_size': 15,
            'lr': 0.001,  # Base learning rate (by default it's fixed).
            'lr_policy': 'fixed',  # Learning rate policy.
            'momentum': 0.9,  # Momentum.
            'weight_decay': 0.005,  # Weight decay.
            'solver_type': 'Adam',  # Solver type (e.g. 'SGD', 'Adam', 'RMSPROP', 'MOMENTUM', 'ADAGRAD').
            # set gpu usage.
            'use_gpu': True,  # Whether or not to use the GPU for training.
            'gpu_id': 0,
            'random_seed': 1,
            'fc_only_iterations': 0,  # Iterations of only FC before normal training
            'gpu_mem_percentage': 0.2,
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
            'l2': 0.0,  # Weight for l2 norm
            'alpha': 1e-5,  # Constant added in square root in l1 norm
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'tgt0': {
                    # 'wp': np.ones_like(target_state),  # State weights - must be set.
                    'wp': np.array([1.0, 1.0]),  # State weights - must be set.
                    'target_state': target_distance_object,  # Target state - must be set.
                    'average': None,  # (12, 3),
                    'data_idx': self.env.get_state_info(name='tgt0')['idx']
                },
            },
        }

        state_final_cost_distance = {
            'type': CostState,
            'ramp_option': RAMP_FINAL_ONLY,  # How target cost ramps over time. RAMP_* :CONSTANT, LINEAR, QUADRATIC, FINAL_ONLY
            'evalnorm': evall1l2term,  # TODO: ALWAYS USE evall1l2term
            'l1': 1.0,  # Weight for l1 norm
            'l2': 0.0,  # Weight for l2 norm
            'alpha': 1e-5,  # Constant added in square root in l1 norm
            'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
            'data_types': {
                'tgt0': {
                    # 'wp': np.ones_like(target_state),  # State weights - must be set.
                    'wp': np.array([1.0, 1.0]),  # State weights - must be set.
                    'target_state': target_distance_object,  # Target state - must be set.
                    'average': None,  # (12, 3),
                    'data_idx': self.env.get_state_info(name='tgt0')['idx']
                },
            },
        }


        # Sum costs
        # costs_and_weights = [(act_cost, 1.0e-1),
        costs_and_weights = [(act_cost, 1.0e-3),
                             # # (fk_cost, 1.0e-0),
                             # (fk_l1_cost, 1.5e-1),
                             # (fk_l2_cost, 1.0e-0),
                             # # (fk_final_cost, 1.0e-0),
                             # (fk_l1_final_cost, 1.5e-1),
                             # (fk_l2_final_cost, 1.0e-0),
                             (state_cost_distance, 10.0e-0),
                             (state_final_cost_distance, 1.0e+3),
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

        for cc, cond in enumerate(initial_cond):
            condition = np.zeros(self.env.get_obs_dim())
            condition[:self.env.get_action_dim()] = np.deg2rad(cond[:3])
            condition[2*self.env.get_action_dim():] = cond[3:]
            print('Adding condition %d: %s' % (cc, condition))
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
                           'init_var': np.array([1.0, 1.0, 1.0])*1.0e-01,
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
            'max_omega': 5.0e-1,  #1e16,  # At max_omega, kl_div < kl_step
            'min_nu': 1e-8,  # At min_nu, kl_div > kl_step
            'max_nu': 5.0e-1,  # At max_nu, kl_div < kl_step,
            'step_tol': 0.1,
            'bad_tol': 0.2,
            'good_tol': 0.6, #0.3,
            'cons_per_step': False,  # Whether or not to enforce separate KL constraints at each time step. #TODO: IF TRUE, MAYBE IT DOES WORK WITH MDGPS because it doesn't consider dual vars
            'use_prev_distr': False,  # Whether or not to measure expected KL under the previous traj distr.
            'update_in_bwd_pass': True,  # Whether or not to update the TVLG controller during the bwd pass.
            }

        good_trajs = None
        bad_trajs = None
        gps_algo_hyperparams = {
            'inner_iterations': 1,  # Times the trajectories are updated
            'good_samples': good_trajs,  # Good samples demos
            'bad_samples': bad_trajs,  # Bad samples demos
            'n_bad_samples': 2,  # Number of bad samples per each trajectory
            'n_good_samples': 2,  # Number of good samples per each trajectory
            'base_kl_bad': 2, #4.2  # (xi) to be used with multiplier | kl_div_b >= kl_bad
            'base_kl_good': 1.0, #2.0,  # (chi) to be used with multiplier | kl_div_g <= kl_good
            'bad_traj_selection_type': 'only_traj',  # 'always', 'only_traj'
            'good_traj_selection_type': 'only_traj',  # 'always', 'only_traj'
            'duality_dynamics_type': 'duality',  # Samples to use to update the dynamics 'duality', 'iteration'
            'init_eta': 4.62,
            'init_nu': 0.001,
            'init_omega': 0.001,
            'min_bad_mult': 0.01,  # Min possible value of step multiplier (multiplies base_kl_bad in LQR)
            'max_bad_mult': 1.0,  # Max possible value of step multiplier (multiplies base_kl_bad in LQR)
            'min_good_mult': 0.01,  # Min possible value of step multiplier (multiplies base_kl_good in LQR)
            'max_good_mult': 1.0,  # Max possible value of step multiplier (multiplies base_kl_good in LQR)
            'min_bad_var': np.array([3.0, 3.0, 3.0])*1.0e-02,
            'min_good_var': np.array([3.0, 3.0, 3.0])*1.0e-02,
            'init_pol_wt': 0.01,  # TODO: remove need for init_pol_wt in MDGPS (It should not work with MDGPS)
            'policy_sample_mode': 'add',
            'step_rule': 'laplace',  # Whether to use 'laplace' or 'mc' cost in step adjustment
            'policy_prior': {'type': ConstantPolicyPrior,
                             'strength': 1e-4,
                             },
            }

        gps_hyperparams = {
            'T': self.task_params['T'],  # Total points
            'dt': self.task_params['Ts'],
            'iterations': self.task_params['iterations'],  # GPS episodes, "inner iterations" --> K iterations
            'test_after_iter': self.task_params['test_after_iter'],  # If test the learned policy after an iteration in the RL algorithm
            'test_samples': self.task_params['test_n_samples'],  # Samples from learned policy after an iteration PER CONDITION (only if 'test_after_iter':True)
            # Samples
            'num_samples': self.task_params['num_samples'],  # Samples for exploration trajs --> N samples
            'noisy_samples': True,
            'sample_on_policy': self.task_params['sample_on_policy'],  # Whether generate on-policy samples or off-policy samples
            'smooth_noise': True,  # Apply Gaussian filter to noise generated
            'smooth_noise_var': 5.0e+0,  # np.power(2*Ts, 2), # Variance to apply to Gaussian Filter. In Kumar (2016) paper, it is the std dev of 2 Ts
            'smooth_noise_renormalize': True,  # Renormalize smooth noise to have variance=1
            'noise_var_scale': 5.e-0*np.ones(self.action_dim),  # Scale to Gaussian noise: N(0, 1)*sqrt(noise_var_scale), only if smooth_noise_renormalize
            # Cost
            'cost': self.cost,
            # Conditions
            'conditions': len(self.init_cond),  # Total number of initial conditions
            'train_conditions': self.task_params['train_cond'],  # Indexes of conditions used for training
            'test_conditions': self.task_params['test_cond'],  # Indexes of conditions used for testing
            # KL step (epsilon)
            'kl_step': 0.2,  # Kullback-Leibler step (base_step)
            'min_step_mult': 0.01,  # Min possible value of step multiplier (multiplies kl_step in LQR)
            'max_step_mult': 10.0,  # Max possible value of step multiplier (multiplies kl_step in LQR)
            # TrajDist
            'init_traj_distr': init_traj_distr,
            'fit_dynamics': True,
            'dynamics': learned_dynamics,
            'initial_state_var': 1e-6,  # Max value for x0sigma in trajectories
            # TrajOpt
            'traj_opt': traj_opt_method,
            'max_ent_traj': 0.0,  # Weight of maximum entropy term in trajectory optimization #TODO: CHECK THIS VALUE
            'use_global_policy': self.task_params['use_global_policy'],  # KL prev or policy(MDGPS)
            # Others
            'gps_algo_hyperparams': gps_algo_hyperparams,
            'data_files_dir': self.hyperparams['log_dir'],
        }

        return DualGPS(self.agent, self.env, **gps_hyperparams)

    def train(self, itr_load=None):
        return self.learn_algo.run(itr_load)

    def test_policy(self, type='global', condition=0, iteration=-1):
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

        if type == 'global':
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
            raise NotImplementedError('Not implemented error')

        self.env.reset(condition=condition)
        input('Press a key to start sampling...')
        sample = self.agent.sample(self.env, condition, self.task_params['T'],
                                   self.task_params['Ts'], noise, policy=policy,
                                   save=False)
        return True
