import numpy as np
import rospy
import copy
from robolearn.agents.agent import Agent
from robolearn.policies.policy import Policy
from robolearn.envs.environment import Environment
from robolearn.utils.sample.sample import Sample
from robolearn.utils.sample.sample_list import SampleList
from robolearn.utils.plot_utils import *
from robolearn.utils.data_logger import DataLogger
from robolearn.agents.agent_utils import generate_noise

default_hyperparams = {
    'smooth_noise': True,
    'smooth_noise_var': 0.00,
    'smooth_noise_renormalize': True,
    'T': None,
    'dt': None
}


class JointSpaceControlSampler(object):
    def __init__(self, policy, env, **kwargs):

        if not issubclass(type(env), Environment):
            raise TypeError("Wrong Environment type for environment argument")
        self.env = env

        # Get hyperparams
        config = copy.deepcopy(default_hyperparams)
        config.update(**kwargs)
        assert isinstance(config, dict)
        self._hyperparams = config

        # Get some values from the environment.
        self.dU = self._hyperparams['dU'] = env.get_action_dim()
        self.dX = self._hyperparams['dX'] = env.get_state_dim()
        self.dO = self._hyperparams['dO'] = env.get_obs_dim()

        # Important hyperparams
        if self._hyperparams['T'] is None:
            raise ValueError("T has not been specified")
        if self._hyperparams['dt'] is None:
            raise ValueError("dt has not been specified")
        self.T = self._hyperparams['T']
        self.dt = self._hyperparams['dt']

        self.policy = policy

        self.n_samples = 1

        self.q = np.zeros(self._hyperparams['q_size'])
        self.qdot = np.zeros(self._hyperparams['qdot_size'])
        self.q_des = np.zeros(self._hyperparams['q_size'])
        self.qdot_des = np.zeros(self._hyperparams['qdot_size'])
        self.qddot_des = np.zeros(self._hyperparams['qdot_size'])

        # Indexes
        self.joints_idx = self._hyperparams['joints_idx']
        self.state_pos_idx = self._hyperparams['state_pos_idx']
        self.state_vel_idx = self._hyperparams['state_vel_idx']

        if self._hyperparams['act_idx'] is None:
            self.act_idx = range(self._hyperparams['qdot_size'])
        else:
            self.act_idx = self._hyperparams['act_idx']

        self.joints_trajectories = self._hyperparams['joints_trajectories']

        self.data_logger = DataLogger()

    def take_samples(self, n_samples, cond=0, noisy=False, save=True):
        self.n_samples = n_samples

        sample_list = SampleList()

        for ii in range(n_samples):
            sample_list.add_sample(self._take_sample(ii, cond, noisy=noisy, save=save))

    def _take_sample(self, i, cond, verbose=True, save=True, noisy=False):
        """
        Collect a sample from the environment.
        :param i: Sample number.
        :param cond: 
        :param verbose: 
        :param save: 
        :param noisy: 
        :return: Sample object
        """

        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        # Create a sample class
        sample = Sample(self.env, self.T)
        history = [None] * self.T
        obs_hist = [None] * self.T

        print("Resetting environment...")
        self.env.reset(time=2, cond=cond)

        ros_rate = rospy.Rate(int(1/self.dt))  # hz
        # Collect history
        for t in range(self.T):
            if verbose:
                print("Sample cond:%d | i:%d/%d | t:%d/%d" % (cond, i+1, self.n_samples, t+1, self.T))

            self.q_des[:] = self.joints_trajectories[cond][0][t, :]
            self.qdot_des[:] = self.joints_trajectories[cond][1][t, :]
            self.qddot_des[:] = self.joints_trajectories[cond][2][t, :]

            obs = self.env.get_observation()
            state = self.env.get_state()
            self.q[self.joints_idx] = state[self.state_pos_idx]
            self.qdot[self.joints_idx] = state[self.state_vel_idx]

            action = self.policy.eval(self.q_des, self.qdot_des, self.qddot_des, self.q, self.qdot)[self.act_idx] + noise[t, :]
            self.env.send_action(action)
            obs_hist[t] = (obs, action)
            history[t] = (state, action)

            ros_rate.sleep()

        all_actions = np.array([hist[1] for hist in history])
        all_states = np.array([hist[0] for hist in history])
        all_obs = np.array([hist[0] for hist in obs_hist])
        sample.set_acts(all_actions)  # Set all actions at the same time
        sample.set_obs(all_obs)  # Set all obs at the same time
        sample.set_states(all_states)  # Set all states at the same time
        sample.set_noise(noise)

        if save:
            name_file = self.data_logger.pickle('/cosa', sample)
            print("Sample saved in %s" % name_file)
        #if save:  # Save sample in agent
        #    self.agent._samples[cond].append(sample)  # TODO: Do this with a method

        #print("Plotting sample %d" % (i+1))
        #plot_sample(sample, data_to_plot='actions', block=True)
        ##plot_sample(sample, data_to_plot='states', block=True)
        ##plot_joint_info(sample.get_acts(), data_to_plot='actions', block=False)

        return sample

