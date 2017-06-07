import numpy as np
import copy
from robolearn.agents.agent import Agent
from robolearn.policies.policy import Policy
from robolearn.envs.environment import Environment
from robolearn.utils.sample import Sample
from robolearn.utils.plot_utils import plot_sample
from robolearn.agents.agent_utils import generate_noise

default_hyperparams = {
    'smooth_noise': True,
    'smooth_noise_var': 0.00,
    'smooth_noise_renormalize': True,
    'T': None,
    'dt': None
}


class Sampler(object):
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

    def take_samples(self, n_samples, cond=0, noisy=False):
        self.n_samples = n_samples

        for ii in range(n_samples):
            self._take_sample(ii, cond, noisy=noisy)

    def _take_sample(self, i, cond, verbose=True, save=True, noisy=False):
        """
        Collect a sample from the environment.
        Args:
            i: Sample number.
        Returns: None
        """

        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        # Create a sample class
        # TODO: In original GPS code this is done with self._init_sample(self, feature_fn=None) in agent
        sample = Sample(self.env, self.T)
        history = [None] * self.T
        obs_hist = [None] * self.T

        print("Resetting environment...")
        self.env.reset(time=1, conf=cond)
        import rospy

        ros_rate = rospy.Rate(int(1/self.dt))  # hz
        # Collect history
        for t in range(self.T):
            if verbose:
                print("Sample cond:%d | i:%d/%d | t:%d/%d" % (cond, i+1, self.n_samples, t+1, self.T))
            obs = self.env.get_observation()
            state = self.env.get_state()
            #action = self.agent.act(obs=obs)
            #action = policy.act(state, obs, t, noise[t, :])
            action = self.policy.eval(state, obs, t, noise[t, :])
            #action = np.zeros_like(action)
            #action[3] = -0.15707963267948966
            self.env.send_action(action)
            obs_hist[t] = (obs, action)
            history[t] = (state, action)
            #print(obs)
            #print("..")
            #print(state)
            #print("--")
            #print("obs_shape:(%s)" % obs.shape)
            #print("state_shape:(%s)" % state.shape)
            #print("obs active names: %s" % bigman_env.get_obs_info()['names'])
            #print("obs active dims: %s" % bigman_env.get_obs_info()['dimensions'])
            #print("state active names: %s" % bigman_env.get_state_info()['names'])
            #print("state active dims: %s" % bigman_env.get_state_info()['dimensions'])
            #print("")

            #sample.set_acts(action, t=i)  # Set action One by one
            #sample.set_obs(obs[:42], obs_name='joint_state', t=i)  # Set action One by one
            #sample.set_states(state[:7], state_name='link_position', t=i)  # Set action One by one

            ros_rate.sleep()

        all_actions = np.array([hist[1] for hist in history])
        all_states = np.array([hist[0] for hist in history])
        all_obs = np.array([hist[0] for hist in obs_hist])
        sample.set_acts(all_actions)  # Set all actions at the same time
        sample.set_obs(all_obs)  # Set all obs at the same time
        sample.set_states(all_states)  # Set all states at the same time

        if save:
            print("This sample is supposed to be saved")
        #if save:  # Save sample in agent
        #    self.agent._samples[cond].append(sample)  # TODO: Do this with a method

        print("Plotting sample %d" % (i+1))
        plot_sample(sample, data_to_plot='actions', block=False)
        #plot_sample(sample, data_to_plot='states', block=True)

        return sample

