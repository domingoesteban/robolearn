from robolearn.utils.sample.sample_list import SampleList
from robolearn.utils.print_utils import ProgressBar
from robolearn.utils.sample.sample import Sample
from robolearn.utils.data_logger import DataLogger
# from robolearn.envs.gym_environment import GymEnv
from robolearn.envs.pybullet.bullet_env import BulletEnv
import rospy
import time
import copy
import numpy as np


class Agent(object):
    """
    Agent base class
    """

    def __init__(self, act_dim, obs_dim, state_dim=None):
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        if state_dim is None:
            state_dim = obs_dim
        self.state_dim = state_dim

        # TODO: We assume that an agent should remember his samples
        # (experience??). Check if we include it in all agents
        # List of lists, one list for each condition
        self._samples = []

        # Good and Bad experiences
        self._good_experience = []
        self._bad_experience = []

    def act(self, **kwargs):
        """
        Return the action given the current policy.
        :param obs: Environment observations
        :return:
        """
        return self.policy.eval(**kwargs)

    def policy(self, **kwargs):
        """
        Function that maps state to action
        :param state:
        :return:
        """
        raise NotImplementedError

    def sample(self, env, cond, T, dt, noise, policy=None, save=True):

        if policy is None:
            policy = self.policy

        # if issubclass(type(self.env), GymEnv):
        if not issubclass(type(env), BulletEnv):  # Asumming is ROS then
            ros_rate = rospy.Rate(int(1/dt))  # hz

        sample = Sample(env, T)
        history = [None] * T
        obs_hist = [None] * T

        sampling_bar = ProgressBar(T, bar_title='Sampling',
                                   total_lines=20)

        for t in range(T):
            sampling_bar.update(t)
            state = env.get_state()
            obs = env.get_observation()

            # Checking NAN
            nan_number = np.isnan(obs)
            if np.any(nan_number):
                print("\e[31mERROR OBSERVATION: NAN!!!!! | ")
            # print(obs)
            # TODO: Avoid TF policy writes in obs
            action = policy.eval(state.copy(), obs.copy(), t,
                                 noise[t].copy())

            # action = np.zeros_like(action)
            # Checking NAN
            nan_number = np.isnan(action)
            if np.any(nan_number):
                print("\e[31mERROR ACTION: NAN!!!!!")
            action[nan_number] = 0
            # self.env.send_action(action)
            env.step(action)

            obs_hist[t] = (obs, action)
            history[t] = (state, action)

            ## if issubclass(type(self.env), GymEnv):
            #if issubclass(type(env), BulletEnv):
            #    time.sleep(dt)
            #else:
            #    ros_rate.sleep()

        sampling_bar.end()

        # Stop environment
        env.stop()

        # print("Generating sample data...")
        all_actions = np.array([hist[1] for hist in history])
        all_states = np.array([hist[0] for hist in history])
        all_obs = np.array([hist[0] for hist in obs_hist])
        sample.set_acts(all_actions)
        sample.set_obs(all_obs)
        sample.set_states(all_states)
        sample.set_noise(noise)

        if save:  # Save sample in agent sample list
            sample_id = self.add_sample(sample, cond)
            print("The sample was added to Agent's sample "
                  "list. Now there are %d sample(s) for "
                  "condition '%d'." % (sample_id+1, cond))

        return sample

    def get_samples(self, condition, start=0, end=None):
        """
        Return a SampleList object with the requested samples based on the start
        and end indices.
        :param condition:
        :param start: Starting index of samples to return.
        :param end: End index of samples to return.
        :return:
        """
        return SampleList(self._samples[condition][start:]) if end is None \
            else SampleList(self._samples[condition][start:end])

    def clear_samples(self, condition=None):
        """
        Reset the samples for a given condition, defaulting to all conditions.
        :param condition: Condition for which to reset samples. If not specified
                          clean all the samples!
        :return:
        """
        if condition is None:
            #self._samples = [[] for _ in range(self.conditions)]
            self._samples = []
        else:
            self._samples[condition] = []

    def add_sample(self, sample, condition):
        """
        Add a sample to the agent samples list.
        :param sample: Sample to be added
        :param condition: Condition ID
        :return: Sample id
        """
        # If it does not exist exist samples from that condition, create one
        if condition > len(self._samples)-1:
            self._samples.append(list())

        self._samples[condition].append(sample)

        return len(self._samples[condition]) - 1


