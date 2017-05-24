from robolearn.algos.algorithm import Algorithm

from robolearn.agents.agent import Agent
from robolearn.envs.environment import Environment


class RLAlgorithm(Algorithm):
    def __init__(self, agent, env, default_hyperparams, hyperparams):
        super(RLAlgorithm, self).__init__(default_hyperparams, hyperparams)

        if not issubclass(type(agent), Agent):
            raise TypeError("Wrong Agent type for agent argument.")
        self.agent = agent

        if not issubclass(type(env), Environment):
            raise TypeError("Wrong Environment type for environment argument")
        self.env = env

    def run(self, **kwargs):
        """
        Run RL Algorithm.
        """
        raise NotImplementedError

    def _take_sample(self, *args, **kwargs):
        """
        Collect a sample from the environment. (i.e. exploration)
        """
        raise NotImplementedError

    def _take_iteration(self, itr, sample_lists):
        """
        One iteration of the RL algorithm.
        """
        raise NotImplementedError

    def _initialize(self, itr_load):
        """
        Initialize algorithm from the specified iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        """
        raise NotImplementedError
