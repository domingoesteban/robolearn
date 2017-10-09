from robolearn.algos.algorithm import Algorithm

from robolearn.agents.agent import Agent
from robolearn.envs.environment import Environment


class RLAlgorithm(Algorithm):
    def __init__(self, agent, env, default_hyperparams, hyperparams):
        """
        Class constructor.
        :param agent: Agent that interacts with the environment.
        :type agent: robolearn.agents.agent.Agent
        :param env: Environment where the agent interacts.
        :type env: robolearn.envs.environment.Environment
        :param default_hyperparams: Default algorithm hyperparameters.
        :param hyperparams: Particular object hyperparameters.
        """
        super(RLAlgorithm, self).__init__(default_hyperparams, hyperparams)

        if not issubclass(type(agent), Agent):
            raise TypeError("Wrong Agent type for agent argument.")
        self.agent = agent

        if not issubclass(type(env), Environment):
            raise TypeError("Wrong Environment type for environment argument")
        self.env = env

    def run(self, *args, **kwargs):
        """
        Run the RL algorithm.
        """
        raise NotImplementedError

    def _initialize(self, itr_load):
        """
        Initialize algorithm from the specified iteration.
        :param itr_load: If specified, loads algorithm state from that iteration, and resumes training at the next
                         iteration.
        :return:
        """
        raise NotImplementedError

    def _take_iteration(self, itr, sample_lists):
        """
        Iterate once the RL algorithm.
        """
        raise NotImplementedError

    def _take_sample(self, *args, **kwargs):
        """
        Collect a sample from the environment (i.e. exploration).
        """
        raise NotImplementedError

    def _end(self):
        """
        Finish the RL algorithm and exit.
        """
        print("")
        print("RL algorithm has finished!")
        self.env.stop()
