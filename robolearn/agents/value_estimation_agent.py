# Code from https://s3-us-west-2.amazonaws.com/cs188websitecontent/projects/release/reinforcement/v1/001/docs/learningAgents.html

from robolearn.agents.agent import Agent


class ValueEstimationAgent(Agent):
    """
      Abstract agent which assigns values to (state,action) Q-Values for an
      environment. As well as a value to a state and a policy given respectively
      by:

      V(s) = max_{a in actions} Q(s,a)
      policy(s) = arg_max_{a in actions} Q(s,a)

      Both ValueIterationAgent and QLearningAgent inherit from this agent.
      While a ValueIterationAgent has a model of the environment via a MDP
      that is used to estimate Q-Values before ever actually acting, the
      QLearningAgent estimates Q-Values while acting in the environment.
    """

    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining = 10):
        """
        :param alpha: learning rate
        :param epsilon: exploration rate
        :param gamma: discount factor
        :param numTraining: number of training episodes, i.e. no learning after
                            these many episodes
        """
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(numTraining)

    def get_qvalue(self, state, action):
        """
        Should return Q(state,action)
        """
        raise NotImplementedError

    def get_value(self, state):
        """
        What is the value of this state under the best action?
        Concretely, this is given by

        V(s) = max_{a in actions} Q(s,a)
        """
        raise NotImplementedError

    def get_policy(self, state):
        """
        What is the best action to take in the state. Note that because we might
        want to explore, this might not coincide with getAction
        Concretely, this is given by

        policy(s) = arg_max_{a in actions} Q(s,a)

        If many actions achieve the maximal Q-value,
        it doesn't matter which is selected.
        """
        raise NotImplementedError

    def get_action(self, state):
        """
        state: can call state.getLegalActions()
        Choose an action and return it.
        """
        raise NotImplementedError
