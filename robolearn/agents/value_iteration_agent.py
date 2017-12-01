# Code from https://s3-us-west-2.amazonaws.com/cs188websitecontent/projects/release/reinforcement/v1/001/docs/valueIterationAgents.html

from robolearn.utils.mdp import MarkovDecisionProcess
from robolearn.agents.value_estimation_agent import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

    def get_value(self, state):
        """
        Return the value of the state (computed in __init__).
        :param state:
        :return:
        """
        return self.values[state]

    def compute_q_value_from_values(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        raise NotImplementedError

    def compute_action_from_values(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        raise NotImplementedError

    def get_policy(self, state):
        return self.compute_action_from_values(state)

    def get_action(self, state):
        """
        Returns the policy at the state (no exploration).
        :param state:
        :return:
        """
        return self.compute_action_from_values(state)

    def getQValue(self, state, action):
        return self.compute_q_value_from_values(state, action)
