from robolearn.agents.agent import Agent
from robolearn.policies.policy_opt.policy_opt import PolicyOpt
from robolearn.policies.policy_opt.policy_opt_random import PolicyOptRandom


class GPSAgent(Agent):
    """
    GPSAgent class: An agent with samples attribute and policy_opt method.
    """
    def __init__(self, act_dim, obs_dim, state_dim, policy_opt=None,
                 agent_name=""):
        super(GPSAgent, self).__init__(act_dim=act_dim, obs_dim=obs_dim,
                                       state_dim=state_dim)

        if policy_opt is None:
            print("Policy optimization not defined."
                  "Using default PolicyOptRandom class!")
            self.policy_opt = PolicyOptRandom({}, self.obs_dim, self.act_dim)
        else:
            if not issubclass(policy_opt['type'], PolicyOpt):
                raise TypeError("'policy_opt' type %s is not a PolicyOpt class"
                                % str(policy_opt['type']))
            # policy_opt['hyperparams']['name'] = agent_name
            policy_opt['hyperparams']['name'] = ""
            self.policy_opt = policy_opt['type'](policy_opt['hyperparams'],
                                                 obs_dim, act_dim)

        # Assign the internal policy in policy_opt class as agent's policy.
        self.policy = self.policy_opt.policy

    # For pickling.
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'policy_opt' in state:
            state.pop('policy_opt')
        if 'policy' in state:
            state.pop('policy')
        return state

    # For unpickling.
    def __setstate__(self, state):
        self.__dict__.update(state)
        #self.__dict__ = state
        #self.__dict__['policy'] = None
