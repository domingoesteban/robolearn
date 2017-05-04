class Policy(object):
    def eval(self, x, obs, t, noise=None):
        """
        Return an action for a state.
        :param x: State vector. 
        :param obs: Observation vector.
        :param t: Time step.
        :param noise: Action noise. This will be scaled by the variance.
        :return: Action u.
        """
        raise NotImplementedError

    def get_params(self):
        """
        Get policy parameters
        :return: 
        """
        raise NotImplementedError

    def nans_like(self):
        """
        Return a new policy object with the same dimensions but all values
         filled with NaNs.
        :return: New NaN policy
        """
        raise NotImplementedError
