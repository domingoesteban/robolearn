class Policy(object):
    def eval(self, **kwargs):
        """
        Abstract method used for evaluate a policy for specified parameters.
        It returns an action for a state/observation.
        :return: Action tensor u.
        """
        raise NotImplementedError

    def get_params(self):
        """
        Get policy parameters.
        :return: 
        """
        raise NotImplementedError
