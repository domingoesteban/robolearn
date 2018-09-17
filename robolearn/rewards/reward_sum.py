from .base import Reward


class RewardSum(Reward):
    def __init__(self, rewards, weights):
        """

        Args:
            rewards: Iterable with reward functions
            weights: Iterable with rewards weights
        """
        self._rewards = rewards

        self._weights = weights

    def eval(self, states, actions, gradients=False):
        reward_composition = list()
        sum_l = 0

        if gradients:
            sum_ls = 0
            sum_la = 0
            sum_lss = 0
            sum_laa = 0
            sum_las = 0

        for ii in range(1, len(self._rewards)):
            # Eval First Reward
            l, reward_info = self._rewards[ii](states, actions,
                                               gradients=gradients)

            sum_l += l

            if gradients:
                sum_ls += reward_info['ls']*self._weights[ii]
                sum_la += reward_info['la']*self._weights[ii]
                sum_lss += reward_info['lss']*self._weights[ii]
                sum_laa += reward_info['laa']*self._weights[ii]
                sum_las += reward_info['las']*self._weights[ii]

        if gradients:
            info_dict = dict(
                ls=sum_ls,
                la=sum_la,
                lss=sum_lss,
                laa=sum_laa,
                las=sum_las,
                cost_composition=reward_composition,
            )
        else:
            info_dict = dict(
                cost_composition=reward_composition,
            )

        return sum_l, info_dict
