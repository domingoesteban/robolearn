from .base import Cost


class CostSum(Cost):
    def __init__(self, costs, weights):
        self._costs = costs

        self._weights = weights

    def eval(self, states, actions, gradients=False):
        cost_composition = list()
        sum_l = 0

        if gradients:
            sum_ls = 0
            sum_la = 0
            sum_lss = 0
            sum_laa = 0
            sum_las = 0

        for ii in range(1, len(self._costs)):
            # Eval First Cost
            l, cost_info = self._costs[ii].eval(states, actions,
                                                gradients=gradients)

            sum_l += l

            if gradients:
                sum_ls += cost_info['ls']*self._weights[ii]
                sum_la += cost_info['la']*self._weights[ii]
                sum_lss += cost_info['lss']*self._weights[ii]
                sum_laa += cost_info['laa']*self._weights[ii]
                sum_las += cost_info['las']*self._weights[ii]

        if gradients:
            info_dict = dict(
                ls=sum_ls,
                la=sum_la,
                lss=sum_lss,
                laa=sum_laa,
                las=sum_las,
                cost_composition=cost_composition,
            )
        else:
            info_dict = dict(
                cost_composition=cost_composition,
            )

        return sum_l, info_dict
