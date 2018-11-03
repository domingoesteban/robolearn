import numpy as np


class CostSum(object):
    def __init__(self, costs, weights=None):
        self._costs = costs

        if weights is None:
            weights = np.ones(len(self._costs))

        self._weights = np.array(weights)

        if len(self._costs) != len(self._weights):
            raise AttributeError("The number of cost types and weights"
                                 "do not match %d != %d"
                                 % (len(self._costs), len(self._weights)))

    def eval(self, path):

        # Compute weighted sum of each cost value and derivatives for fist cost
        l, lx, lu, lxx, luu, lux = self._costs[0].eval(path)
        weight = self._weights[0]
        l = l * weight
        lx = lx * weight
        lu = lu * weight
        lxx = lxx * weight
        luu = luu * weight
        lux = lux * weight

        # Cost composition list
        cost_composition = list()
        cost_composition.append(l.copy())

        for i in range(1, len(self._costs)):
            pl, plx, plu, plxx, pluu, plux = self._costs[i].eval(path)
            # print("Cost %d: %f" % (i, sum(pl)))
            weight = self._weights[i]

            cost_composition.append(pl*weight)

            l = l + pl * weight
            lx = lx + plx * weight
            lu = lu + plu * weight
            lxx = lxx + plxx * weight
            luu = luu + pluu * weight
            lux = lux + plux * weight

        # print('lx', lx[-1, :])
        # print('lu', lu[-1, :])
        # print('---')
        # print('lxx', lxx[-1, :, :])
        # print('luu', luu[-1, :, :])
        # print('lxx', np.diag(lxx[-1, :, :]))
        # input('wuuuu')

        return l, lx, lu, lxx, luu, lux, cost_composition
