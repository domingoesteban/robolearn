"""
This file defines a cost sum of arbitrary other costs.
Author: C. Finn et al. Code in https://github.com/cbfinn/gps
"""
import copy

from robolearn.costs.config import COST_SUM
from robolearn.costs.cost import Cost


class CostSum(Cost):
    """ A wrapper cost function that adds other cost functions. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_SUM)
        config.update(hyperparams)
        Cost.__init__(self, config)

        self._costs = []
        self._weights = self._hyperparams['weights']

        for cost in self._hyperparams['costs']:
            self._costs.append(cost['type'](cost))

        if len(self._costs) != len(self._weights):
            raise AttributeError("The number of cost types and weights do not match %d != %d" % (len(self._costs),
                                                                                                 len(self._weights)))

    def eval(self, sample):
        """
        Evaluate cost function and derivatives.
        Args:
            sample:  A single sample
        """
        l, lx, lu, lxx, luu, lux = self._costs[0].eval(sample)
        # print("Cost 0: %f" % sum(l))

        # Compute weighted sum of each cost value and derivatives.
        weight = self._weights[0]
        l = l * weight
        lx = lx * weight
        lu = lu * weight
        lxx = lxx * weight
        luu = luu * weight
        lux = lux * weight

        cost_composition = list()
        cost_composition.append(l.copy())

        for i in range(1, len(self._costs)):
            pl, plx, plu, plxx, pluu, plux = self._costs[i].eval(sample)
            # print("Cost %d: %f" % (i, sum(pl)))
            weight = self._weights[i]

            cost_composition.append(pl*weight)

            l = l + pl * weight
            lx = lx + plx * weight
            lu = lu + plu * weight
            lxx = lxx + plxx * weight
            luu = luu + pluu * weight
            lux = lux + plux * weight

        return l, lx, lu, lxx, luu, lux, cost_composition
