# Based on: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
import numpy as np
from collections import deque

class ExperienceBuffer(object):
    def __init__(self, size, good_or_bad, temp_or_cost):
        self._maxsize = size

        if good_or_bad.lower() not in ['good', 'bad']:
            raise ValueError('Wrong option.')
        self._good_or_bad = good_or_bad.lower()

        if temp_or_cost.lower() not in ['temp', 'cost']:
            raise ValueError('Wrong option.')
        self._temp_or_cost = temp_or_cost.lower()

        if self._temp_or_cost == 'temp':
            self._trajs = deque()
            self._costs = deque()
        else:
            self._trajs = list()
            self._costs = list()

    def __len__(self):
        return len(self._trajs)

    def add(self, trajs, costs):
        cost_sums = [np.sum(cost) for cost in costs]
        if self._temp_or_cost == 'cost' and not len(self._trajs) < self._maxsize:
            input_order = self._get_extreme_idx(cost_sums, len(costs))
        else:
            input_order = np.arange(len(trajs))

        for ii in input_order:
            traj = trajs[ii]
            cost = costs[ii]
            if len(self._trajs) < self._maxsize:
                self._trajs.append(traj)
                self._costs.append(cost)
            else:
                if self._temp_or_cost == 'temp':
                    self._trajs.popleft()
                    self._costs.popleft()
                    self._trajs.append(traj)
                    self._costs.append(cost)
                else:
                    replace = self._compare_with_extreme(cost_sums[ii])
                    if replace:
                        self._trajs[replace] = traj
                        self._costs[replace] = cost

            if self._temp_or_cost == 'cost':
                self.

    def _compare_with_extreme(self, cost):
        extreme_idx = self._get_extremes(1)[0]
        if self._good_or_bad == 'bad':
            if cost >= np.sum(self._costs[extreme_idx]):
                return extreme_idx
            else:
                return False
        else:
            if cost <= np.sum(self._costs[extreme_idx]):
                return extreme_idx
            else:
                return False

    def _get_extremes(self, number):
        costs = [np.sum(cost) for cost in self._costs]
        return self._get_extreme_idx(costs, number)

    def _get_extreme_idx(self, values, number):
        input('Values %r | number %r' % (values, number))
        if self._good_or_bad == 'bad':
            if len(values) < number:
                return self._get_smaller_idx(values, number)
            else:
                temp_idx = self._get_smaller_idx(values, number-1)
                all_idx = list(range(number))
                not_in_list = all_idx - temp_idx
                raise ValueError(not_in_list)

        else:
            if len(values) > number:
                return self._get_bigger_idx(values, number)
            else:
                raise NotImplementedError


    @staticmethod
    def _get_smaller_idx(values, number):
        return np.argpartition(values, number)[:number]

    @staticmethod
    def _get_bigger_idx(values, number):
        return np.argpartition(values, -number)[-number:]
