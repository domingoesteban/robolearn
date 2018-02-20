# Based on: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
import numpy as np
from collections import deque
import bisect


class ExperienceBuffer(object):
    """
    Experience Buffer class for Good or Bad Experience
    """
    def __init__(self, size, good_or_bad, temp_or_cost):
        """

        Args:
            size: Buffer size
            good_or_bad: 'good' or 'bad' experience
            temp_or_cost: 'temp' or 'cost' criterion
        """
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
            self._total_cost = list()

    def __len__(self):
        return len(self._trajs)

    def add(self, trajs, costs):
        """ Add a trajectory (with its cost) to the experience buffer.

        Args:
            trajs:
            costs:

        Returns:

        """
        cost_sums = [np.sum(cost) for cost in costs]
        if self._temp_or_cost == 'cost' and not len(self._trajs) < self._maxsize:
            input_order = self._get_no_extreme_idx(cost_sums, len(costs))
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
                    replace = self._compare_with_no_extreme(cost_sums[ii])
                    if replace >= 0:
                        self._trajs[replace] = traj
                        self._costs[replace] = cost

    def get_trajs(self, number=1):
        if self._temp_or_cost == 'temp':
            idxs = list(range(-number, 0))
        else:
            idxs = self._get_extremes(number)
        return [self._trajs[ii] for ii in idxs]

    def get_costs(self, number=1):
        if self._temp_or_cost == 'temp':
            idxs = list(range(-number, 0))
        else:
            idxs = self._get_extremes(number)
        return [self._costs[ii] for ii in idxs]

    def get_trajs_and_costs(self, number=1):
        if self._temp_or_cost == 'temp':
            idxs = list(range(-number, 0))
        else:
            idxs = self._get_extremes(number)
        return ([self._trajs[ii] for ii in idxs],
                [self._costs[ii] for ii in idxs])

    def _compare_with_no_extreme(self, cost):
        no_extreme_idx = self._get_no_extremes(1)[0]
        if self._good_or_bad == 'bad':
            if cost >= np.sum(self._costs[no_extreme_idx]):
                print('It is bigger!!')
                return no_extreme_idx
            else:
                return -1
        else:
            if cost <= np.sum(self._costs[no_extreme_idx]):
                print('It is smaller!!')
                return no_extreme_idx
            else:
                return -1

    def _get_no_extremes(self, number):
        costs = [np.sum(cost) for cost in self._costs]
        return self._get_no_extreme_idx(costs, number)

    def _get_no_extreme_idx(self, values, number):
        if self._good_or_bad == 'bad':
            if len(values) <= number:
                temp_idx = get_smaller_idx(values, number-1)
                all_idx = list(range(number))
                not_in_list = np.setdiff1d(all_idx, temp_idx)
                return np.concatenate((temp_idx, not_in_list))
            else:
                return get_smaller_idx(values, number)

        else:
            if len(values) <= number:
                temp_idx = get_bigger_idx(values, number-1)
                all_idx = list(range(number))
                not_in_list = np.setdiff1d(all_idx, temp_idx)
                return np.concatenate((temp_idx, not_in_list))
            else:
                return get_bigger_idx(values, number)

    def _get_extremes(self, number):
        costs = [np.sum(cost) for cost in self._costs]
        return self._get_extreme_idx(costs, number)

    def _get_extreme_idx(self, values, number):
        if self._good_or_bad == 'bad':
            if len(values) <= number:
                temp_idx = get_bigger_idx(values, number-1)
                all_idx = list(range(number))
                not_in_list = np.setdiff1d(all_idx, temp_idx)
                return np.concatenate((temp_idx, not_in_list))
            else:
                return get_bigger_idx(values, number)

        else:
            if len(values) <= number:
                temp_idx = get_smaller_idx(values, number-1)
                all_idx = list(range(number))
                not_in_list = np.setdiff1d(all_idx, temp_idx)
                return np.concatenate((temp_idx, not_in_list))
            else:
                return get_smaller_idx(values, number)


def get_smaller_idx(values, number):
    return np.argpartition(values, number)[:number]


def get_bigger_idx(values, number):
    return np.argpartition(values, -number)[-number:]
