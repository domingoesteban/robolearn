"""
Common evaluation utilities.
"""

from collections import OrderedDict
from numbers import Number

import numpy as np


def get_generic_path_information(paths, stat_prefix=''):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    returns = np.array([sum(path["rewards"]) for path in paths])

    rewards = np.vstack([path["rewards"].reshape((-1, 1)) for path in paths])
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix,
                                                always_show_all_stats=True))
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix,
                                                always_show_all_stats=True))
    actions = [path["actions"] for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["actions"] for path in paths])
    else:
        actions = np.vstack([path["actions"] for path in paths])
    statistics.update(create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))
    statistics['Num Paths'] = len(paths)

    return statistics


def get_average_returns(paths):
    returns = [float(sum(path["rewards"])) for path in paths]
    return np.mean(returns)


def get_average_multigoal_returns(paths, multigoal_idx):
    returns = [float(sum([r_multi['reward_multigoal'][multigoal_idx]
                          for r_multi in path['env_infos']]))
               for path in paths]
    return np.mean(returns)


def get_average_multigoal_rewards(paths, multigoal_idx):
    n = 0
    accum_r = 0
    for path in paths:
        for r_multi in path['env_infos']:
            accum_r += float(r_multi['reward_multigoal'][multigoal_idx])
            n += 1
    return accum_r/n


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{} {}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats
