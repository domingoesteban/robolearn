import numpy as np
import matplotlib.pyplot as plt
from robolearn.utils.plots import subplots
from builtins import input

IU_COLORS = [
    'black',
    'red',
    'saddlebrown',
    'green',
    'magenta',
    'orange',
    'blue',
    'cadetblue',
    'mediumslateblue'
]

COMPO_COLORS = [
    'black',
    'red',
    'saddlebrown',
    'green',
    'magenta',
    'orange',
    'blue',
    'cadetblue',
    'mediumslateblue'
]

STATE_COLORS = [
    'black',
    'red',
    'saddlebrown',
    'green',
    'magenta',
    'orange',
    'blue',
    'cadetblue',
    'mediumslateblue'
]

ACTION_COLORS = [
    'black',
    'red',
    'saddlebrown',
    'green',
    'magenta',
    'orange',
    'blue',
    'cadetblue',
    'mediumslateblue'
]


def plot_reward_composition(path_list, ignore_last=True, block=False):
    n_reward_vector = len(path_list['env_infos'][-1]['reward_vector'])
    H = len(path_list['env_infos'])
    fig, axs = subplots(n_reward_vector+1, sharex=True)

    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    fig.subplots_adjust(hspace=0)
    fig.suptitle('Composition of Rewards',
                 fontweight='bold')

    data = np.zeros((n_reward_vector+1, H))

    ts = np.arange(H)
    for rr in range(n_reward_vector):
        for tt in range(H):
            data[rr, tt] = \
                path_list['env_infos'][tt]['reward_vector'][rr]
        axs[rr].plot(ts, data[rr, :], color=COMPO_COLORS[rr], linestyle=':')
        axs[rr].set_ylabel('%02d' % rr)
        # ax = fig.add_subplot(n_reward_vector, 1, rr+1)
        # ax.plot(ts, data)
    data[-1, :] = np.sum(data[:n_reward_vector, :], axis=0)

    if ignore_last:
        rewards_to_plot = n_reward_vector-1
    else:
        rewards_to_plot = n_reward_vector

    for rr in range(rewards_to_plot):
        axs[-1].plot(ts, data[rr, :], linestyle=':', label='%02d' % rr,
                     color=COMPO_COLORS[rr])
    axs[-1].plot(ts, data[-1, :], linewidth=2,
                 color=COMPO_COLORS[n_reward_vector], label='Reward')
    axs[-1].set_ylabel('Reward')
    axs[-1].set_xlabel('Time step')
    axs[-1].legend()

    plt.show(block=block)

    return fig, axs


def plot_reward_composition_v010(cost_list, ignore_last=False,
                                 plot_last=False):
    n_reward_vector = len(cost_list)
    H = len(cost_list[-1])

    fig, axs = subplots(n_reward_vector+1, sharex=True)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Reward composition',
                 fontweight='bold')

    # fig = plt.figure()
    # ax = fig.add_subplot(n_reward_vector, 1, 1)

    data = np.zeros((n_reward_vector+1, H))

    ts = np.arange(H)
    for rr in range(n_reward_vector):
        data[rr, :] = cost_list[rr]
        axs[rr].plot(ts, data[rr, :], color=COMPO_COLORS[rr], linestyle=':')
        axs[rr].set_ylabel('Reward %02d' % rr)
        # ax = fig.add_subplot(n_reward_vector, 1, rr+1)
        # ax.plot(ts, data)
    data[-1, :] = np.sum(data[:n_reward_vector, :], axis=0)

    if ignore_last:
        rewards_to_plot = n_reward_vector-1
    else:
        rewards_to_plot = n_reward_vector

    if plot_last:
        max_t = H
    else:
        max_t = H - 1

    for rr in range(rewards_to_plot):
        axs[-1].plot(ts[:max_t], data[rr, :max_t], linestyle=':',
                     label='%02d' % rr, color=COMPO_COLORS[rr])

    axs[-1].plot(ts[:max_t], data[-1, :max_t], linewidth=2,
                 color=COMPO_COLORS[n_reward_vector], label='Total Reward')
    axs[-1].set_xlabel('Time')
    axs[-1].legend()

    plt.show(block=False)


def plot_reward_iu(path_list, block=False):

    H = len(path_list['rewards'])
    n_unintentional = len(path_list['env_infos'][-1]['reward_multigoal'])
    fig, axs = subplots(n_unintentional+1, sharex=True)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Reward of Intentional and Unintentional Policies',
                 fontweight='bold')

    data = np.zeros((n_unintentional+1, H))

    ts = np.arange(H)
    for tt in range(H):
        for uu in range(n_unintentional):
            data[uu, tt] = \
                path_list['env_infos'][tt]['reward_multigoal'][uu]
        # ax = fig.add_subplot(n_reward_vector, 1, rr+1)
        # ax.plot(ts, data)
    data[-1, :] = path_list['rewards'].squeeze()

    for aa, ax in enumerate(axs[:-1]):
        ax.plot(ts, data[aa, :], linestyle=':', label='U-%02d' % aa,
                color=COMPO_COLORS[aa])
        ax.set_ylabel('Reward U-%02d' % aa)

    axs[-1].plot(ts, data[-1, :], linewidth=2,
                 color=COMPO_COLORS[n_unintentional+1], label='I')
    axs[-1].set_ylabel('Reward Intentional')

    axs[-1].set_xlabel('Time step')
    # axs[-1].legend()

    plt.show(block=block)

    return fig, axs


def plot_weigths_unintentionals(path_list, block=False):
    """Plot the weights of the set of unintentional policies."""
    if 'mixing_coeff' not in path_list['agent_infos'][-1]:
        print('There is not mixing_coeff. Then not plotting anything!')
        return
    H = len(path_list['agent_infos'])
    act_dim = path_list['agent_infos'][-1]['mixing_coeff'].shape[0]
    n_unintentional = path_list['agent_infos'][-1]['mixing_coeff'].shape[1]

    fig, axs = subplots(act_dim, sharex=True)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Mixing weights for Unintentional Policies',
                 fontweight='bold')

    data = np.zeros((H, act_dim, n_unintentional))
    for tt in range(H):
        data[tt] = path_list['agent_infos'][tt]['mixing_coeff']
        # print(tt, '|', data[tt])

    ts = np.arange(H)
    for aa in range(act_dim):
        # axs[aa].plot(ts, data[:, aa, :], color=COMPO_COLORS[aa], linestyle=':')
        axs[aa].plot(ts, data[:, aa, :], linestyle=':')
        axs[aa].set_ylabel('U - %02d' % aa)
        axs[aa].set_xlabel('Time step')
        # axs[aa].set_ylim(-0.1, 1.1)

    plt.show(block=block)

    return fig, axs


def plot_q_vals(path_list, q_fcn, block=False):
    obs = path_list['observations']
    actions = path_list['actions']
    H = obs.shape[0]

    fig, axs = subplots(1, sharex=True)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Q-vals',
                 fontweight='bold')

    q_values = q_fcn.get_values(obs, actions)[0]
    q_values.squeeze(-1)

    ts = np.arange(H)

    axs[-1].plot(ts, q_values)
    axs[-1].set_ylabel('Q-Value')
    axs[-1].set_xlabel('Time step')

    plt.show(block=block)

    return fig, axs


def plot_state_v010(state, state_name=None):

    H = state.shape[0]
    dS = state.shape[1]

    fig, axs = subplots(dS, sharex=True)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)

    if state_name is None:
        state_name = 'State'

    fig.suptitle('%s Trajectory' % state_name, fontweight='bold')

    # fig = plt.figure()
    # ax = fig.add_subplot(n_reward_vector, 1, 1)

    ts = np.arange(H)
    for ss in range(dS):
        axs[ss].plot(ts, state[:, ss], color=STATE_COLORS[ss], linestyle='-')
        axs[ss].set_ylabel('State %02d' % ss)
        # ax = fig.add_subplot(n_reward_vector, 1, rr+1)
        # ax.plot(ts, data)
    axs[-1].set_xlabel('Time')
    # axs[-1].legend()

    plt.show(block=False)


def plot_action_v010(action, action_name=None):

    H = action.shape[0]
    dA = action.shape[1]

    fig, axs = subplots(dA, sharex=True)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)

    if action_name is None:
        action_name = 'Action'

    fig.suptitle('%s Trajectory' % action_name, fontweight='bold')

    # fig = plt.figure()
    # ax = fig.add_subplot(n_reward_vector, 1, 1)

    ts = np.arange(H)
    for aa in range(dA):
        axs[aa].plot(ts, action[:, aa], color=ACTION_COLORS[aa], linestyle='-')
        axs[aa].set_ylabel('%s %02d' % (action_name, aa))
        # ax = fig.add_subplot(n_reward_vector, 1, rr+1)
        # ax.plot(ts, data)
    axs[-1].set_xlabel('Time')
    # axs[-1].legend()

    plt.show(block=False)

