import numpy as np
import matplotlib.pyplot as plt
from robolearn.utils.plots import subplots
import pandas as pd
from builtins import input


def plot_process_iu_returns(csv_file, n_unintentional=None, block=False):
    labels_to_plot = ['Test AverageReturn']

    if n_unintentional is None:
        n_unintentional = 0
    else:
        n_unintentional += 1

    # Add Intentional-Unintentional Label
    new_labels = list()
    for label in labels_to_plot:
        for uu in range(n_unintentional):
            new_string = ('[U-%02d] ' % uu) + label
            new_labels.append(new_string)

        new_string = '[I] ' + label
        new_labels.append(new_string)

    n_subplots = len(labels_to_plot) * (n_unintentional + 1)

    data = get_csv_data(csv_file, new_labels)

    fig, axs = subplots(n_subplots)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Avg Return and Avg Reward',
                 fontweight='bold')

    for aa, ax in enumerate(axs):
        ax.plot(data[aa])
        ax.set_ylabel(new_labels[aa])
        plt.setp(ax.get_xticklabels(), visible=False)

    axs[-1].set_xlabel('Episodes')
    plt.setp(axs[-1].get_xticklabels(), visible=True)

    plt.show(block=block)


def plot_process_iu_policies(csv_file, n_unintentional=None, block=False,
                             plot_initial=False):
    labels_to_plot = [
        'Mixing Weights',
        'Pol KL Loss',
        'Rewards',
        'Policy Entropy',
        # 'Log Policy Target',
        # 'Policy Mean',
        'Policy Std'
        ]

    if n_unintentional is None:
        n_unintentional = 0
    else:
        n_unintentional += 1

    if plot_initial:
        idx0 = 0
    else:
        idx0 = 1

    # Add Intentional-Unintentional Label
    new_labels = list()
    for ll, label in enumerate(labels_to_plot):
        for uu in range(n_unintentional):
            new_string = ('[U-%02d] ' % uu) + label
            new_labels.append(new_string)

        if ll > 0:
            new_string = '[I] ' + label
            new_labels.append(new_string)

    n_subplots = len(labels_to_plot)

    data = get_csv_data(csv_file, new_labels)

    fig, axs = subplots(n_subplots)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Policy Properties',
                 fontweight='bold')

    idx_counter = 0
    lines = list()
    labels = list()
    for aa, ax in enumerate(axs):
        for uu in range(n_unintentional):
            line, = ax.plot(data[idx_counter, idx0:], label='[U-%02d] ' % uu)
            idx_counter += 1
            if aa == 1:
                lines.append(line)
                labels.append('[U-%02d] ' % uu)

        if aa > 0:
            line, = ax.plot(data[idx_counter, idx0:], label='[I]')
            idx_counter += 1
            if aa == 1:
                lines.append(line)
                labels.append('[I]')

        ax.set_ylabel(labels_to_plot[aa])
        plt.setp(ax.get_xticklabels(), visible=False)

    axs[-1].set_xlabel('Episodes')
    plt.setp(axs[-1].get_xticklabels(), visible=True)

    fig.legend(lines, labels, loc='right', ncol=1, labelspacing=0.)

    plt.show(block=block)


def plot_process_iu_values_errors(csv_file, n_unintentional=None, block=False):
    labels_to_plot = ['Qf Loss', 'Vf Loss']

    if n_unintentional is None:
        n_unintentional = 0
    else:
        n_unintentional += 1

    # Add Intentional-Unintentional Label
    new_labels = list()
    for label in labels_to_plot:
        for uu in range(n_unintentional):
            new_string = ('[U-%02d] ' % uu) + label
            new_labels.append(new_string)

        new_string = '[I] ' + label
        new_labels.append(new_string)

    n_subplots = len(labels_to_plot) * (n_unintentional + 1)

    data = get_csv_data(csv_file, new_labels)

    fig, axs = subplots(n_subplots)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Value Functions Errors',
                 fontweight='bold')

    for aa, ax in enumerate(axs):
        ax.plot(data[aa])
        ax.set_ylabel(new_labels[aa])
        plt.setp(ax.get_xticklabels(), visible=False)

    axs[-1].set_xlabel('Episodes')
    plt.setp(axs[-1].get_xticklabels(), visible=True)

    plt.show(block=block)


def plot_process_general_data(csv_file, n_unintentional=None, block=False):
    labels_to_plot = [
        # 'mean-sq-bellman-error',
        # 'Bellman Residual (QFcn)',
        # 'Surrogate Cost (Policy)',
        # 'return-average',
        'Exploration Returns Mean',
        'Test Returns Mean',
        # 'episode-length-min',
        # 'episode-length-max',

        # 'Log Pis'
    ]

    # if n_unintentional is None:
    #     n_unintentional = 0
    # else:
    #     n_unintentional += 1
    n_unintentional = 0
    #
    # # Add Intentional-Unintentional Label
    # new_labels = list()
    # for label in labels_to_plot:
    #     for uu in range(n_unintentional):
    #         new_string = ('[U-%02d] ' % uu) + label
    #         new_labels.append(new_string)
    #
    #     new_string = '[I] ' + label
    #     new_labels.append(new_string)

    new_labels = labels_to_plot

    n_subplots = len(labels_to_plot) * (n_unintentional + 1)

    data = get_csv_data(csv_file, new_labels)

    fig, axs = subplots(n_subplots)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.suptitle('General Info',
                 fontweight='bold')

    for aa, ax in enumerate(axs):
        ax.plot(data[aa])
        ax.set_ylabel(new_labels[aa])
        plt.setp(ax.get_xticklabels(), visible=False)

    axs[-1].set_xlabel('Episodes')
    plt.setp(axs[-1].get_xticklabels(), visible=True)

    plt.show(block=block)


def plot_process_haarnoja(csv_file, n_unintentional=None, block=False):
    labels_to_plot = ['return-average', 'episode-length-avg', 'log-pi-mean', 'log-sigs-mean']

    if n_unintentional is None:
        n_unintentional = 0
    else:
        n_unintentional += 1

    # Add Intentional-Unintentional Label
    new_labels = list()
    for label in labels_to_plot:
        for uu in range(n_unintentional):
            new_string = ('[U-%02d] ' % uu) + label
            new_labels.append(new_string)

        # new_string = '[I] ' + label
        new_string = label
        new_labels.append(new_string)

    n_subplots = len(labels_to_plot) * (n_unintentional + 1)

    data = get_csv_data(csv_file, new_labels)

    fig, axs = subplots(n_subplots)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Avg Return and Avg Reward',
                 fontweight='bold')

    for aa, ax in enumerate(axs):
        ax.plot(data[aa])
        ax.set_ylabel(new_labels[aa])
        plt.setp(ax.get_xticklabels(), visible=False)

    axs[-1].set_xlabel('Episodes')
    plt.setp(axs[-1].get_xticklabels(), visible=True)

    plt.show(block=block)


def get_csv_data(csv_file, labels):
    data, all_labels = get_csv_data_and_labels(csv_file)

    n_data = data.shape[0]

    new_data = np.zeros((len(labels), n_data))

    print(all_labels)
    for ll, name in enumerate(labels):
        if name in all_labels:
            idx = all_labels.index(name)
            new_data[ll, :] = data[:, idx]
        else:
            raise ValueError("Label '%s' not available in file '%s'"
                             % (name, csv_file))

    return new_data


def get_csv_data_and_labels(csv_file):
    # Read from CSV file
    series = pd.read_csv(csv_file)

    data = series.as_matrix()
    labels = list(series)

    return data, labels