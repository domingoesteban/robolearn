import numpy as np
import matplotlib.pyplot as plt
from robolearn.utils.plots import subplots
from robolearn.utils.plots import get_csv_data
from robolearn.utils.plots import set_latex_plot
from builtins import input
import os
import json

LOG_FILE = 'progress.csv'
PARAMS_FILE = 'params.pkl'
VARIANT_FILE = 'variant.json'

N_UNINTENTIONS = 2


def plot_multiple_process_iu_returns(
        csv_file_dict, block=False,
        max_iter=500,
        steps_per_iter=None,
        latex_plot=True,
        fig_name_prefix=None,
):
    """
    It plots the 'Test Returns Mean' label of the progress file.
    If algorithm of experiment is HIU, the unintentional data is considered an
    independent experiment.
    The keys of the categories dict are used for the axis title.
    The keys of the experiments dict are used for the labels in the legend.
    Args:
        csv_file_dict (dict): A dictionary of categories.
            - Category (dict): One figure per category.
                - Experiment (list): One for each figure.
                    - Seed (String): Experiment run with a specific seed.

            dict(
                Criteria1 = dict(
                    Experiment1 = list(
                        ('full_path_of_experiment_with_seed_X', [-1])
                        ('full_path_of_experiment_with_seed_Y', [-1])
                        ('full_path_of_experiment_with_seed_Z', [-1])
                    )
                    Experiment2 = list(
                        'full_path_of_experiment_with_seed_X', [-1, 0, 1])
                        'full_path_of_experiment_with_seed_Y', [-1, 0, 1])
                        'full_path_of_experiment_with_seed_Z', [-1, 0, 1])
                    )
                )
                Criteria2 = dict(
                    Experiment1 = list(
                        ('full_path_of_experiment_with_seed_X', [-1])
                        ('full_path_of_experiment_with_seed_Y', [-1])
                        ('full_path_of_experiment_with_seed_Z', [-1])
                    )
                    Experiment2 = list(
                        'full_path_of_experiment_with_seed_X', [-1, 0, 1])
                        'full_path_of_experiment_with_seed_Y', [-1, 0, 1])
                        'full_path_of_experiment_with_seed_Z', [-1, 0, 1])
                    )
                )
            )

        block (bool): Block the figure
        max_iter:
        steps_per_iter:
        latex_plot:

    Returns:

    """
    labels_to_plot = ['Test Returns Mean']
    labels_y_axis = ['Average Return']

    if latex_plot:
        set_latex_plot()

    i_labels = list()
    u_labels = list()
    for ll, label in enumerate(labels_to_plot):
        for uu in range(N_UNINTENTIONS):
            new_string = ('[U-%02d] ' % uu) + label
            u_labels.append(new_string)
        intent_string = '[I] ' + label
        i_labels.append(intent_string)

    categories = list(csv_file_dict.keys())

    if steps_per_iter is None:
        x_data = np.arange(0, max_iter)
        x_label = 'Iterations'
    else:
        x_data = np.arange(0, max_iter) * steps_per_iter
        x_label = 'Time steps (%s)' % '{:.0e}'.format(steps_per_iter)

    for cc, cate in enumerate(categories):
        # ######## #
        # Get data #
        # ######## #
        catego_dict = csv_file_dict[cate]
        n_subplots = len(i_labels)
        expts = list(catego_dict.keys())

        nexpts = len(expts)
        nseeds = len(catego_dict[expts[-1]])
        niters = max_iter
        nunint = N_UNINTENTIONS

        all_data = [np.zeros((nexpts, nseeds, nunint+1, niters))
                    for _ in i_labels]

        algos = list()
        infos = list()

        for ee, expt in enumerate(expts):
            seeds = catego_dict[expt]
            algos.append(list())

            for ss, seed in enumerate(seeds):
                data_dir = catego_dict[expt][ss][0]
                info = [ii + 1 for ii in catego_dict[expt][ss][1]]  # Because Main is 0 not -1

                variant_file = os.path.join(data_dir,
                                            VARIANT_FILE)
                with open(variant_file) as json_data:
                    algo_name = json.load(json_data)['algo_name']
                    algos[-1].append(algo_name)
                if ss == 0:
                    infos.append(info)

                csv_file = os.path.join(data_dir, LOG_FILE)
                # print(csv_file)
                if algo_name.upper() in ['HIUSAC', 'HIUSACNEW', 'HIUDDPG']:
                    data_csv = get_csv_data(csv_file, i_labels + u_labels)
                else:
                    data_csv = get_csv_data(csv_file, i_labels)

                for dd in range(n_subplots):
                    if data_csv.shape[-1] < max_iter:
                        raise ValueError('por ahora hay solo %02d iters. En %s'
                                         % (data_csv.shape[-1],
                                            csv_file))
                    n_data = data_csv.shape[0]
                    all_data[dd][ee, ss, :n_data, :] = data_csv[:, :max_iter]

                # TODO: Assuming only AvgReturn
                rew_scales = catego_dict[expt][ss][2]
                for ii, rew_scale in zip(info, rew_scales):
                    all_data[-1][ee, ss, ii, :] *= 1 / rew_scale

        # ############# #
        # Plot the data #
        # ############# #
        fig, axs = subplots(n_subplots)
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        fig.subplots_adjust(hspace=0)
        # fig.suptitle('Expected Return - '+str(cate), fontweight='bold')
        if fig_name_prefix is None:
            fig_name_prefix = ""
        fig_title = (fig_name_prefix + 'Expected Return '+str(cate)).replace(" ", "_")
        fig.canvas.set_window_title(fig_title)
        lines = list()
        labels = list()

        for aa, ax in enumerate(axs):
            for ee, expt in enumerate(expts):
                print('----> cat:', cate, '|', expt, all_data[aa][ee, :, :, :].shape, '| info:', infos[ee])
                for ii, iu_idx in enumerate(infos[ee]):
                # for ii in range(max_unint):
                    data_mean = np.mean(all_data[aa][ee, :, iu_idx, :], axis=0)
                    data_std = np.std(all_data[aa][ee, :, iu_idx, :], axis=0)
                    # 85:1.440, 90:1.645, 95:1.960, 99:2.576
                    ax.fill_between(
                        x_data,
                        (data_mean - 0.5 * data_std),
                        (data_mean + 0.5 * data_std), alpha=.3)
                    mean_plot = ax.plot(x_data, data_mean)[0]

                    if aa == 0:
                        lines.append(mean_plot)
                        if algos[ee][ii].upper() == 'HIUSAC':
                            if iu_idx == 0:
                                i_suffix = ' [I]'
                            else:
                                i_suffix = ' [U-%02d]' % iu_idx
                            labels.append(expt + i_suffix)
                        else:
                            labels.append(expt)

            xdiff = x_data[1] - x_data[0]
            ax.set_xlim(x_data[0]-xdiff, x_data[-1] + xdiff)
            ax.set_ylabel(labels_y_axis[aa], fontsize=50)
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.xaxis.set_major_locator(plt.MultipleLocator(50))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(10))

        axs[-1].set_xlabel(x_label, fontsize=50)
        plt.setp(axs[-1].get_xticklabels(), visible=True)

        legend = fig.legend(lines, labels, loc='lower right', ncol=1,
        # legend = fig.legend(lines, labels, loc=(-1, 0), ncol=1,
                            labelspacing=0., prop={'size': 40})
        fig.set_size_inches(19, 11)  # 1920 x 1080
        fig.tight_layout()
        legend.draggable(True)

    plt.show(block=block)
