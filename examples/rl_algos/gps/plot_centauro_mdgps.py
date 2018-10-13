import argparse
import numpy as np
import matplotlib.pyplot as plt
from robolearn.utils.plots.core import subplots
from robolearn.utils.plots import get_csv_data


def main(args):
    # plot_process_general_data(csv_file=args.file, n_unintentional=args.un,
    #                           block=False)

    # plot_process_iu_values_errors(csv_file=args.file, n_unintentional=args.un,
    #                               block=False)
    #
    # plot_process_iu_policies(csv_file=args.file, n_unintentional=args.un,
    #                          block=False)
    #
    # plot_process_iu_returns(csv_file=args.file, n_unintentional=args.un,
    #                         block=False)

    plot_global_pol_return(csv_file=args.file, n_local_pols=args.n_locals)


def plot_global_pol_return(csv_file, n_local_pols=1, block=False):
    labels_to_plot = ['Global Mean Return']

    # Add Intentional-Unintentional Label
    new_labels = list()
    for label in labels_to_plot:
        for uu in range(n_local_pols):
            new_string = ('[Cond-%02d] ' % uu) + label
            new_labels.append(new_string)

        # new_string = '' + label
        # new_labels.append(new_string)

    n_subplots = len(labels_to_plot) * (n_local_pols)

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

    if n_subplots > 0:
        axs[-1].set_xlabel('Episodes')
        plt.setp(axs[-1].get_xticklabels(), visible=True)

    plt.show(block=block)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default='./progress.csv',
                        help='path to the progress.csv file')
    parser.add_argument('--n_locals', type=int, default=1,
                        help='N locals')
    args = parser.parse_args()

    main(args)
    input('Press a key to close script')
