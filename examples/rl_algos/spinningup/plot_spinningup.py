import argparse
import numpy as np
import matplotlib.pyplot as plt
from robolearn.utils.plots import subplots
from robolearn.utils.plots import get_csv_data


def main(args):
    labels_to_plot = ['AverageEpRet']

    data = get_csv_data(args.file, labels_to_plot, space_separated=True)

    fig, axs = subplots(1)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Avg Return', fontweight='bold')

    max_iter = len(data[-1])
    if args.max_iter > 0:
        max_iter = args.max_iter

    for aa, ax in enumerate(axs):
        ax.plot(data[aa][:max_iter])
        ax.set_ylabel(labels_to_plot[aa])
        plt.setp(ax.get_xticklabels(), visible=False)

    axs[-1].set_xlabel('Episodes')
    plt.setp(axs[-1].get_xticklabels(), visible=True)

    plt.show(block=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default='./progress.txt',
                        help='path to the progress.csv file')
    parser.add_argument('--max_iter', '-i', type=int, default=-1,
                        help='Unintentional id')
    args = parser.parse_args()

    main(args)
    input('Press a key to close script')
