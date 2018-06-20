import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main(args):
    series = pd.read_csv(args.file)

    data = series.as_matrix()

    labels = list(series)
    print('%'*20)
    print('Available labels are: \n', labels)
    print('%'*20)

    labels_names_to_plot = [
        'mean-sq-bellman-error',
        'Bellman Residual (QFcn)',
        'Surrogate Cost (Policy)',
        'return-average',
        'Exploration Returns',
        'Test Returns Mean',
        'episode-length-min',
        'episode-length-max',

        'Log Pis'
        # '[I] Test AverageReturn',
        # '[I] Test Returns Mean',
        # '[I] Test Rewards Mean',

        # '[ACCUM] QF Loss',
        # '[ACCUM] VF Loss',
        # '[ACCUM] Policy Loss',

        '[U-00] Test AverageReturn',
        '[U-00] Rewards',
        '[U-00] Policy Entropy',
        '[U-00] Log Policy Target',
        '[U-00] Policy Mean',
        '[U-00] Policy Log Std',
        '[U-00] Qf Loss',
        '[U-00] Vf Loss',

        # '[U-01] Test AverageReturn',
        # '[U-01] Rewards',
        # '[U-01] Policy Entropy',
        # '[U-01] Log Policy Target',
        # '[U-01] Policy Mean',
        # '[U-01] Policy Log Std',
        # '[U-01] Qf Loss',
        # '[U-01] Vf Loss',


        # '[I] Test Returns Mean',
        # '[U-00] Test Returns Mean',
        # '[U-01] Test Returns Mean',
    ]
    labels_to_plot = list()
    for name in labels_names_to_plot:
        if name in labels:
            idx = labels.index(name)
            print(name, 'is the index', idx)
            labels_to_plot.append(idx)

    fig, ax = plt.subplots(len(labels_to_plot), 1)

    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    for ll, idx in enumerate(labels_to_plot):
        ax[ll].plot(data[:, idx])
        ax[ll].set_xlabel('Epoch')
        ax[ll].set_ylabel(labels[idx])
    plt.show(block=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default='./parameters.pkl',
                        help='path to the progress.csv file')
    args = parser.parse_args()

    main(args)
    input('Press a key to close script')
