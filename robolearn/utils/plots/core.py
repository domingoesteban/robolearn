import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def get_csv_data(csv_file, labels):
    data, all_labels = get_csv_data_and_labels(csv_file)

    n_data = data.shape[0]

    new_data = np.zeros((len(labels), n_data))

    # # Uncomment for debugging
    # print(all_labels)

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


def set_latex_plot():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # rc('font', **{'family': 'serif','serif':['Times']})
    matplotlib.rcParams['font.family'] = ['serif']
    matplotlib.rcParams['font.serif'] = ['Times New Roman']


def subplots(*args, **kwargs):
    fig, axs = plt.subplots(*args, **kwargs)

    if isinstance(axs, np.ndarray):
        for aa in axs:
            axis_format(aa)
    else:
        axis_format(axs)

    return fig, axs


def fig_format(fig):
    fig.subplots_adjust(hspace=0)
    fig.set_facecolor((1, 1, 1))


def axis_format(axis):
    # axis.tick_params(axis='x', labelsize=25)
    # axis.tick_params(axis='y', labelsize=25)
    axis.tick_params(axis='x', labelsize=15)
    axis.tick_params(axis='y', labelsize=15)

    # Background
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.xaxis.grid(color='white', linewidth=2)
    axis.set_facecolor((0.917, 0.917, 0.949))
