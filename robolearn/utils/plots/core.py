import sys
import traceback
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def get_csv_data(csv_file, labels, space_separated=False):
    data, all_labels = get_csv_data_and_labels(csv_file,
                                               space_separated=space_separated)

    for label in all_labels:
        print(label)
    print('***\n'*3)
    n_data = data.shape[0]

    new_data = np.zeros((len(labels), n_data))

    # # Uncomment for debugging
    # print(all_labels)

    for ll, name in enumerate(labels):
        if name in all_labels:
            idx = all_labels.index(name)
            try:
                new_data[ll, :] = data[:, idx]
            except Exception:
                print(traceback.format_exc())
                print("Error with data in %s" % csv_file)
                sys.exit(1)
        else:
            raise ValueError("Label '%s' not available in file '%s'"
                             % (name, csv_file))

    return new_data


def get_csv_data_and_labels(csv_file, space_separated=False):
    # Read from CSV file
    try:
        if space_separated:
            series = pd.read_csv(csv_file, delim_whitespace=True)
        else:
            series = pd.read_csv(csv_file)
    except Exception:
        print(traceback.format_exc())
        print("Error reading %s" % csv_file)
        sys.exit(1)

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
