import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import os, sys

# NOTE: IT REQUIRES TO GENERATE ALL THE REQUIRED DATA IN DIR_NAME

dir_name = 'state_data_mdgps_log/run_00/'

cond = 3  # Training condition to plot
option = -1  # Option:-1:2d | -2:2dObst | -3:2dTgt | Positive: specified state idx
block = True  # Block plot or not (for visualization)

tgt_positions = [(0.3693,   0.6511),
                 (0.3913,   0.6548),
                 (0.3296,   0.6863),
                 (0.6426,   0.4486),
                 (0.1991,   0.7172)]
obst_positions = [(0.6545,  -0.0576),
                  (0.6964,  -0.0617),
                  (0.6926,  -0.0929),
                  (0.6778,  -0.013),
                  (0.6781,  -0.0882)]

safe_distance = 0.15

if option in [-1, -3]:
    idx_to_plot = [6, 7]
    plt.rcParams["figure.figsize"] = (5, 5)
elif option == -2:
    idx_to_plot = [9, 10]
    plt.rcParams["figure.figsize"] = (5, 5)
else:
    idx_to_plot = [option]
    plt.rcParams["figure.figsize"] = (30, 15)

file_path = os.path.dirname(os.path.realpath(__file__)) + '/../' + dir_name

lines = list()
labels = list()
for itp in range(1, 50):

    file_to_load = file_path + 'all_x_itr_%02d_cond_%02d.pkl' % (itp, cond)

    all_x = pickle.load(open(file_to_load, 'rb'))

    sample_data = pickle.load(open(file_to_load, 'rb'))

    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(hspace=0)
    fig.canvas.set_window_title('Traj Iteration %02d '
                                'Condition %02d' % (itp, cond))
    fig.set_facecolor((1, 1, 1))

    mus_new = all_x['new']
    mus_prev = all_x['prev']
    mus_bad = all_x['bad']
    mus_good = all_x['good']

    T = mus_new.shape[0]
    mus_to_plot = np.zeros((T, len(idx_to_plot)))
    mus_prev_to_plot = np.zeros((T, len(idx_to_plot)))
    mus_bad_to_plot = np.zeros((T, len(idx_to_plot)))
    mus_good_to_plot = np.zeros((T, len(idx_to_plot)))

    if option in [-1, -3]:
        list_to_substact = tgt_positions
    else:
        list_to_substact = obst_positions

    if option in [-1, -2, -3]:
        for ii, index in enumerate(idx_to_plot):
            mus_to_plot[:, ii] = list_to_substact[cond][ii] + mus_new[:, index]
            mus_prev_to_plot[:, ii] = list_to_substact[cond][ii] + mus_prev[:, index]
            mus_bad_to_plot[:, ii] = list_to_substact[cond][ii] + mus_bad[:, index]
            mus_good_to_plot[:, ii] = list_to_substact[cond][ii] + mus_good[:, index]
    else:
        for ii, index in enumerate(idx_to_plot):
            mus_to_plot[:, ii] = mus_new[:, index]
            mus_prev_to_plot[:, ii] = mus_prev[:, index]
            mus_bad_to_plot[:, ii] = mus_bad[:, index]
            mus_good_to_plot[:, ii] = mus_good[:, index]

    if option not in [-1, -2, -3]:
        # ###########
        # PER STATE #
        # ###########

        line = ax.plot(np.arange(T)*0.03,
                       mus_prev_to_plot[:],
                       c='black')[0]
        if itp == 1:
            lines.append(line)
            labels.append('Previous Traj.')
        line = ax.plot(np.arange(T)*0.03,
                       mus_bad_to_plot[:],
                       c='red')[0]
        if itp == 1:
            lines.append(line)
            labels.append('Bad Traj.')

        line = ax.plot(np.arange(T)*0.03,
                       mus_good_to_plot[:],
                       c='green')[0]
        if itp == 1:
            lines.append(line)
            labels.append('Good Traj.')
        line = ax.plot(np.arange(T)*0.03,
                       mus_to_plot[:],
                       c='blue')[0]
        if itp == 1:
            lines.append(line)
            labels.append('Updated Traj.')

        ax.set_xlabel("Time", fontsize=20, weight='bold')
        ax.set_ylabel("State %02d" % idx_to_plot[0],
                      fontsize=20, weight='bold')
        ax.xaxis.grid(color='white', linewidth=2)
        ax.set_xlim([0., 15.])

    else:
        # ####
        # 2D #
        # ####

        line = ax.plot(mus_prev_to_plot[:, 0],
                       mus_prev_to_plot[:, 1],
                       c='black')[0]
        if itp == 1:
            lines.append(line)
            labels.append('Previous Traj.')
        line = ax.plot(mus_bad_to_plot[:, 0],
                       mus_bad_to_plot[:, 1],
                       c='red')[0]
        if itp == 1:
            lines.append(line)
            labels.append('Bad Traj.')

        line = ax.plot(mus_good_to_plot[:, 0],
                       mus_good_to_plot[:, 1],
                       c='green')[0]
        if itp == 1:
            lines.append(line)
            labels.append('Good Traj.')
        line = ax.plot(mus_to_plot[:, 0],
                       mus_to_plot[:, 1],
                       c='blue')[0]
        if itp == 1:
            lines.append(line)
            labels.append('Updated Traj.')

        if tgt_positions is not None:
            tgt = tgt_positions[cond]
            circle1 = plt.Circle(tgt, 0.02, facecolor='green', alpha=0.6,
                                 edgecolor='black')
            ax.add_artist(circle1)
        if obst_positions is not None:
            if safe_distance is not None:
                obstacle = np.array(obst_positions[cond])
                circle2 = plt.Circle(obstacle, safe_distance, color='black',
                                     alpha=0.1)
                ax.add_artist(circle2)
            obstacle = obst_positions[cond]
            circle2 = plt.Circle(obstacle, 0.05, color='red', alpha=0.2)
            ax.add_artist(circle2)

        ax.set_xlabel("X", fontsize=20, weight='bold')
        ax.set_ylabel("Y",
                      fontsize=20, weight='bold')

        if option == -1:
            ax.set_xlim([-0.0, 1.])
            ax.set_ylim([-1.0, 0.75])
            ax.set_aspect(1.75)
        elif option == -2:
            ax.set_title('Obstacle', fontweight='bold', fontsize=25)
            ax.set_xlim([list_to_substact[cond][0] - 0.2,
                         list_to_substact[cond][0] + 0.2])
            ax.set_ylim([list_to_substact[cond][1] - 0.2,
                         list_to_substact[cond][1] + 0.2])
            ax.set_aspect(1)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            ax.set_title('Target', fontweight='bold', fontsize=25)
            ax.set_xlim([list_to_substact[cond][0] - 0.2,
                         list_to_substact[cond][0] + 0.2])
            ax.set_ylim([list_to_substact[cond][1] - 0.2,
                         list_to_substact[cond][1] + 0.2])
            ax.set_aspect(1)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

    # Background
    ax.set_facecolor((0.917, 0.917, 0.949))

    if option == -1:
        fig.legend(lines, labels, loc='center right', ncol=1)
        plt.savefig("all_plots/2d/itr_%02d_cond_%02d.svg" % (itp, cond))
        plt.savefig("all_plots/itr_%02d_cond_%02d.svg" % (itp, cond))
        plt.savefig("all_plots/2d/itr_%02d_cond_%02d.png" % (itp, cond))
        plt.savefig("all_plots/itr_%02d_cond_%02d.png" % (itp, cond))
    elif option == -2:
        # fig.legend(lines, labels, loc='center right', ncol=1)
        plt.savefig("all_plots/obst/itr_%02d_cond_%02d_obst.svg" % (itp, cond))
        plt.savefig("all_plots/obst/itr_%02d_cond_%02d_obst.png" % (itp, cond))
    elif option == -3:
        # fig.legend(lines, labels, loc='center right', ncol=1)
        plt.savefig("all_plots/tgt/itr_%02d_cond_%02d_tgt.svg" % (itp, cond))
        plt.savefig("all_plots/tgt/itr_%02d_cond_%02d_tgt.png" % (itp, cond))
    else:
        fig.legend(lines, labels, loc='center right', ncol=1, fontsize=15)
        plt.savefig("all_plots/itr_%02d_cond_%02d_state_%02d.svg" % (itp, cond, option))
        plt.savefig("all_plots/itr_%02d_cond_%02d_state_%02d.png" % (itp, cond, option))

plt.show(block=block)

