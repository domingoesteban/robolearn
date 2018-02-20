import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import os, sys


def plot_policy_final_distance(gps_directory_names, states_tuples,
                               itr_to_load=None, gps_models_labels=None,
                               method='gps', block=False, print_info=True):

    if gps_models_labels is None:
        gps_models_labels = gps_directory_names

    gps_models_line_styles = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
    gps_models_colors = ['black', 'red', 'saddlebrown', 'green', 'magenta', 'orange', 'blue', 'cadetblue', 'mediumslateblue']
    gps_models_markers = ['^', 's', 'D', '^', 's', 'D', '^', 's', 'D', '^', 's', 'D']

    init_itr = 0
    final_itr = 200
    max_traj_plots = None  # None, plot all
    last_n_iters = None  # 5  # None, plot all iterations

    colormap = plt.cm.rainbow  # nipy_spectral, Set1, Paired, winter
    total_gps = len(gps_directory_names)

    samples_list = [list() for _ in range(total_gps)]
    iteration_ids = [list() for _ in range(total_gps)]

    # Get the data
    for gps, gps_directory_name in enumerate(gps_directory_names):
        dir_path = gps_directory_name

        max_available_runs = 0
        for rr in range(100):
            if os.path.exists(dir_path + ('/run_%02d' % rr)):
                max_available_runs += 1

        if max_available_runs == 0:
            print("There is not any runs data. Is the path '%s' correct?"
                  % dir_path)
            exit(-1)

        print("Max available runs: %d in file %s"
              % (max_available_runs, gps_directory_name))

        for rr in range(max_available_runs):

            max_available_itr = 0
            for pp in range(init_itr, final_itr):
                if os.path.exists(dir_path + ('/run_%02d' % rr)
                                      + ('/itr_%02d' % pp)):
                    max_available_itr += 1

            if max_available_itr == 0:
                print("There is not any iteration data. Is the path '%s' correct?"
                      % dir_path)
                exit(-1)

            print("Max available iterations: %d in file %s/run_%02d"
                  % (max_available_itr, gps_directory_name, rr))

            if itr_to_load is None:
                if last_n_iters is not None:
                    init_itr = max(max_available_itr - last_n_iters + 1, 0)

                if max_traj_plots is not None:
                    if max_available_itr > max_traj_plots:
                        itr_to_load = np.linspace(init_itr, max_available_itr,
                                                  max_traj_plots, dtype=np.uint8)
                    else:
                        itr_to_load = list(range(init_itr, max_available_itr+1))

                else:
                    itr_list = list(range(init_itr, max_available_itr))
            else:
                itr_list = itr_to_load

            print("Desired iterations to load in %s: %s" % (gps_directory_name,
                                                            itr_list))

            first_itr_data = True
            total_itr = len(itr_list)
            for ii, itr_idx in enumerate(itr_list):
                itr_path = dir_path + str('/run_%02d' % rr) + \
                           str('/itr_%02d/' % itr_idx)
                # Samples costs
                if method == 'gps':
                    file_to_load = itr_path + 'pol_sample_itr_' + \
                                   str('%02d' % itr_idx)+'.pkl'
                else:
                    file_to_load = itr_path + 'traj_sample_itr_' + \
                                   str('%02d' % itr_idx)+'.pkl'
                if os.path.isfile(file_to_load):
                    if print_info:
                        print('Loading policy sample from iteration %02d'
                              % itr_idx)
                    sample_data = pickle.load(open(file_to_load, 'rb'))
                    iteration_ids[gps].append(itr_idx+1)
                    n_cond = len(sample_data)
                    n_samples, T, dX = sample_data[-1].get_states().shape
                    if first_itr_data:
                        samples = \
                            np.zeros((max_available_runs, n_cond, total_itr,
                                      n_samples, T, dX))
                        first_itr_data = False

                    for nn in range(n_cond):
                        samples[rr, nn, ii, :, :] = sample_data[nn].get_states()

                else:
                    raise ValueError('ItrData does not exist! | gps[%02d] run[%02d]'
                                     ' itr[%02d]' % (gps, rr, itr_idx))

            # Clear all the loaded data
            samples_list[gps] = samples
            del sample_data

    total_runs = samples_list[-1].shape[0]
    total_cond = samples_list[-1].shape[1]
    total_itr = samples_list[-1].shape[2]
    total_samples = samples_list[-1].shape[3]
    T = samples_list[-1].shape[4]
    dX = samples_list[-1].shape[5]

    n_state = len(states_tuples)

    for cond in range(total_cond):
        fig, ax = plt.subplots(n_state, 1)
        fig.subplots_adjust(hspace=0)
        fig.suptitle("Policy Samples Distance | Condition %d (over %02d runs)"
                     % (cond, max_available_runs),
                     fontsize=30, weight='bold')
        fig.canvas.set_window_title('Policy Samples Distance Condition %02d'
                                    % cond)
        fig.set_facecolor((1, 1, 1))
        des_colormap = [colormap(i) for i in np.linspace(0, 1, total_gps)]

        lines = list()
        labels = list()
        for gps in range(total_gps):
            for nn, (ee, tt) in enumerate(states_tuples):
                ee_data = samples_list[gps][:, cond, :, :, -1, ee]
                avg_ee = np.mean(ee_data, axis=2)  # Mean over policy samples
                # Data over runs
                mean_ee = np.max(avg_ee, axis=0)
                max_ee = np.max(avg_ee, axis=0)
                min_ee = np.min(avg_ee, axis=0)
                std_ee = np.std(avg_ee, axis=0)

                tgt_data = samples_list[gps][:, cond, :, :, -1, tt]
                avg_tgt = np.mean(tgt_data, axis=2)  # Mean over policy samples
                # Data over runs
                mean_tgt = np.mean(avg_tgt, axis=0)
                max_tgt = np.max(avg_tgt, axis=0)
                min_tgt = np.min(avg_tgt, axis=0)
                std_tgt = np.std(avg_tgt, axis=0)

                dist_data = ee_data - tgt_data
                avg_dist = np.mean(dist_data, axis=2)  # Mean over policy samples
                # Data over runs
                mean_dist = np.mean(avg_dist, axis=0)
                max_dist = np.max(avg_dist, axis=0)
                min_dist = np.min(avg_dist, axis=0)
                std_dist = np.std(avg_dist, axis=0)

                aa = ax[nn] if isinstance(ax, np.ndarray) else ax
                label = '%s' % gps_models_labels[gps]
                line = aa.plot(iteration_ids[gps], mean_ee,
                               marker=gps_models_markers[gps],
                               label=gps_models_labels[gps],
                               linestyle=gps_models_line_styles[gps],
                               color=gps_models_colors[gps])[0]

                aa.fill_between(iteration_ids[gps], min_ee,
                                max_ee, alpha=0.5,
                                color=gps_models_colors[gps], zorder=2)
                if nn == 0:
                    lines.append(line)
                    labels.append(gps_models_labels[gps])

                fix_line = aa.plot(iteration_ids[gps], mean_tgt,
                                   marker=None,
                                   label='Tgt',
                                   linestyle=':',
                                   color='black')[0]

        for nn in range(n_state):
            aa = ax[nn] if isinstance(ax, np.ndarray) else ax
            max_lim = 0
            for ll in aa.lines:
                if len(ll.get_xdata()) > max_lim:
                    max_lim = len(ll.get_xdata())
            aa.set_xlim(0, max_lim+1)
            #ax.set_xticks(range(min_iteration, max_iteration+1))
            #ax.set_xticks(range(0, 26, 5))

            aa.set_xlabel("Iteration", fontsize=30, weight='bold')
            aa.set_ylabel("EE Distance (%02d)" % nn,
                          fontsize=10, weight='bold')
            aa.tick_params(axis='x', labelsize=15)
            aa.tick_params(axis='y', labelsize=15)

            # Background
            aa.xaxis.set_major_locator(MaxNLocator(integer=True))
            aa.xaxis.grid(color='white', linewidth=2)
            aa.set_facecolor((0.917, 0.917, 0.949))

        if isinstance(ax, np.ndarray):
            for aa in ax[:-1]:
                aa.xaxis.set_ticklabels([])

        # Legend
        fig.legend(lines, labels, loc='center right', ncol=1)

    plt.show(block=block)

