import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import os, sys


def plot_policy_final_distance_new(gps_directory_names, states_idxs,
                                   itr_to_load=None, gps_models_labels=None,
                                   method='gps', per_element=True, block=False,
                                   conds_to_combine=None,
                                   latex_plot=False,
                                   print_info=True,
                                   tolerance=0.1,
                                   plot_title=None):
    if latex_plot:
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        # rc('font', **{'family': 'serif','serif':['Times']})
        matplotlib.rcParams['font.family'] = ['serif']
        matplotlib.rcParams['font.serif'] = ['Times New Roman']

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
            sys.exit(-1)

        if print_info:
            print("Max available runs: %d in file %s"
                  % (max_available_runs, gps_directory_name))

        first_itr_data = True
        for rr in range(max_available_runs):

            max_available_itr = 0
            for pp in range(init_itr, final_itr):
                if os.path.exists(dir_path + ('/run_%02d' % rr)
                                      + ('/itr_%02d' % pp)):
                    max_available_itr += 1

            if max_available_itr == 0:
                print("There is not any iteration data. Is the path '%s' correct?"
                      % dir_path)
                sys.exit(-1)

            if print_info:
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

            if print_info:
                print("Desired iterations to load in %s: %s"
                      % (gps_directory_name, itr_list))

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

                    if rr == 0:
                        iteration_ids[gps].append(itr_idx+1)

                    n_cond = len(sample_data)
                    n_samples, T, dX = sample_data[-1].get_states().shape
                    if first_itr_data:
                        samples = \
                            np.ones((max_available_runs, n_cond, total_itr,
                                      n_samples, T, dX))*-10000
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

    n_state = len(states_idxs)

    if conds_to_combine is None:
        for cond in range(total_cond):

            if per_element:
                fig, ax = plt.subplots(n_state, 1)
                fig.subplots_adjust(hspace=0)
            else:
                fig, ax = plt.subplots(1, 1)
                fig.subplots_adjust(hspace=0)

            if not latex_plot:
                fig.suptitle("Policy Samples Final Distance | Condition %d (over %02d runs)"
                             % (cond, max_available_runs),
                             fontsize=30, weight='bold')
            fig.canvas.set_window_title('Policy Samples Distance Condition %02d'
                                        % cond)
            fig.set_facecolor((1, 1, 1))
            des_colormap = [colormap(i) for i in np.linspace(0, 1, total_gps)]

            lines = list()
            labels = list()
            for gps in range(total_gps):
                if per_element:
                    for nn, dd in enumerate(states_idxs):
                        dist_data = samples_list[gps][:, cond, :, :, -1, dd]
                        avg_dist = np.mean(dist_data, axis=2)  # Mean over policy samples
                        # Data over runs
                        mean_dist = np.mean(avg_dist, axis=0)
                        max_dist = np.max(avg_dist, axis=0)
                        min_dist = np.min(avg_dist, axis=0)
                        std_dist = np.std(avg_dist, axis=0)

                        aa = ax[nn] if isinstance(ax, np.ndarray) else ax
                        label = '%s' % gps_models_labels[gps]
                        line = aa.plot(iteration_ids[gps], mean_dist,
                                       marker=gps_models_markers[gps],
                                       label=gps_models_labels[gps],
                                       linestyle=gps_models_line_styles[gps],
                                       color=gps_models_colors[gps])[0]

                        aa.fill_between(iteration_ids[gps], min_dist,
                                        max_dist, alpha=0.5,
                                        color=gps_models_colors[gps], zorder=2)
                        if nn == 0:
                            lines.append(line)
                            labels.append(gps_models_labels[gps])

                        fix_line = aa.plot(iteration_ids[gps],
                                           tolerance*np.ones_like(iteration_ids[gps]),
                                           marker=None,
                                           label='Tgt',
                                           linestyle=':',
                                           color='black')[0]

                # NORM DISTANCE
                else:
                    # TODO: ASSUMING X AND Y IS PROVIDED
                    idxs_dist = slice(states_idxs[0], states_idxs[1]+1)

                    dist_data = samples_list[gps][:, cond, :, :, -1, idxs_dist]  # Get last state data
                    dist_norm = np.linalg.norm(dist_data, axis=3)  # Norm over states
                    avg_dist = np.mean(dist_norm, axis=2)  # Mean over policy samples
                    # Data over runs
                    mean_dist = np.mean(avg_dist, axis=0)
                    std_dist = np.std(avg_dist, axis=0)
                    # max_dist = np.max(avg_dist, axis=0)
                    # min_dist = np.min(avg_dist, axis=0)
                    max_dist = mean_dist + std_dist
                    min_dist = mean_dist - std_dist

                    label = '%s' % gps_models_labels[gps]
                    line = ax.plot(iteration_ids[gps], mean_dist,
                                   marker=gps_models_markers[gps],
                                   label=gps_models_labels[gps],
                                   linestyle=gps_models_line_styles[gps],
                                   color=gps_models_colors[gps])[0]

                    ax.fill_between(iteration_ids[gps], min_dist,
                                    max_dist, alpha=0.5,
                                    color=gps_models_colors[gps], zorder=2)

                    fix_line = ax.plot(iteration_ids[gps],
                                       tolerance*np.ones_like(iteration_ids[gps]),
                                       marker=None,
                                       label='Tgt',
                                       linestyle=':',
                                       color='black')[0]

                    lines.append(line)
                    labels.append(gps_models_labels[gps])

            if not latex_plot:
                for nn in range(n_state):
                    aa = ax[nn] if isinstance(ax, np.ndarray) else ax
                    max_lim = 0
                    for ll in aa.lines:
                        if len(ll.get_xdata()) > max_lim:
                            max_lim = len(ll.get_xdata())
                    aa.set_xlim(0, max_lim+1)
                    aa.set_xticks(range(0, max_lim+1, 5))
                    #ax.set_xticks(range(0, 26, 5))

                    aa.set_xlabel("Iteration", fontsize=40, weight='bold')
                    aa.set_ylabel("EE Distance (%02d)" % nn,
                                  fontsize=40, weight='bold')
                    aa.tick_params(axis='x', labelsize=25)
                    aa.tick_params(axis='y', labelsize=25)

                    # Background
                    aa.xaxis.set_major_locator(MaxNLocator(integer=True))
                    aa.xaxis.grid(color='white', linewidth=2)
                    aa.set_facecolor((0.917, 0.917, 0.949))

                if isinstance(ax, np.ndarray):
                    for aa in ax[:-1]:
                        aa.xaxis.set_ticklabels([])
            else:
                max_lim = 0
                for ll in ax.lines:
                    if len(ll.get_xdata()) > max_lim:
                        max_lim = len(ll.get_xdata())
                ax.set_xlim(0, max_lim+1)
                # ax.set_xticks(range(0, max_lim+1, 5))
                #ax.set_xticks(range(0, 26, 5))

                ax.set_xlabel("Iteration", fontsize=40, weight='bold')
                ax.set_ylabel("Distance to target", fontsize=40, weight='bold')
                ax.tick_params(axis='x', labelsize=25)
                ax.tick_params(axis='y', labelsize=25)

                # Background
                # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xticks(range(0, 51, 5))
                ax.xaxis.grid(color='white', linewidth=2)
                ax.set_facecolor((0.917, 0.917, 0.949))

            if not latex_plot:
                # Legend
                fig.legend(lines, labels, loc='center right', ncol=1)
            else:
                legend = plt.legend(handles=lines, loc=1, fontsize=30)

    else:
        if per_element:
            raise NotImplementedError('Not implemented for per_element option')

        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(hspace=0)

        if not latex_plot:
            fig.suptitle("Policy Samples Final Distance (over %02d runs)"
                         % (max_available_runs),
                         fontsize=30, weight='bold')
        fig.canvas.set_window_title('Policy Samples Distance')
        fig.set_facecolor((1, 1, 1))
        des_colormap = [colormap(i) for i in np.linspace(0, 1, total_gps)]

        lines = list()
        labels = list()
        for gps in range(total_gps):
            # TODO: ASSUMING X AND Y IS PROVIDED
            idxs_dist = slice(states_idxs[0], states_idxs[1]+1)

            dist_data = samples_list[gps][:, :, :, :, -1, idxs_dist]  # Get idxs at final T
            print(dist_data.shape)

            dist_norm = np.linalg.norm(dist_data, axis=4)  # Norm over states
            avg_dist = np.mean(dist_norm, axis=3)  # Mean over policy samples
            print(avg_dist.shape)
            # Data over runs
            mean_dist = np.mean(avg_dist, axis=0)
            max_dist = np.max(avg_dist, axis=0)
            min_dist = np.min(avg_dist, axis=0)
            std_dist = np.std(avg_dist, axis=0)

            # Get only the desired conditions
            print('Generating costs over', conds_to_combine)
            print(mean_dist.shape)
            conds_mean_dist = mean_dist[conds_to_combine, :].mean(axis=0)
            conds_max_dist = max_dist[conds_to_combine, :].mean(axis=0)
            conds_min_dist = min_dist[conds_to_combine, :].mean(axis=0)
            conds_std_dist = std_dist[conds_to_combine, :].mean(axis=0)

            line = ax.plot(iteration_ids[gps], conds_mean_dist,
                           marker=gps_models_markers[gps],
                           label=gps_models_labels[gps],
                           linestyle=gps_models_line_styles[gps],
                           color=gps_models_colors[gps])[0]

            ax.fill_between(iteration_ids[gps], conds_min_dist,
                            conds_max_dist, alpha=0.5,
                            color=gps_models_colors[gps], zorder=2)

            fix_line = ax.plot(iteration_ids[gps],
                               tolerance*np.ones_like(iteration_ids[gps]),
                               marker=None,
                               label='Tgt',
                               linestyle=':',
                               color='black')[0]

            lines.append(line)
            labels.append(gps_models_labels[gps])


        max_lim = 0
        for ll in ax.lines:
            if len(ll.get_xdata()) > max_lim:
                max_lim = len(ll.get_xdata())
        ax.set_xlim(0, max_lim+1)
        # ax.set_xticks(range(0, max_lim+1, 5))
        #ax.set_xticks(range(0, 26, 5))

        ax.set_xlabel("Iteration", fontsize=40, weight='bold')
        ax.set_ylabel("Distance to target", fontsize=40, weight='bold')
        ax.tick_params(axis='x', labelsize=25)
        ax.tick_params(axis='y', labelsize=25)

        if plot_title is not None:
            ax.set_title(plot_title, fontweight='bold', fontsize=25)

        # Background
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(range(0, 51, 5))
        ax.xaxis.grid(color='white', linewidth=2)
        ax.set_facecolor((0.917, 0.917, 0.949))

        legend = plt.legend(handles=lines, loc=1, fontsize=30)

    plt.show(block=block)

