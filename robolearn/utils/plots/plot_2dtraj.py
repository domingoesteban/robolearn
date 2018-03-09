import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import os, sys


def plot_dual_2dtraj_updates(gps_directory_names, idx_to_plot, itr_to_plot=None,
                           itr_to_load=None,
                           tgt_positions=None,
                           obst_positions=None,
                           gps_models_labels=None,
                           safe_distance=None,
                           method='gps', block=False, print_info=True):

    if itr_to_plot is None:
        itr_to_plot = [-1]

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

            first_itr_data = True
            total_itr = len(itr_list)
            for ii, itr_idx in enumerate(itr_list):
                itr_path = dir_path + str('/run_%02d' % rr) + \
                           str('/itr_%02d/' % itr_idx)
                # Samples costs
                file_to_load = itr_path + 'dualist_trajs_itr_' + \
                                   str('%02d' % itr_idx)+'.pkl'
                if os.path.isfile(file_to_load):
                    if print_info:
                        print('Loading dualist trajectories from iteration %02d'
                              % itr_idx)
                    sample_data = pickle.load(open(file_to_load, 'rb'))
                    iteration_ids[gps].append(itr_idx+1)
                    mus = sample_data['mus']
                    mus_bad = sample_data['mus_bad']
                    mus_good = sample_data['mus_good']

                    sigmas = sample_data['sigmas']
                    sigmas_bad = sample_data['sigmas_bad']
                    sigmas_good = sample_data['sigmas_good']
                    n_cond = mus.shape[0]
                    T = mus.shape[1]
                    data_dim = mus.shape[2]
                    if first_itr_data:
                        all_mus = \
                            np.zeros((max_available_runs, n_cond, total_itr, T,
                                      data_dim))
                        all_mus_bad = \
                            np.zeros((max_available_runs, n_cond, total_itr, T,
                                      data_dim))
                        all_mus_good = \
                            np.zeros((max_available_runs, n_cond, total_itr, T,
                                      data_dim))
                        all_sigmas = \
                            np.zeros((max_available_runs, n_cond, total_itr, T,
                                      data_dim, data_dim))
                        all_sigmas_bad = \
                            np.zeros((max_available_runs, n_cond, total_itr, T,
                                      data_dim, data_dim))
                        all_sigmas_good = \
                            np.zeros((max_available_runs, n_cond, total_itr, T,
                                      data_dim, data_dim))
                        first_itr_data = False

                    all_mus[rr, :, ii, :, :] = mus
                    all_mus_bad[rr, :, ii, :, :] = mus_bad
                    all_mus_good[rr, :, ii, :, :] = mus_good
                    all_sigmas[rr, :, ii, :, :, :] = sigmas
                    all_sigmas_bad[rr, :, ii, :, :, :] = sigmas_bad
                    all_sigmas_good[rr, :, ii, :, :, :] = sigmas_good

                else:
                    raise ValueError('dualist_trajs does not exist! | '
                                     'gps[%02d] run[%02d] itr[%02d]'
                                     % (gps, rr, itr_idx))

            # Clear all the loaded data
            dualist_trajs = dict()
            dualist_trajs['mus'] = all_mus
            dualist_trajs['mus_bad'] = all_mus_bad
            dualist_trajs['mus_good'] = all_mus_good
            dualist_trajs['sigmas'] = all_sigmas
            dualist_trajs['sigmas_bad'] = all_sigmas_bad
            dualist_trajs['sigmas_good'] = all_sigmas_good
            samples_list[gps] = dualist_trajs
            del sample_data

    total_cond = samples_list[-1]['mus'].shape[1]

    n_state = len(idx_to_plot)

    for cond in range(total_cond):

        for itp in itr_to_plot:


            lines = list()
            labels = list()
            for gps in range(total_gps):
                rr = -1
                # fig, ax = plt.subplots(n_state, 1)
                fig, ax = plt.subplots(1, 1)
                fig.subplots_adjust(hspace=0)
                fig.suptitle("Dualist Updates Itr: %02d | \n"
                             "Condition %d (over only ONE run)"
                             % (itp, cond),
                             fontsize=20, weight='bold')
                fig.canvas.set_window_title('Policy Samples Distance '
                                            'Condition %02d | %s'
                                            % (cond, gps_models_labels[gps]))
                fig.set_facecolor((1, 1, 1))
                des_colormap = [colormap(i) for i in np.linspace(0, 1, total_gps)]


                mus = samples_list[gps]['mus'][rr, cond, :, :]
                mus_bad = samples_list[gps]['mus_bad'][rr, cond, :, :]
                mus_good = samples_list[gps]['mus_good'][rr, cond, :, :]

                T = mus.shape[1]
                mus_to_plot = np.zeros((1, T, len(idx_to_plot)))
                mus_prev_to_plot = np.zeros((1, T, len(idx_to_plot)))
                mus_bad_to_plot = np.zeros((1, T, len(idx_to_plot)))
                mus_good_to_plot = np.zeros((1, T, len(idx_to_plot)))

                for ii, index in enumerate(idx_to_plot):
                    mus_to_plot[:, :, ii] = mus[itp, :, index]
                    mus_prev_to_plot[:, :, ii] = mus[itp-1, :, index]
                    mus_bad_to_plot[:, :, ii] = mus_bad[itp, :, index]
                    mus_good_to_plot[:, :, ii] = mus_good[itp, :, index]

                line = ax.plot(mus_prev_to_plot[-1, :, 0],
                               mus_prev_to_plot[-1, :, 1],
                               c='black')[0]
                lines.append(line)
                labels.append('new_traj')
                line = ax.plot(mus_bad_to_plot[-1, :, 0],
                               mus_bad_to_plot[-1, :, 1],
                               c='red')[0]
                lines.append(line)
                labels.append('bad_traj')

                line = ax.plot(mus_good_to_plot[-1, :, 0],
                               mus_good_to_plot[-1, :, 1],
                               c='green')[0]
                lines.append(line)
                labels.append('good_traj')
                line = ax.plot(mus_to_plot[-1, :, 0],
                               mus_to_plot[-1, :, 1],
                               c='blue')[0]
                lines.append(line)
                labels.append('new_traj')

                if tgt_positions is not None:
                    tgt = tgt_positions[cond]
                    circle1 = plt.Circle(tgt, 0.02, facecolor='yellow', alpha=0.2,
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


                # # SIGMAS
                # edges = 100
                # p = np.linspace(0, 2*np.pi, edges)
                # xy_ellipse = np.c_[np.cos(p), np.sin(p)]

                ax.set_xlabel("X", fontsize=10, weight='bold')
                ax.set_ylabel("Y",
                              fontsize=10, weight='bold')
                ax.tick_params(axis='x', labelsize=15)
                ax.tick_params(axis='y', labelsize=15)
                ax.set_xlim([-1., 1.])
                ax.set_ylim([-1.5, 1.])
                # plt.axis('equal')

                # Background
                # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                # ax.xaxis.grid(color='white', linewidth=2)
                ax.set_facecolor((0.917, 0.917, 0.949))

                # # Legend
                # fig.legend(lines, labels, loc='center right', ncol=1)

    plt.show(block=block)

