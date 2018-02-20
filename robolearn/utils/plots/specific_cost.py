import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import os, sys


def plot_specific_cost(gps_directory_names, itr_to_load=None,
                       specific_costs=None, gps_models_labels=None,
                       method='gps', block=False, print_info=True):

    #gps_models_line_styles = [':', '--', '-']
    gps_models_line_styles = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
    gps_models_colors = ['black', 'red', 'saddlebrown', 'green', 'magenta', 'orange', 'blue', 'cadetblue', 'mediumslateblue']
    gps_models_markers = ['^', 's', 'D', '^', 's', 'D', '^', 's', 'D', '^', 's', 'D']

    init_itr = 0
    final_itr = 200
    samples_idx = [-1]  # List of samples / None: all samples
    max_traj_plots = None  # None, plot all
    last_n_iters = None  # 5  # None, plot all iterations

    colormap = plt.cm.rainbow  # nipy_spectral, Set1, Paired, winter
    plot_sample_list_max_min = False
    plot_joint_limits = True
    total_gps = len(gps_directory_names)

    cs_list = [list() for _ in range(total_gps)]
    iteration_ids = [list() for _ in range(total_gps)]
    pol_sample_lists_costs = [list() for _ in range(total_gps)]
    pol_sample_lists_cost_compos = [list() for _ in range(total_gps)]


    # Get the data
    for gps, gps_directory_name in enumerate(gps_directory_names):
        # dir_path = os.path.dirname(os.path.realpath(__file__)) + '/../' + gps_directory_name
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

        cs_list[gps] = [list() for _ in range(max_available_runs)]
        iteration_ids[gps] = [list() for _ in range(max_available_runs)]
        pol_sample_lists_costs[gps] = [list() for _ in range(max_available_runs)]
        pol_sample_lists_cost_compos[gps] = [list() for _ in range(max_available_runs)]

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
            first_pol_cost_comp_data = True
            total_itr = len(itr_list)
            for ii, itr_idx in enumerate(itr_list):
                itr_path = dir_path + str('/run_%02d' % rr) + \
                           str('/itr_%02d/' % itr_idx)

                # Pol Cost Composition
                if method == 'gps':
                    file_to_load = itr_path + 'pol_sample_cost_composition_itr_' + \
                                   str('%02d' % itr_idx)+'.pkl'
                else:
                    file_to_load = itr_path + 'sample_cost_composition_itr_' + \
                                   str('%02d' % itr_idx)+'.pkl'
                if os.path.isfile(file_to_load):
                    if print_info:
                        print('Loading policy sample cost composition from '
                              'iteration %02d' % itr_idx)
                    pol_cost_comp_data = pickle.load(open(file_to_load, 'rb'))
                    iteration_ids[gps][rr].append(itr_idx+1)
                    # pol_sample_lists_cost_compositions[gps][rr].\
                    #     append(pol_cost_comp_data)
                    n_cond = len(pol_cost_comp_data)
                    n_samples = len(pol_cost_comp_data[-1])
                    n_cost_types = len(pol_cost_comp_data[-1][-1])
                    T = pol_cost_comp_data[-1][-1][-1].shape[0]

                    if first_pol_cost_comp_data:
                        pol_cost_compos = np.zeros((n_cond, total_itr, n_samples,
                                                    n_cost_types, T))
                        first_pol_cost_comp_data = False

                    for cc in range(n_cond):
                        for ss in range(n_samples):
                            for ct in range(n_cost_types):
                                pol_cost_compos[cc, ii, ss, ct, :] = \
                                    pol_cost_comp_data[cc][ss][ct]
                else:
                    raise ValueError('PolSampleCostComposition does not exist! | '
                                     'gps[%02d] run[%02d] itr[%02d]'
                                     % (gps, rr, itr_idx))

            pol_sample_lists_cost_compos[gps][rr] = pol_cost_compos
            del pol_cost_comp_data


    plots_type = 'iteration'  # 'iteration' or 'episode'
    include_last_T = False  # Only in iteration
    iteration_to_plot = -1
    colormap = plt.cm.rainbow  # nipy_spectral, Set1, Paired, winter, rainbow

    # TODO: Assuming all hsa the same number of conds
    total_cond = pol_sample_lists_cost_compos[-1][-1].shape[0]
    # TODO: Assuming last has more cost types
    total_cost_types = pol_sample_lists_cost_compos[-1][-1].shape[3]

    if specific_costs is None:
        specific_costs = range(total_cost_types)


    if plots_type.lower() == 'iteration':
        #marker = 'o'
        marker = None

        for cond in range(total_cond):
            fig, ax = plt.subplots(len(specific_costs), 1)
            fig.subplots_adjust(hspace=0)
            fig.suptitle("Policy Specific Costs for condition %02d (over %02d runs)"
                         % (cond, max_available_runs),
                         fontsize=30, weight='bold')
            fig.canvas.set_window_title('Policy Specific Cost Condition %02d '
                                        % cond)
            fig.set_facecolor((1, 1, 1))
            des_colormap = [colormap(i) for i in np.linspace(0, 1, total_gps)]

            lines = list()
            labels = list()

            min_iteration = np.inf
            max_iteration = -np.inf

            for gps in range(total_gps):
                # total_cost_types = pol_sample_lists_cost_compos[gps][-1].shape[3]
                total_runs = len(pol_sample_lists_cost_compos[gps])
                total_itr = pol_sample_lists_cost_compos[gps][-1].shape[1]
                total_samples = pol_sample_lists_cost_compos[gps][-1].shape[2]
                T = pol_sample_lists_cost_compos[gps][-1].shape[4]

                # Composition cost
                avg_costs = np.zeros((total_runs, total_itr, total_cost_types))
                for rr in range(total_runs):
                    pol_cost_sum = pol_sample_lists_cost_compos[gps][rr][cond, :, :, :, :].sum(axis=3)
                    # Average over samples
                    avg_costs[rr, :, :] = np.mean(pol_cost_sum, axis=1)

                mean_cost_types = np.mean(avg_costs, axis=0)
                max_cost_types = np.max(avg_costs, axis=0)
                min_cost_types = np.min(avg_costs, axis=0)
                std_cost_types = np.std(avg_costs, axis=0)

                for cc, cost_idx in enumerate(specific_costs):
                    aa = ax[cc] if isinstance(ax, np.ndarray) else ax
                    label = '%s' % gps_models_labels[gps]
                    line = aa.plot(iteration_ids[gps][rr],
                                   mean_cost_types[:, cost_idx],
                                   marker=marker, label=label,
                                   color=gps_models_colors[gps])[0]
                    aa.fill_between(iteration_ids[gps][rr],
                                    min_cost_types[:, cost_idx],
                                    max_cost_types[:, cost_idx], alpha=0.5,
                                    color=gps_models_colors[gps], zorder=2)
                    aa.xaxis.set_major_locator(MaxNLocator(integer=True))
                    if cc == 0:
                        lines.append(line)
                        labels.append(label)

                    if print_info:
                        print('%'*10)
                    print('gps: %d' % gps)
                    for rr in range(total_runs):
                        print('run %02d - total specific_cost(%02d): %r'
                              % (rr, cc, np.array(mean_cost_types[:, cc]).sum()))
                    if print_info:
                        print('%'*10)

            for cc, cost_idx in enumerate(specific_costs):
                aa = ax[cc] if isinstance(ax, np.ndarray) else ax
                max_lim = 0
                for ll in aa.lines:
                    if len(ll.get_xdata()) > max_lim:
                        max_lim = len(ll.get_xdata())
                aa.set_xlim(0, max_lim+1)
                #ax.set_xticks(range(min_iteration, max_iteration+1))
                #ax.set_xticks(range(0, 26, 5))

                aa.set_xlabel("Iteration", fontsize=30, weight='bold')
                aa.set_ylabel("Average Cost (%02d)" % cost_idx,
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
            # #legend = ax.legend(loc='best', ncol=1, fontsize=20)
            # legend = ax.legend(ncol=1, fontsize=25)
            # #legend = plt.figlegend(lines, labels, loc='center right', ncol=1, labelspacing=0., borderaxespad=1.)
            # #legend.get_frame().set_alpha(0.4)

    else:
        raise NotImplementedError

    plt.show(block=block)
