import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import os, sys


def plot_avg_specific_costs(gps_directory_names, itr_to_load=None,
                            specific_costs=None, gps_models_labels=None,
                            method='gps', block=False, print_info=True,
                            conds_to_combine=None,
                            latex_plot=False, train_conds=None,
                            ):
    if latex_plot:
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        # rc('font', **{'family': 'serif','serif':['Times']})
        matplotlib.rcParams['font.family'] = ['serif']
        matplotlib.rcParams['font.serif'] = ['Times New Roman']

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
            sys.exit(-1)

        if print_info:
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

        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(hspace=0)
        if not latex_plot:
            fig.suptitle("Policy Avg Specific Costs for training conditions"
                         " (over %02d runs)" % max_available_runs,
                         fontsize=30, weight='bold')
        fig.canvas.set_window_title('Policy Avg Specific Cost')
        fig.set_facecolor((1, 1, 1))
        des_colormap = [colormap(i) for i in np.linspace(0, 1, total_gps)]

        lines = list()
        labels = list()

        for gps in range(total_gps):
            total_runs = len(pol_sample_lists_cost_compos[gps])
            total_cond = pol_sample_lists_cost_compos[gps][-1].shape[0]
            total_itr = pol_sample_lists_cost_compos[gps][-1].shape[1]
            total_samples = pol_sample_lists_cost_compos[gps][-1].shape[2]
            T = pol_sample_lists_cost_compos[gps][-1].shape[4]

            avg_costs = np.zeros((total_runs, total_cond, total_itr, total_cost_types))
            sum_costs = np.zeros((total_runs, total_cond, total_itr))
            for rr in range(total_runs):
                # Sum over T
                pol_cost_sum = \
                    pol_sample_lists_cost_compos[gps][rr][:, :, :, :, :].sum(axis=4)

                # Average over samples
                avg_costs[rr, :, :, :] = np.mean(pol_cost_sum, axis=2)

                # Sum the specific costs
                for cc in specific_costs:
                    sum_costs[rr, :, :] += avg_costs[rr, :, :, cc]

            # Average over runs
            runs_avg = np.mean(sum_costs, axis=0)
            # print(runs_avg.shape)
            # print(runs_avg)
            # print("BORRAMEEE")

            # Only over selected conditions
            if conds_to_combine is None:
                total_cost_avg = np.mean(runs_avg, axis=0)
            else:
                total_cost_avg = np.mean(runs_avg[conds_to_combine, :], axis=0)

            label = '%s' % gps_models_labels[gps]
            line = ax.plot(iteration_ids[gps][-1],
                           total_cost_avg,
                           marker=marker, label=label,
                           color=gps_models_colors[gps])[0]
            # ax.fill_between(iteration_ids[gps][-1],
            #                 min_sum_costs,
            #                 max_sum_costs, alpha=0.5,
            #                 color=gps_models_colors[gps], zorder=2)
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            lines.append(line)
            labels.append(gps_models_labels[gps])

        max_lim = 0
        for ll in ax.lines:
            if len(ll.get_xdata()) > max_lim:
                max_lim = len(ll.get_xdata())
        ax.set_xlim(0, max_lim+1)
        ax.set_xticks(range(0, 51, 5))
        #ax.set_xticks(range(min_iteration, max_iteration+1))
        #ax.set_xticks(range(0, 26, 5))

        ax.set_xlabel("Iteration", fontsize=40, weight='bold')
        ax.set_ylabel("Safe-Distance Cost",
                      fontsize=40, weight='bold')
        ax.tick_params(axis='x', labelsize=25)
        ax.tick_params(axis='y', labelsize=25)

        # Background
        # aa.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.grid(color='white', linewidth=2)
        ax.set_facecolor((0.917, 0.917, 0.949))

        if not latex_plot:
            # Legend
            fig.legend(lines, labels, loc='center right', ncol=1)
            # #legend = ax.legend(loc='best', ncol=1, fontsize=20)
            # legend = ax.legend(ncol=1, fontsize=25)
            # #legend = plt.figlegend(lines, labels, loc='center right', ncol=1, labelspacing=0., borderaxespad=1.)
            # #legend.get_frame().set_alpha(0.4)
        else:
            legend = plt.legend(handles=lines, loc=1, fontsize=30)

    else:
        raise NotImplementedError

    plt.show(block=block)
