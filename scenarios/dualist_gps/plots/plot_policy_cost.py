import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import os, sys

method = 'trajopt'  # 'gps' or 'trajopt'
gps_directory_names = ['gps_log1']#, 'gps_log2', 'gps_log3', 'gps_log4']
gps_directory_names = ['trajopt_log1']#, 'gps_log2', 'gps_log3', 'gps_log4']
gps_models_labels = ['gps1']#, 'gps2', 'gps3', 'gps4']

#gps_models_line_styles = [':', '--', '-']
gps_models_line_styles = ['-', '-', '-', '-', '-']
gps_models_colors = ['black', 'magenta', 'orange', 'blue', 'green']
gps_models_markers = ['^', 's', 'D', '^', 's', 'D']

init_itr = 0
final_itr = 200
samples_idx = [-1]  # List of samples / None: all samples
max_traj_plots = None  # None, plot all
last_n_iters = None  # 5  # None, plot all iterations
itr_to_load = list(range(5))
# itr_to_load = [0, 4, 8]

plot_cs = True
plot_policy_costs = True
plot_cost_types = True
# if method == 'gps':
#     plot_policy_costs = True
#     plot_cost_types = True
# else:
#     plot_policy_costs = False
#     plot_cost_types = False

colormap = plt.cm.rainbow  # nipy_spectral, Set1, Paired, winter
plot_sample_list_max_min = False
plot_joint_limits = True
gps_num = 0
total_gps = len(gps_directory_names)

cs_list = [list() for _ in range(total_gps)]
iteration_ids = [list() for _ in range(total_gps)]
pol_sample_lists_costs = [list() for _ in range(total_gps)]
pol_sample_lists_cost_compos = [list() for _ in range(total_gps)]


# Get the data
for gps, gps_directory_name in enumerate(gps_directory_names):
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/../' + gps_directory_name

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
                itr_to_load = list(range(init_itr, max_available_itr+1))

        print("Desired iterations to load in %s: %s" % (gps_directory_name,
                                                        itr_to_load))

        first_itr_data = True
        first_pol_cost_data = True
        first_pol_cost_comp_data = True
        total_itr = len(itr_to_load)
        for ii, itr_idx in enumerate(itr_to_load):
            itr_path = dir_path + str('/run_%02d' % rr) + \
                       str('/itr_%02d/' % itr_idx)
            # Samples costs
            if plot_cs:
                file_to_load = itr_path + 'iteration_data_itr_' + \
                               str('%02d' % itr_idx)+'.pkl'
                if os.path.isfile(file_to_load):
                    print('Loading GPS iteration_data from iteration %02d'
                          % itr_idx)
                    iter_data = pickle.load(open(file_to_load, 'rb'))
                    iteration_ids[gps][rr].append(itr_idx+1)
                    n_cond = len(iter_data)
                    n_samples, T = iter_data[-1].cs.shape
                    if first_itr_data:
                        sample_costs = np.zeros((n_cond, total_itr, n_samples, T))
                        first_itr_data = False

                    for cc in range(n_cond):
                        sample_costs[cc, ii, :, :] = iter_data[cc].cs
                else:
                    raise ValueError('ItrData does not exist! | gps[%02d] run[%02d]'
                                     ' itr[%02d]' % (gps, rr, itr_idx))

            if plot_policy_costs:
                # Pol Cost
                if method == 'gps':
                    file_to_load = itr_path + 'pol_sample_cost_itr_' + \
                                   str('%02d' % itr_idx)+'.pkl'
                else:
                    file_to_load = itr_path + 'sample_cost_itr_' + \
                                   str('%02d' % itr_idx)+'.pkl'

                if os.path.isfile(file_to_load):
                    print('Loading policy sample cost from iteration %02d'
                          % itr_idx)
                    pol_cost_data = pickle.load(open(file_to_load, 'rb'))
                    # pol_sample_lists_costs[gps][rr].append(pol_cost_data)
                    n_cond = len(pol_cost_data)
                    n_samples, T = pol_cost_data[-1].shape
                    if first_pol_cost_data:
                        pol_costs = np.zeros((n_cond, total_itr, n_samples, T))
                        first_pol_cost_data = False

                    for cc in range(n_cond):
                        pol_costs[cc, ii, :, :] = pol_cost_data[cc]

                else:
                    raise ValueError('PolSampleCost does not exist! | gps[%02d] '
                                     'run[%02d] itr[%02d]' % (gps, rr, itr_idx))

            if plot_cost_types:
                # Pol Cost Composition
                if method == 'gps':
                    file_to_load = itr_path + 'pol_sample_cost_composition_itr_' + \
                                   str('%02d' % itr_idx)+'.pkl'
                else:
                    file_to_load = itr_path + 'sample_cost_composition_itr_' + \
                                   str('%02d' % itr_idx)+'.pkl'
                if os.path.isfile(file_to_load):
                    print('Loading policy sample cost composition from '
                          'iteration %02d' % itr_idx)
                    pol_cost_comp_data = pickle.load(open(file_to_load, 'rb'))
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


        # Clear all the loaded data
        if plot_cs:
            cs_list[gps][rr] = sample_costs
            del iter_data
        if plot_policy_costs:
            pol_sample_lists_costs[gps][rr] = pol_costs
            del pol_cost_data
        if plot_cost_types:
            pol_sample_lists_cost_compos[gps][rr] = pol_cost_compos
            del pol_cost_comp_data


if plot_cs:
    total_runs = len(cs_list[-1])
    total_cond = cs_list[-1][-1].shape[0]
    total_itr = cs_list[-1][-1].shape[1]
    total_samples = cs_list[-1][-1].shape[2]
    T = cs_list[-1][-1].shape[3]

    for cond in range(total_cond):
        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('Trajectories Costs Condition %02d '
                                    '(over %02d runs)'
                                    % (cond, total_runs))
        fig.set_facecolor((1, 1, 1))
        des_colormap = [colormap(i) for i in np.linspace(0, 1, total_gps)]

        lines = list()
        labels = list()

        for gps in range(total_gps):
            print('&&&'*5)
            print('&&&'*5)
            print('TODO: WE ARE USING ONLY ONE RUN')
            print('&&&'*5)
            rr = -1

            samples_cost_sum = cs_list[gps][rr][cond, :, :, :].sum(axis=-1)
            mean_costs = np.mean(samples_cost_sum, axis=-1)
            max_costs = np.max(samples_cost_sum, axis=-1)
            min_costs = np.min(samples_cost_sum, axis=-1)
            std_costs = np.std(samples_cost_sum, axis=-1)

            line = ax.plot(iteration_ids[gps][rr], mean_costs,
                           marker=gps_models_markers[gps],
                           label=gps_models_labels[gps],
                           linestyle=gps_models_line_styles[gps],
                           color=gps_models_colors[gps])[0]

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.fill_between(iteration_ids[gps][rr], min_costs, max_costs,
                            alpha=0.5, color=gps_models_colors[gps])
            lines.append(line)
            labels.append(gps_models_labels[gps])


if plot_policy_costs:
    plots_type = 'iteration'  # 'iteration' or 'episode'
    include_last_T = False  # Only in iteration
    iteration_to_plot = -1
    colormap = plt.cm.rainbow  # nipy_spectral, Set1, Paired, winter, rainbow

    total_runs = len(pol_sample_lists_costs[-1])
    total_cond = pol_sample_lists_costs[-1][-1].shape[0]
    total_itr = pol_sample_lists_costs[-1][-1].shape[1]
    total_samples = pol_sample_lists_costs[-1][-1].shape[2]
    T = pol_sample_lists_costs[-1][-1].shape[3]
    total_cost_types = pol_sample_lists_cost_compos[-1][-1].shape[3]

    if plots_type.lower() == 'iteration':
        #marker = 'o'
        marker = None
        for cond in range(total_cond):
            fig, ax = plt.subplots(1, 1)
            fig.subplots_adjust(hspace=0)
            fig.canvas.set_window_title('Policy Costs Condition %02d '
                                        '(over %02d runs)'
                                        % (cond, max_available_runs))
            fig.set_facecolor((1, 1, 1))
            des_colormap = [colormap(i) for i in np.linspace(0, 1, total_gps)]

            lines = list()
            labels = list()

            min_iteration = np.inf
            max_iteration = -np.inf

            for gps in range(total_gps):

                avg_costs = np.zeros((total_runs, total_itr))

                for rr in range(total_runs):
                    pol_cost_sum = pol_sample_lists_costs[gps][rr][cond, :, :, :].sum(axis=-1)
                    # Average over samples
                    avg_costs[rr, :] = np.mean(pol_cost_sum, axis=-1)

                mean_costs = np.mean(avg_costs, axis=0)
                max_costs = np.max(avg_costs, axis=0)
                min_costs = np.min(avg_costs, axis=0)
                std_costs = np.std(avg_costs, axis=0)

                line = ax.plot(iteration_ids[gps][rr], mean_costs,
                               marker=gps_models_markers[gps],
                               label=gps_models_labels[gps],
                               linestyle=gps_models_line_styles[gps],
                               color=gps_models_colors[gps])[0]
                ax.fill_between(iteration_ids[gps][rr], min_costs, max_costs,
                                alpha=0.5, color=gps_models_colors[gps])
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                lines.append(line)
                labels.append(gps_models_labels[gps])

                # Composition cost
                if plot_cost_types:
                    avg_costs = np.zeros((total_runs, total_itr, total_cost_types))
                    for rr in range(total_runs):
                        pol_cost_sum = pol_sample_lists_cost_compos[gps][rr][cond, :, :, :, :].sum(axis=3)
                        # Average over samples
                        avg_costs[rr, :, :] = np.mean(pol_cost_sum, axis=1)

                    mean_cost_types = np.mean(avg_costs, axis=0)
                    max_cost_types = np.max(avg_costs, axis=0)
                    min_cost_types = np.min(avg_costs, axis=0)
                    std_cost_types = np.std(avg_costs, axis=0)

                    for c in range(total_cost_types):
                        label = '%s-cost%d' % (gps_models_labels[gps], c)
                        line = ax.plot(iteration_ids[gps][rr],
                                       mean_cost_types[:, c],
                                       marker=marker, label=label)[0]
                        ax.fill_between(iteration_ids[gps][rr],
                                        min_cost_types[:, c],
                                        max_cost_types[:, c], alpha=0.5)
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        lines.append(line)
                        labels.append(label)

                print('%'*10)
                print('gps: %d' % gps)
                for rr in range(total_runs):
                    best_index = [iteration_ids[gps][rr][ii]
                                  for ii in np.argsort(mean_costs)]
                    print('run %02d - best_index: %r' % (rr, best_index))
                print('%'*10)

            #ax.set_xticks(range(min_iteration, max_iteration+1))
            #ax.set_xlim(min_iteration, max_iteration)
            #ax.set_xticks(range(0, 26, 5))
            #ax.set_xlim(0, 25)
            ax.set_xlabel("Iterations", fontsize=30, weight='bold')
            ax.set_ylabel("Average Cost", fontsize=30, weight='bold')
            ax.tick_params(axis='x', labelsize=25)
            ax.tick_params(axis='y', labelsize=25)

            #legend = ax.legend(loc='best', ncol=1, fontsize=20)
            legend = ax.legend(ncol=1, fontsize=25)
            #legend = plt.figlegend(lines, labels, loc='center right', ncol=1, labelspacing=0., borderaxespad=1.)
            #legend.get_frame().set_alpha(0.4)

    else:
        raise NotImplementedError
        # T = pol_sample_lists_costs[0][0].shape[1]
        # if include_last_T is False:
        #     T = T - 1
        #
        # if iteration_to_plot is not None:
        #     if iteration_to_plot == -1:
        #         iteration_to_plot = total_itr - 1
        #     itr_to_plot = [iteration_to_plot]
        # else:
        #     itr_to_plot = range(total_itr)
        #
        # for cond in range(total_cond):
        #     lines = list()
        #     labels = list()
        #
        #     total_cost_types = len(pol_sample_lists_cost_compositions[-1][-1][-1])
        #
        #     fig, ax = plt.subplots(1, 1)
        #     fig.canvas.set_window_title('Policy Costs | Condition %d' % cond)
        #     fig.set_facecolor((1, 1, 1))
        #     if plot_cost_types:
        #         colormap_list = [colormap(i) for i in np.linspace(0, 1, (len(itr_to_plot)*total_cost_types)+1)]
        #     else:
        #         colormap_list = [colormap(i) for i in np.linspace(0, 1, len(itr_to_plot))]
        #     ax.set_prop_cycle('color', colormap_list)
        #     ax.set_title('Policy Costs | Condition %d' % cond)
        #
        #     mean_costs = np.zeros([total_itr, T])
        #     max_costs = np.zeros([total_itr, T])
        #     min_costs = np.zeros([total_itr, T])
        #     std_costs = np.zeros([total_itr, T])
        #     for itr in itr_to_plot:
        #         total_samples = len(pol_sample_lists_costs[itr][cond])
        #         samples_cost = pol_sample_lists_costs[itr][cond][:, :T]
        #         mean_costs[itr, :] = samples_cost.mean(axis=0)
        #         max_costs[itr, :] = samples_cost.max(axis=0)
        #         min_costs[itr, :] = samples_cost.min(axis=0)
        #         std_costs[itr, :] = samples_cost.std(axis=0)
        #         label = 'Total Cost (itr%d)' % itr
        #         line = ax.plot(mean_costs[itr, :], label=label)[0]
        #         ax.fill_between(range(T), min_costs[itr, :], max_costs[itr, :], alpha=0.5)
        #         ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        #         lines.append(line)
        #         labels.append(label)
        #
        #     # Composition cost
        #     if plot_cost_types:
        #         mean_cost_types = np.zeros([total_itr, total_cost_types, T])
        #         max_cost_types = np.zeros([total_itr, total_cost_types, T])
        #         min_cost_types = np.zeros([total_itr, total_cost_types, T])
        #         std_cost_types = np.zeros([total_itr, total_cost_types, T])
        #         for itr in itr_to_plot:
        #             total_samples = len(pol_sample_lists_cost_compositions[itr][cond])
        #             for c in range(total_cost_types):
        #                 cost_type = np.zeros([total_samples, T])
        #                 for n in range(total_samples):
        #                     cost_type[n, :] = pol_sample_lists_cost_compositions[itr][cond][n][c][:T]
        #                 mean_cost_types[itr, c, :] = cost_type.mean(axis=0)
        #                 max_cost_types[itr, c, :] = cost_type.max(axis=0)
        #                 min_cost_types[itr, c, :] = cost_type.min(axis=0)
        #                 std_cost_types[itr, c, :] = cost_type.std(axis=0)
        #
        #         for c in range(total_cost_types):
        #             label = 'Cost type %d (itr%d)' % (c, itr)
        #             line = ax.plot(mean_cost_types[itr, c, :], label=label)[0]
        #             ax.fill_between(range(T), min_cost_types[itr, c, :], max_cost_types[itr, c, :], alpha=0.5)
        #             ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        #
        #             lines.append(line)
        #             labels.append(label)
        #
        #     plt.xlabel('Time')
        #     plt.ylabel('Cost')
        #     ax.set_xlim([0, T])
        #     ax.set_xticks(np.arange(0, T+2, 50))
        #
        #     legend = plt.figlegend(lines, labels, loc='center right', ncol=1, labelspacing=0., borderaxespad=1.)
        #     legend.get_frame().set_alpha(0.4)

plt.show(block=False)

input('Showing plots. Press a key to close...')
