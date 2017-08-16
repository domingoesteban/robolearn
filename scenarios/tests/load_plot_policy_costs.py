import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle

#gps_directory_name = 'GPS_2017-08-04_20:32:12'  # l1: 1.0, l2: 1.0e-3
#gps_directory_name = 'GPS_2017-08-07_16:05:32'  # l1: 1.0, l2: 0.0
#gps_directory_name = 'GPS_2017-08-07_19:35:58'  # l1: 1.0, l2: 1.0
#gps_directory_name = 'GPS_2017-08-09_14:11:15'  # 2 arms
gps_directory_name = 'GPS_2017-08-14_10:35:40'  # dummy test
gps_directory_name = 'GPS_2017-08-15_17:26:51'

init_pol_sample_itr = 0
final_pol_sample_itr = 100
#plots_type = 'iteration'  # 'iteration' or 'episode'
plots_type = 'episode'  # 'iteration' or 'episode'
include_last_T = False  # Only in iteration
iteration_to_plot = -1
plot_cost_types = True
colormap = plt.cm.rainbow  # nipy_spectral, Set1, Paired, winter, rainbow

gps_path = '/home/desteban/workspace/robolearn/scenarios/robolearn_log/' + gps_directory_name

pol_sample_lists_costs = list()
pol_sample_lists_cost_compositions = list()
print('Loading data from %s directory name.' % gps_directory_name)
for pp in range(init_pol_sample_itr, final_pol_sample_itr):
    if os.path.isfile(gps_path+'/pol_sample_cost_itr_'+str('%02d' % pp)+'.pkl'):
        print('Loading policy sample cost from iteration %d' % pp)
        pol_sample_lists_costs.append(pickle.load(open(gps_path+'/pol_sample_cost_itr_'+str('%02d' % pp)+'.pkl', 'rb')))
    if os.path.isfile(gps_path+'/pol_sample_cost_composition_itr_'+str('%02d' % pp)+'.pkl'):
        print('Loading policy sample cost composition from iteration %d' % pp)
        pol_sample_lists_cost_compositions.append(pickle.load(open(gps_path+'/pol_sample_cost_composition_itr_'+str('%02d' % pp)+'.pkl', 'rb')))

total_cond = len(pol_sample_lists_costs[0])
total_itr = len(pol_sample_lists_costs)

if plots_type.lower() == 'iteration':
    #marker = 'o'
    marker = None
    for cond in range(total_cond):
        lines = list()
        labels = list()

        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Policy Costs | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        mean_costs = np.zeros(total_itr)
        max_costs = np.zeros(total_itr)
        min_costs = np.zeros(total_itr)
        std_costs = np.zeros(total_itr)
        for itr in range(total_itr):
            total_samples = len(pol_sample_lists_costs[itr][cond])
            samples_cost_sum = pol_sample_lists_costs[itr][cond].sum(axis=1)
            mean_costs[itr] = samples_cost_sum.mean()
            max_costs[itr] = samples_cost_sum.max()
            min_costs[itr] = samples_cost_sum.min()
            std_costs[itr] = samples_cost_sum.std()
        ax.set_title('Policy Costs | Condition %d' % cond)
        label = 'Total Cost'
        line = ax.plot(mean_costs, marker=marker, label=label)[0]
        ax.fill_between(range(total_itr), min_costs, max_costs, alpha=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        lines.append(line)
        labels.append(label)

        # Composition cost
        if plot_cost_types:
            total_cost_types = len(pol_sample_lists_cost_compositions[-1][-1][-1])
            mean_cost_types = np.zeros([total_itr, total_cost_types])
            max_cost_types = np.zeros([total_itr, total_cost_types])
            min_cost_types = np.zeros([total_itr, total_cost_types])
            std_cost_types = np.zeros([total_itr, total_cost_types])
            for itr in range(total_itr):
                total_samples = len(pol_sample_lists_cost_compositions[itr][cond])
                for c in range(total_cost_types):
                    cost_type_sum = np.zeros(total_samples)
                    for n in range(total_samples):
                        cost_type_sum[n] = np.sum(pol_sample_lists_cost_compositions[itr][cond][n][c])
                    mean_cost_types[itr, c] = cost_type_sum.mean()
                    max_cost_types[itr, c] = cost_type_sum.max()
                    min_cost_types[itr, c] = cost_type_sum.min()
                    std_cost_types[itr, c] = cost_type_sum.std()

            for c in range(total_cost_types):
                label = 'Cost type %d' % c
                line = ax.plot(mean_cost_types[:, c], marker=marker, label=label)[0]
                ax.fill_between(range(total_itr), min_cost_types[:, c], max_cost_types[:, c], alpha=0.5)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                lines.append(line)
                labels.append(label)

        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        ax.set_xlim([0, total_itr])
        ax.set_xticks(np.arange(0, total_itr))

        legend = plt.figlegend(lines, labels, loc='center right', ncol=1, labelspacing=0., borderaxespad=1.)
        legend.get_frame().set_alpha(0.4)

else:
    T = pol_sample_lists_costs[0][0].shape[1]
    if include_last_T is False:
        T = T - 1

    if iteration_to_plot is not None:
        if iteration_to_plot == -1:
            iteration_to_plot = total_itr - 1
        itr_to_plot = [iteration_to_plot]
    else:
        itr_to_plot = range(total_itr)

    for cond in range(total_cond):
        lines = list()
        labels = list()

        total_cost_types = len(pol_sample_lists_cost_compositions[-1][-1][-1])

        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Policy Costs | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        if plot_cost_types:
            colormap_list = [colormap(i) for i in np.linspace(0, 1, (len(itr_to_plot)*total_cost_types)+1)]
        else:
            colormap_list = [colormap(i) for i in np.linspace(0, 1, len(itr_to_plot))]
        ax.set_prop_cycle('color', colormap_list)
        ax.set_title('Policy Costs | Condition %d' % cond)

        mean_costs = np.zeros([total_itr, T])
        max_costs = np.zeros([total_itr, T])
        min_costs = np.zeros([total_itr, T])
        std_costs = np.zeros([total_itr, T])
        for itr in itr_to_plot:
            total_samples = len(pol_sample_lists_costs[itr][cond])
            samples_cost = pol_sample_lists_costs[itr][cond][:, :T]
            mean_costs[itr, :] = samples_cost.mean(axis=0)
            max_costs[itr, :] = samples_cost.max(axis=0)
            min_costs[itr, :] = samples_cost.min(axis=0)
            std_costs[itr, :] = samples_cost.std(axis=0)
            label = 'Total Cost (itr%d)' % itr
            line = ax.plot(mean_costs[itr, :], label=label)[0]
            ax.fill_between(range(T), min_costs[itr, :], max_costs[itr, :], alpha=0.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            lines.append(line)
            labels.append(label)

        # Composition cost
        if plot_cost_types:
            mean_cost_types = np.zeros([total_itr, total_cost_types, T])
            max_cost_types = np.zeros([total_itr, total_cost_types, T])
            min_cost_types = np.zeros([total_itr, total_cost_types, T])
            std_cost_types = np.zeros([total_itr, total_cost_types, T])
            for itr in itr_to_plot:
                total_samples = len(pol_sample_lists_cost_compositions[itr][cond])
                for c in range(total_cost_types):
                    cost_type = np.zeros([total_samples, T])
                    for n in range(total_samples):
                        cost_type[n, :] = pol_sample_lists_cost_compositions[itr][cond][n][c][:T]
                    mean_cost_types[itr, c, :] = cost_type.mean(axis=0)
                    max_cost_types[itr, c, :] = cost_type.max(axis=0)
                    min_cost_types[itr, c, :] = cost_type.min(axis=0)
                    std_cost_types[itr, c, :] = cost_type.std(axis=0)

            for c in range(total_cost_types):
                label = 'Cost type %d (itr%d)' % (c, itr)
                line = ax.plot(mean_cost_types[itr, c, :], label=label)[0]
                ax.fill_between(range(T), min_cost_types[itr, c, :], max_cost_types[itr, c, :], alpha=0.5)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                lines.append(line)
                labels.append(label)

        plt.xlabel('Time')
        plt.ylabel('Cost')
        ax.set_xlim([0, T])
        ax.set_xticks(np.arange(0, T+2, 50))

        legend = plt.figlegend(lines, labels, loc='center right', ncol=1, labelspacing=0., borderaxespad=1.)
        legend.get_frame().set_alpha(0.4)

plt.show(block=False)

raw_input('Showing plots. Press a key to close...')
