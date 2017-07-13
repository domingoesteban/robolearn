import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import math
import os
from robolearn.utils.plot_utils import plot_sample_list, plot_sample_list_distribution
from robolearn.algos.gps.gps_utils import IterationData

gps_directory_name = 'GPS_2017-07-13_11:30:33'

init_itr = 0
final_itr = 100

plot_eta = False
plot_step_mult = False
plot_cs = True
plot_sample_list_actions = False
plot_sample_list_states = False
plot_sample_list_obs = False

eta_color = 'black'
cs_color = 'red'
step_mult_color = 'red'
sample_list_cols = 3
plot_sample_list_max_min = False

gps_path = '/home/desteban/workspace/robolearn/scenarios/' + gps_directory_name

iteration_data_list = list()
for pp in range(init_itr, final_itr):
    if os.path.isfile(gps_path+'/MDGPS_iteration_data_itr_'+str('%02d' % pp)+'.pkl'):
        print('Loading GPS iteration data from iteration %d' % pp)
        iteration_data_list.append(pickle.load(open(gps_path+'/MDGPS_iteration_data_itr_'+str('%02d' % pp)+'.pkl',
                                                    'rb')))

# total_cond = len(pol_sample_lists_costs[0])
total_itr = len(iteration_data_list)
total_cond = len(iteration_data_list[0])

if plot_eta:
    for cond in range(total_cond):
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Eta values | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        etas = np.zeros(total_itr)
        for itr in range(total_itr):
            etas[itr] = iteration_data_list[itr][cond].eta
        ax.set_title('Eta values | Condition %d' % cond)
        ax.plot(range(1, total_itr+1), etas, color=eta_color)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

if plot_step_mult:
    for cond in range(total_cond):
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Step multiplier | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        etas = np.zeros(total_itr)
        for itr in range(total_itr):
            etas[itr] = iteration_data_list[itr][cond].step_mult
        ax.set_title('Step multiplier | Condition %d' % cond)
        ax.plot(range(1, total_itr+1), etas, color=eta_color)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

if plot_cs:
    for cond in range(total_cond):
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Samples Costs | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        mean_costs = np.zeros(total_itr)
        max_costs = np.zeros(total_itr)
        min_costs = np.zeros(total_itr)
        std_costs = np.zeros(total_itr)
        for itr in range(total_itr):
            samples_cost_sum = iteration_data_list[itr][cond].cs.sum(axis=1)
            mean_costs[itr] = samples_cost_sum.mean()
            max_costs[itr] = samples_cost_sum.max()
            min_costs[itr] = samples_cost_sum.min()
            std_costs[itr] = samples_cost_sum.std()
        ax.set_title('Samples Costs | Condition %d' % cond)
        ax.plot(range(1, total_itr+1), mean_costs, color=cs_color)
        ax.fill_between(range(1, total_itr+1), min_costs, max_costs, alpha=0.5, color=cs_color)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

if plot_sample_list_actions:
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].sample_list.get_actions().shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('Actions')
        fig.set_facecolor((1, 1, 1))
        for itr in range(total_itr):
            actions = iteration_data_list[itr][cond].sample_list.get_actions()
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("Action %d" % (ii+1))
                    ax.plot(actions.mean(axis=0)[:, ii], label=("itr %d" % itr))
                    if plot_sample_list_max_min:
                        ax.fill_between(range(actions.mean(axis=0).shape[0]) + 1, actions.min(axis=0)[:, ii],
                                        actions.max(axis=0)[:, ii], alpha=0.5)
                    legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                    legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

if plot_sample_list_states:
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].sample_list.get_states().shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('States')
        fig.set_facecolor((1, 1, 1))
        for itr in range(total_itr):
            states = iteration_data_list[itr][cond].sample_list.get_states()
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("State %d" % (ii+1))
                    ax.plot(states.mean(axis=0)[:, ii], label=("itr %d" % itr))
                    if plot_sample_list_max_min:
                        ax.fill_between(range(states.mean(axis=0).shape[0]) + 1, states.min(axis=0)[:, ii],
                                        states.max(axis=0)[:, ii], alpha=0.5)
                    legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                    legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

if plot_sample_list_obs:
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].sample_list.get_obs().shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('Observations')
        fig.set_facecolor((1, 1, 1))
        for itr in range(total_itr):
            obs = iteration_data_list[itr][cond].sample_list.get_obs()
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("Observation %d" % (ii+1))
                    ax.plot(obs.mean(axis=0)[:, ii], label=("itr %d" % itr))
                    if plot_sample_list_max_min:
                        ax.fill_between(range(states.mean(axis=0).shape[0]) + 1, states.min(axis=0)[:, ii],
                                        states.max(axis=0)[:, ii], alpha=0.5)
                    legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                    legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

plt.show(block=False)

raw_input('Showing plots. Press a key to close...')
