import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import pickle
import math
import os
from robolearn.utils.plot_utils import plot_sample_list, plot_sample_list_distribution, lqr_forward, plot_3d_gaussian
from robolearn.algos.gps.gps_utils import IterationData
from robolearn.utils.iit.iit_robots_params import bigman_params
import scipy.stats

gps_directory_name = 'GPS_2017-09-01_15:22:55'  # Test MDGPS | Weekend
gps_directory_name = 'GPS_2017-09-04_10:45:00'  # Test MDGPS | New cov_bad
gps_directory_name = 'GPS_2017-09-07_17:15:05'


init_itr = 7
final_itr = 100
#final_itr = 30
samples_idx = [-1]  # List of samples / None: all samples
max_traj_plots = None  # None, plot all
last_n_iters = None  # None, plot all iterations
sensed_joints = 'RA'
method = 'MDGPS_MDREPS'

plot_eta = False
plot_nu = False
plot_omega = False
plot_step_mult = False  # If linearized policy(then NN policy) is worse, epsilon is reduced.
plot_cs = False
plot_sample_list_actions = False
plot_sample_list_states = False
plot_sample_list_obs = False
plot_sample_list_actions_dual = False
plot_policy_costs = False
plot_policy_output = False
plot_policy_actions = False
plot_policy_states = False
plot_policy_obs = False
plot_traj_distr = False
plot_duality_traj_distr = True
plot_3d_traj = False
plot_3d_pol_traj = False

eta_color = 'black'
cs_color = 'red'
step_mult_color = 'red'
sample_list_cols = 3
plot_sample_list_max_min = False
plot_joint_limits = True
gps_num = 0

gps_path = '/home/desteban/workspace/robolearn/scenarios/robolearn_log/' + gps_directory_name

iteration_data_list = list()
good_duality_info_list = list()
good_trajectories_info_list = list()
bad_duality_info_list = list()
bad_trajectories_info_list = list()
iteration_ids = list()
pol_sample_lists_costs = list()
pol_sample_lists_cost_compositions = list()

max_available_itr = None
for pp in range(init_itr, final_itr):
    if os.path.isfile(gps_path+'/' + str('gps%02d_' % gps_num) + method.upper() + '_iteration_data_itr_'+str('%02d' % pp)+'.pkl'):
        if os.path.isfile(gps_path+'/' + str('gps%02d_' % gps_num) + method.upper() + '_iteration_data_itr_'+str('%02d' % pp)+'.pkl'):
            max_available_itr = pp

if max_available_itr is not None:
    print("Max available iterations: %d" % max_available_itr)

    if last_n_iters is not None:
        init_itr = max(max_available_itr - last_n_iters + 1, 0)

    if max_traj_plots is not None:
        if max_available_itr > max_traj_plots:
            itr_to_load = np.linspace(init_itr, max_available_itr, max_traj_plots, dtype=np.uint8)
        else:
            itr_to_load = range(init_itr, max_available_itr+1)

    else:
        itr_to_load = range(init_itr, max_available_itr+1)

    print("Iterations to load: %s" % itr_to_load)
    for pp in itr_to_load:
        if os.path.isfile(gps_path+'/' + str('gps%02d_' % gps_num) + method.upper() + '_iteration_data_itr_'+str('%02d' % pp)+'.pkl'):
            print('Loading GPS iteration_data from iteration %d' % pp)
            iteration_data_list.append(pickle.load(open(gps_path+'/' + str('gps%02d_' % gps_num) + method.upper() +'_iteration_data_itr_'+str('%02d' % pp)+'.pkl',
                                                        'rb')))
            print('Loading GPS good_data from iteration %d' % pp)
            bad_duality_info_list.append(pickle.load(open(gps_path+'/' + str('gps%02d_' % gps_num) + 'bad_duality_info_itr_'+str('%02d' % pp)+'.pkl',
                                                           'rb')))
            good_duality_info_list.append(pickle.load(open(gps_path+'/' + str('gps%02d_' % gps_num) + 'good_duality_info_itr_'+str('%02d' % pp)+'.pkl',
                                                        'rb')))
            bad_trajectories_info_list.append(pickle.load(open(gps_path+'/' + str('gps%02d_' % gps_num) + 'bad_trajectories_info_itr_'+str('%02d' % pp)+'.pkl',
                                                          'rb')))
            good_trajectories_info_list.append(pickle.load(open(gps_path+'/' + str('gps%02d_' % gps_num) + 'good_trajectories_info_itr_'+str('%02d' % pp)+'.pkl',
                                                               'rb')))

        if os.path.isfile(gps_path+'/' + str('gps%02d_' % gps_num)+'pol_sample_cost_itr_'+str('%02d' % pp)+'.pkl'):
            print('Loading policy sample cost from iteration %d' % pp)
            pol_sample_lists_costs.append(pickle.load(open(gps_path+'/' + str('gps%02d_' % gps_num)+'pol_sample_cost_itr_'+str('%02d' % pp)+'.pkl', 'rb')))
        if os.path.isfile(gps_path+'/' + str('gps%02d_' % gps_num)+'pol_sample_cost_composition_itr_'+str('%02d' % pp)+'.pkl'):
            print('Loading policy sample cost composition from iteration %d' % pp)
            pol_sample_lists_cost_compositions.append(pickle.load(open(gps_path+'/' + str('gps%02d_' % gps_num)+'pol_sample_cost_composition_itr_'+str('%02d' % pp)+'.pkl', 'rb')))

            iteration_ids.append(pp)

    # total_cond = len(pol_sample_lists_costs[0])
    total_itr = len(iteration_data_list)
    total_cond = len(iteration_data_list[0])
    colormap = plt.cm.rainbow  # nipy_spectral, Set1, Paired, winter

joint_limits = [bigman_params['joints_limits'][ii] for ii in bigman_params['joint_ids'][sensed_joints]]
T = iteration_data_list[-1][-1].sample_list.get_actions(samples_idx).shape[1]

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

if plot_nu:
    for cond in range(total_cond):
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Nu values | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        nus = np.zeros(total_itr)
        for itr in range(total_itr):
            nus[itr] = iteration_data_list[itr][cond].nu
        ax.set_title('Nu values | Condition %d' % cond)
        ax.plot(range(1, total_itr+1), nus, color=eta_color)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

if plot_omega:
    for cond in range(total_cond):
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Omega values | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        omegas = np.zeros(total_itr)
        for itr in range(total_itr):
            omegas[itr] = iteration_data_list[itr][cond].omega
        ax.set_title('Omega values | Condition %d' % cond)
        ax.plot(range(1, total_itr+1), omegas, color=eta_color)
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

if plot_sample_list_actions_dual:
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].sample_list.get_actions(samples_idx).shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('Actions | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        lines = list()
        labels = list()
        for itr in range(total_itr):
            actions = iteration_data_list[itr][cond].sample_list.get_actions(samples_idx)
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("Action %d" % (ii+1))
                    label = "itr %d" % iteration_ids[itr]
                    line = ax.plot(actions.mean(axis=0)[:, ii], label=label)[0]

                    if ii == 0:
                        lines.append(line)
                        labels.append(label)

                    if itr == 0:
                        ax.tick_params(axis='both', direction='in')
                        #ax.set_xlim([0, actions.shape[2]])
                        #ax.set_ylim([ymin, ymax])

                    if plot_sample_list_max_min:
                        ax.fill_between(range(actions.mean(axis=0).shape[0]), actions.min(axis=0)[:, ii],
                                        actions.max(axis=0)[:, ii], alpha=0.5)
                        # # One legend for each ax
                        # legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                        # legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        legend.get_frame().set_alpha(0.4)

if plot_sample_list_actions:
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].sample_list.get_actions(samples_idx).shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('Actions | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        lines = list()
        labels = list()
        for itr in range(total_itr):
            actions = iteration_data_list[itr][cond].sample_list.get_actions(samples_idx)
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("Action %d" % (ii+1))
                    label = "itr %d" % iteration_ids[itr]
                    line = ax.plot(actions.mean(axis=0)[:, ii], label=label)[0]

                    if ii == 0:
                        lines.append(line)
                        labels.append(label)

                    if itr == 0:
                        ax.tick_params(axis='both', direction='in')
                        #ax.set_xlim([0, actions.shape[2]])
                        #ax.set_ylim([ymin, ymax])

                    if plot_sample_list_max_min:
                        ax.fill_between(range(actions.mean(axis=0).shape[0]), actions.min(axis=0)[:, ii],
                                        actions.max(axis=0)[:, ii], alpha=0.5)
                    # # One legend for each ax
                    # legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                    # legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        legend.get_frame().set_alpha(0.4)

if plot_sample_list_states:
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].sample_list.get_states(samples_idx).shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('States | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        lines = list()
        labels = list()
        for itr in range(total_itr):
            states = iteration_data_list[itr][cond].sample_list.get_states(samples_idx)
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("State %d" % (ii+1))
                    label = "itr %d" % iteration_ids[itr]
                    line = ax.plot(states.mean(axis=0)[:, ii], label=label)[0]

                    if ii == 0:
                        lines.append(line)
                        labels.append(label)

                    if itr == 0:
                        ax.tick_params(axis='both', direction='in')

                    if plot_sample_list_max_min:
                        ax.fill_between(range(states.mean(axis=0).shape[0]), states.min(axis=0)[:, ii],
                                        states.max(axis=0)[:, ii], alpha=0.5)
                    # # One legend for each ax
                    # legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                    # legend.get_frame().set_alpha(0.4)

                    if plot_joint_limits and ii < len(joint_limits):
                        ax.plot(np.tile(joint_limits[ii][0], [T]), color='black', linestyle='--', linewidth=1)
                        ax.plot(np.tile(joint_limits[ii][1], [T]), color='black', linestyle='--', linewidth=1)
                else:
                    plt.setp(ax, visible=False)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        legend.get_frame().set_alpha(0.4)

if plot_sample_list_obs:
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].sample_list.get_obs(samples_idx).shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('Observations | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        lines = list()
        labels = list()
        for itr in range(total_itr):
            obs = iteration_data_list[itr][cond].sample_list.get_obs(samples_idx)
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("Observation %d" % (ii+1))
                    label = "itr %d" % iteration_ids[itr]
                    line = ax.plot(obs.mean(axis=0)[:, ii], label=label)[0]

                    if ii == 0:
                        lines.append(line)
                        labels.append(label)

                    if itr == 0:
                        ax.tick_params(axis='both', direction='in')

                    if plot_sample_list_max_min:
                        ax.fill_between(range(states.mean(axis=0).shape[0]), states.min(axis=0)[:, ii],
                                        states.max(axis=0)[:, ii], alpha=0.5)
                    # # One legend for each ax
                    # legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                    # legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        legend.get_frame().set_alpha(0.4)

if plot_policy_output:
    pol_sample_to_vis = -1
    pol_confidence = 0.95
    plot_confidence_interval = False
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].pol_info.pol_mu.shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title("Policy's Actions | Condition %d" % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])
        for itr in range(total_itr):
            mus = iteration_data_list[itr][cond].pol_info.pol_mu[pol_sample_to_vis]
            sigs = iteration_data_list[itr][cond].pol_info.pol_sig[pol_sample_to_vis]
            mins = np.zeros_like(mus)
            maxs = np.zeros_like(mus)
            for tt in range(mins.shape[0]):
                for dd in range(mins.shape[1]):
                    mins[tt, dd], maxs[tt, dd] = scipy.stats.norm.interval(pol_confidence,
                                                                           loc=mus[tt, dd],
                                                                           scale=sigs[tt, dd, dd])
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("Action %d" % (ii+1))
                    ax.plot(mus[:, ii], label=("itr %d" % iteration_ids[itr]))
                    if plot_confidence_interval:
                        ax.fill_between(range(mus.shape[0]), mins[:, ii], maxs[:, ii], alpha=0.5)
                    legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                    legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

if plot_policy_actions:
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].pol_info.policy_samples.get_actions(samples_idx).shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('Policy Actions | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        lines = list()
        labels = list()
        for itr in range(total_itr):
            actions = iteration_data_list[itr][cond].pol_info.policy_samples.get_actions(samples_idx)
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("Action %d" % (ii+1))
                    label = "itr %d" % iteration_ids[itr]
                    line = ax.plot(actions.mean(axis=0)[:, ii], label=label)[0]

                    if ii == 0:
                        lines.append(line)
                        labels.append(label)

                    if itr == 0:
                        ax.tick_params(axis='both', direction='in')
                        #ax.set_xlim([0, actions.shape[2]])
                        #ax.set_ylim([ymin, ymax])

                    if plot_sample_list_max_min:
                        ax.fill_between(range(actions.mean(axis=0).shape[0]), actions.min(axis=0)[:, ii],
                                        actions.max(axis=0)[:, ii], alpha=0.5)
                        # # One legend for each ax
                        # legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                        # legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        legend.get_frame().set_alpha(0.4)


if plot_policy_states:
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].pol_info.policy_samples.get_states(samples_idx).shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('Policy States | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        lines = list()
        labels = list()
        for itr in range(total_itr):
            states = iteration_data_list[itr][cond].pol_info.policy_samples.get_states(samples_idx)
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("State %d" % (ii+1))
                    label = "itr %d" % iteration_ids[itr]
                    line = ax.plot(states.mean(axis=0)[:, ii], label=label)[0]

                    if ii == 0:
                        lines.append(line)
                        labels.append(label)

                    if itr == 0:
                        ax.tick_params(axis='both', direction='in')

                    if plot_sample_list_max_min:
                        ax.fill_between(range(states.mean(axis=0).shape[0]), states.min(axis=0)[:, ii],
                                        states.max(axis=0)[:, ii], alpha=0.5)
                        # # One legend for each ax
                        # legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                        # legend.get_frame().set_alpha(0.4)

                    if plot_joint_limits and ii < len(joint_limits):
                        ax.plot(np.tile(joint_limits[ii][0], [T]), color='black', linestyle='--', linewidth=1)
                        ax.plot(np.tile(joint_limits[ii][1], [T]), color='black', linestyle='--', linewidth=1)
                else:
                    plt.setp(ax, visible=False)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        legend.get_frame().set_alpha(0.4)

if plot_policy_obs:
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].pol_info.policy_samples.get_obs(samples_idx).shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('Policy Observations | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        lines = list()
        labels = list()
        for itr in range(total_itr):
            obs = iteration_data_list[itr][cond].pol_info.policy_samples.get_obs(samples_idx)
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("Observation %d" % (ii+1))
                    label = "itr %d" % iteration_ids[itr]
                    line = ax.plot(obs.mean(axis=0)[:, ii], label=label)[0]

                    if ii == 0:
                        lines.append(line)
                        labels.append(label)

                    if itr == 0:
                        ax.tick_params(axis='both', direction='in')

                    if plot_sample_list_max_min:
                        ax.fill_between(range(states.mean(axis=0).shape[0]), states.min(axis=0)[:, ii],
                                        states.max(axis=0)[:, ii], alpha=0.5)
                    # # One legend for each ax
                    # legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                    # legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        legend.get_frame().set_alpha(0.4)

if plot_traj_distr:
    traj_distr_confidence = 0.95
    plot_confidence_interval = False
    plot_legend = True
    for cond in range(total_cond):
        dX = iteration_data_list[-1][cond].traj_distr.dX
        dU = iteration_data_list[-1][cond].traj_distr.dU
        fig_act, axs_act = plt.subplots(int(math.ceil(float(dU)/sample_list_cols)), sample_list_cols)
        fig_act.subplots_adjust(hspace=0)
        fig_act.canvas.set_window_title("Trajectory Distribution's Actions | Condition %d" % cond)
        fig_act.set_facecolor((1, 1, 1))
        fig_state, axs_state = plt.subplots(int(math.ceil(float(dX)/sample_list_cols)), sample_list_cols)
        fig_state.subplots_adjust(hspace=0)
        fig_state.canvas.set_window_title("Trajectory Distribution's States | Condition %d" % cond)
        fig_state.set_facecolor((1, 1, 1))

        for ii in range(axs_act.size):
            ax = axs_act[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])
        for ii in range(axs_state.size):
            ax = axs_state[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        for itr in range(total_itr):
            traj_distr = iteration_data_list[itr][cond].traj_distr
            traj_info = iteration_data_list[itr][cond].traj_info

            mu, sigma = lqr_forward(traj_distr, traj_info)
            T = traj_distr.T
            dU = traj_distr.dU
            dX = traj_distr.dX
            x_idxs = range(dX)
            u_idxs = range(dX, dX+dU)
            mins = np.zeros_like(mu)
            maxs = np.zeros_like(mu)
            if plot_confidence_interval:
                for tt in range(T):
                    sigma_diag = np.diag(sigma[tt, :, :])
                    mins[tt, :], maxs[tt, :] = scipy.stats.norm.interval(traj_distr_confidence, loc=mu[tt, :],
                                                                         scale=sigma_diag[:])

            for ii in range(axs_act.size):
                ax = axs_act[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dU:
                    ax.set_title("Action %d" % (ii+1))
                    ax.plot(mu[:, u_idxs[ii]], label=("itr %d" % iteration_ids[itr]))
                    if plot_confidence_interval:
                        ax.fill_between(T, mins[:, ii], maxs[:, ii], alpha=0.5)
                    if plot_legend:
                        legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                        legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

            for ii in range(axs_state.size):
                ax = axs_state[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dX:
                    ax.set_title("State %d" % (ii+1))
                    ax.plot(mu[:, x_idxs[ii]], label=("itr %d" % iteration_ids[itr]))
                    if plot_confidence_interval:
                        ax.fill_between(T, mins[:, ii], maxs[:, ii], alpha=0.5)
                    if plot_legend:
                        legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                        legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

if plot_duality_traj_distr:
    traj_distr_confidence = 0.95
    plot_confidence_interval = True
    plot_legend = True
    for cond in range(total_cond):
        dX = iteration_data_list[-1][cond].traj_distr.dX
        dU = iteration_data_list[-1][cond].traj_distr.dU
        fig_act, axs_act = plt.subplots(int(math.ceil(float(dU)/sample_list_cols)), sample_list_cols)
        fig_act.subplots_adjust(hspace=0)
        fig_act.canvas.set_window_title("Trajectory Distribution's Actions | Condition %d" % cond)
        fig_act.set_facecolor((1, 1, 1))
        fig_state, axs_state = plt.subplots(int(math.ceil(float(dX)/sample_list_cols)), sample_list_cols)
        fig_state.subplots_adjust(hspace=0)
        fig_state.canvas.set_window_title("Trajectory Distribution's States | Condition %d" % cond)
        fig_state.set_facecolor((1, 1, 1))

        for ii in range(axs_act.size):
            ax = axs_act[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])
        for ii in range(axs_state.size):
            ax = axs_state[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        for itr in range(total_itr):
            traj_distr = iteration_data_list[itr][cond].traj_distr
            traj_info = iteration_data_list[itr][cond].traj_info

            mu, sigma = lqr_forward(traj_distr, traj_info)
            T = traj_distr.T
            dU = traj_distr.dU
            dX = traj_distr.dX
            x_idxs = range(dX)
            u_idxs = range(dX, dX+dU)
            mins = np.zeros_like(mu)
            maxs = np.zeros_like(mu)
            if plot_confidence_interval:
                for tt in range(T):
                    sigma_diag = np.diag(sigma[tt, :, :])
                    mins[tt, :], maxs[tt, :] = scipy.stats.norm.interval(traj_distr_confidence, loc=mu[tt, :],
                                                                         scale=sigma_diag[:])

            for ii in range(axs_act.size):
                ax = axs_act[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dU:
                    ax.set_title("Action %d" % (ii+1))
                    ax.plot(mu[:, u_idxs[ii]], label=("itr %d" % iteration_ids[itr]))
                    if plot_confidence_interval:
                        ax.fill_between(range(T), mins[:, u_idxs[ii]], maxs[:, u_idxs[ii]], alpha=0.5)
                    if plot_legend:
                        legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                        legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

            for ii in range(axs_state.size):
                ax = axs_state[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dX:
                    ax.set_title("State %d" % (ii+1))
                    ax.plot(mu[:, x_idxs[ii]], label=("itr %d" % iteration_ids[itr]))
                    if plot_confidence_interval:
                        ax.fill_between(range(T), mins[:, x_idxs[ii]], maxs[:, x_idxs[ii]], alpha=0.5)
                    if plot_legend:
                        legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                        legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

if plot_3d_traj:
    distance_idxs = [24, 25, 26]  # NOT TO USE -1, -2, etc because it will get the mu and variance of u !!!
    linestyle = '-'
    linewidth = 1.0
    marker = None
    markersize = 5.0
    markeredgewidth = 1.0
    alpha = 1.0

    gauss_linestyle = ':'
    gauss_linewidth = 0.2
    gauss_marker = None
    gauss_markersize = 2.0
    gauss_markeredgewidth = 0.2
    gauss_alpha = 0.3

    views = ['XY', 'XZ']

    des_colormap = [colormap(i) for i in np.linspace(0, 1, total_itr)]

    for cond in range(total_cond):
        fig_3d_traj = plt.figure()
        lines = list()
        labels = list()

        for vv, view in enumerate(views):
            ax_3d_traj = fig_3d_traj.add_subplot(1, len(views), vv+1, projection='3d')
            ax_3d_traj.set_prop_cycle('color', des_colormap)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            plot = ax_3d_traj.plot([0], [0], [0], color='green', marker='o', markersize=10)

            fig_3d_traj.canvas.set_window_title("Expected Trajectories | Condition %d" % cond)
            ax_3d_traj.set_xlabel('X')
            ax_3d_traj.set_ylabel('Y')
            ax_3d_traj.set_zlabel('Z')

            if view == 'XY':
                azim = 0.
                elev = 90.
            elif view == 'XZ':
                azim = 90.
                elev = 0.
            elif view == 'YZ':
                azim = 90.
                elev = 90.
            else:
                raise AttributeError("Wrong view %s" % view)

            ax_3d_traj.view_init(elev=elev, azim=azim)

            for itr in range(total_itr):
                traj_distr = iteration_data_list[itr][cond].traj_distr
                traj_info = iteration_data_list[itr][cond].traj_info

                mu, sigma = lqr_forward(traj_distr, traj_info)

                label = "itr %d" % iteration_ids[itr]

                xs = np.linspace(5, 0, 100)
                plot = ax_3d_traj.plot(mu[:, distance_idxs[0]],
                                       mu[:, distance_idxs[1]],
                                       zs=mu[:, distance_idxs[2]],
                                       linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize,
                                       markeredgewidth=markeredgewidth, alpha=alpha, color=des_colormap[itr],
                                       label=label)[0]

                if vv == 0:
                    lines.append(plot)
                    labels.append(label)

                sigma_idx = np.ix_(distance_idxs, distance_idxs)
                plot_3d_gaussian(ax_3d_traj, mu[:, distance_idxs], sigma[:, sigma_idx[0], sigma_idx[1]],
                                 sigma_axes=view, edges=100, linestyle=gauss_linestyle, linewidth=gauss_linewidth,
                                 color=des_colormap[itr], alpha=gauss_alpha, label='',
                                 markeredgewidth=gauss_markeredgewidth)

                X = np.append(mu[:, distance_idxs[0]], 0)
                Y = np.append(mu[:, distance_idxs[1]], 0)
                Z = np.append(mu[:, distance_idxs[2]], 0)
                mid_x = (X.max() + X.min()) * 0.5
                mid_y = (Y.max() + Y.min()) * 0.5
                mid_z = (Z.max() + Z.min()) * 0.5
                max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

                ax_3d_traj.set_xlim(mid_x - max_range, mid_x + max_range)
                ax_3d_traj.set_ylim(mid_y - max_range, mid_y + max_range)
                ax_3d_traj.set_zlim(mid_z - max_range, mid_z + max_range)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0.)
        legend.get_frame().set_alpha(0.4)

if plot_3d_pol_traj:
    distance_idxs = [24, 25, 26]  # NOT TO USE -1, -2, etc because it will get the mu and variance of u !!!
    linestyle = '-'
    linewidth = 1.0
    marker = None
    markersize = 5.0
    markeredgewidth = 1.0
    alpha = 1.0

    gauss_linestyle = ':'
    gauss_linewidth = 0.2
    gauss_marker = None
    gauss_markersize = 2.0
    gauss_markeredgewidth = 0.2
    gauss_alpha = 0.3

    views = ['XY', 'XZ']

    des_colormap = [colormap(i) for i in np.linspace(0, 1, total_itr)]

    samples_idx = -1

    for cond in range(total_cond):
        fig_3d_traj = plt.figure()
        lines = list()
        labels = list()

        for vv, view in enumerate(views):
            ax_3d_traj = fig_3d_traj.add_subplot(1, len(views), vv+1, projection='3d')
            ax_3d_traj.set_prop_cycle('color', des_colormap)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            plot = ax_3d_traj.plot([0], [0], [0], color='green', marker='o', markersize=10)

            fig_3d_traj.canvas.set_window_title("Expected Trajectories | Condition %d" % cond)
            ax_3d_traj.set_xlabel('X')
            ax_3d_traj.set_ylabel('Y')
            ax_3d_traj.set_zlabel('Z')

            if view == 'XY':
                azim = 0.
                elev = 90.
            elif view == 'XZ':
                azim = 90.
                elev = 0.
            elif view == 'YZ':
                azim = 90.
                elev = 90.
            else:
                raise AttributeError("Wrong view %s" % view)

            ax_3d_traj.view_init(elev=elev, azim=azim)

            for itr in range(total_itr):
                # traj_distr = iteration_data_list[itr][cond].traj_distr
                # traj_info = iteration_data_list[itr][cond].traj_info
                # mu, sigma = lqr_forward(traj_distr, traj_info)

                mu = iteration_data_list[itr][cond].pol_info.policy_samples.get_states()[samples_idx, :, :]

                label = "itr %d" % iteration_ids[itr]

                xs = np.linspace(5, 0, 100)
                plot = ax_3d_traj.plot(mu[:, distance_idxs[0]],
                                       mu[:, distance_idxs[1]],
                                       zs=mu[:, distance_idxs[2]],
                                       linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize,
                                       markeredgewidth=markeredgewidth, alpha=alpha, color=des_colormap[itr],
                                       label=label)[0]

                if vv == 0:
                    lines.append(plot)
                    labels.append(label)

                # sigma_idx = np.ix_(distance_idxs, distance_idxs)
                # plot_3d_gaussian(ax_3d_traj, mu[:, distance_idxs], sigma[:, sigma_idx[0], sigma_idx[1]],
                #                  sigma_axes=view, edges=100, linestyle=gauss_linestyle, linewidth=gauss_linewidth,
                #                  color=des_colormap[itr], alpha=gauss_alpha, label='',
                #                  markeredgewidth=gauss_markeredgewidth)

                X = np.append(mu[:, distance_idxs[0]], 0)
                Y = np.append(mu[:, distance_idxs[1]], 0)
                Z = np.append(mu[:, distance_idxs[2]], 0)
                mid_x = (X.max() + X.min()) * 0.5
                mid_y = (Y.max() + Y.min()) * 0.5
                mid_z = (Z.max() + Z.min()) * 0.5
                max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

                ax_3d_traj.set_xlim(mid_x - max_range, mid_x + max_range)
                ax_3d_traj.set_ylim(mid_y - max_range, mid_y + max_range)
                ax_3d_traj.set_zlim(mid_z - max_range, mid_z + max_range)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0.)
        legend.get_frame().set_alpha(0.4)


if plot_policy_costs:
    plots_type = 'iteration'  # 'iteration' or 'episode'
    include_last_T = False  # Only in iteration
    iteration_to_plot = -1
    plot_cost_types = True
    colormap = plt.cm.rainbow  # nipy_spectral, Set1, Paired, winter, rainbow

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
