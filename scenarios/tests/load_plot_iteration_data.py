import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import pickle
import math
import os
from robolearn.utils.plot_utils import plot_sample_list, plot_sample_list_distribution
from robolearn.algos.gps.gps_utils import IterationData
import scipy.stats

#gps_directory_name = 'GPS_2017-08-04_20:32:12'  # l1: 1.0, l2: 1.0e-3
#gps_directory_name = 'GPS_2017-08-07_16:05:32'  # l1: 1.0, l2: 0.0
gps_directory_name = 'GPS_2017-08-07_19:35:58'  # l1: 1.0, l2: 1.0
gps_directory_name = 'GPS_2017-08-10_13:08:54'  # dummy test

init_itr = 0
final_itr = 100
samples_idx = [-1]  # List of samples / None: all samples
max_traj_plots = None  # None, plot all
last_n_iters = 5  # None, plot all iterations

plot_eta = False
plot_step_mult = False  # If linearized policy(then NN policy) is worse, epsilon is reduced.
plot_cs = False
plot_sample_list_actions = False
plot_sample_list_states = False
plot_sample_list_obs = False
plot_policy_output = False
plot_policy_actions = False
plot_policy_states = True
plot_policy_obs = False
plot_traj_distr = False
plot_3d_traj = True
plot_3d_pol_traj = True

eta_color = 'black'
cs_color = 'red'
step_mult_color = 'red'
sample_list_cols = 3
plot_sample_list_max_min = False

gps_path = '/home/desteban/workspace/robolearn/scenarios/robolearn_log/' + gps_directory_name

iteration_data_list = list()
iteration_ids = list()

max_available_itr = None
for pp in range(init_itr, final_itr):
    if os.path.isfile(gps_path+'/MDGPS_iteration_data_itr_'+str('%02d' % pp)+'.pkl'):
        if os.path.isfile(gps_path+'/MDGPS_iteration_data_itr_'+str('%02d' % pp)+'.pkl'):
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
        if os.path.isfile(gps_path+'/MDGPS_iteration_data_itr_'+str('%02d' % pp)+'.pkl'):
            print('Loading GPS iteration_data from iteration %d' % pp)
            iteration_data_list.append(pickle.load(open(gps_path+'/MDGPS_iteration_data_itr_'+str('%02d' % pp)+'.pkl',
                                                        'rb')))
            iteration_ids.append(pp)

    # total_cond = len(pol_sample_lists_costs[0])
    total_itr = len(iteration_data_list)
    total_cond = len(iteration_data_list[0])
    colormap = plt.cm.rainbow  # nipy_spectral, Set1, Paired, winter

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


def plot_3d_gaussian(ax, mu, sigma, edges=100, sigma_axes='XY', linestyle='-.', linewidth=1.0, color='black', alpha=0.1,
                     label='', markeredgewidth=1.0):
    """
    Plots ellipses in the xy plane representing the Gaussian distributions 
    specified by mu and sigma.
    Args:
        mu    - Tx3 mean vector for (x, y, z)
        sigma - Tx3x3 covariance matrix for (x, y, z)
        edges - the number of edges to use to construct each ellipse
    """
    p = np.linspace(0, 2*np.pi, edges)
    xy_ellipse = np.c_[np.cos(p), np.sin(p)]
    T = mu.shape[0]

    if sigma_axes == 'XY':
        axes = [0, 1]
    elif sigma_axes == 'XZ':
        axes = [0, 2]
    elif sigma_axes == 'YZ':
        axes = [1, 2]
    else:
        raise AttributeError("Wrong sigma_axes")

    xyz_idx = np.ix_(axes)
    sigma_idx = np.ix_(axes, axes)

    sigma_axes = np.clip(sigma[:, sigma_idx[0], sigma_idx[1]], 0, 0.05)
    u, s, v = np.linalg.svd(sigma_axes)

    for t in range(T):
        xyz = np.repeat(mu[t, :].reshape((1, 3)), edges, axis=0)
        xyz[:, xyz_idx[0]] += np.dot(xy_ellipse, np.dot(np.diag(np.sqrt(s[t, :])), u[t, :, :].T))
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], linestyle=linestyle, linewidth=linewidth, marker=marker,
                markersize=markersize, markeredgewidth=markeredgewidth, alpha=alpha, color=color, label=label)


def lqr_forward(traj_distr, traj_info):
    """
    Perform LQR forward pass. Computes state-action marginals from dynamics and policy.
    Args:
        traj_distr: A linear Gaussian policy object.
        traj_info: A TrajectoryInfo object.
    Returns:
        mu: A T x dX mean action vector.
        sigma: A T x dX x dX covariance matrix.
    """
    # Compute state-action marginals from specified conditional
    # parameters and current traj_info.
    T = traj_distr.T
    dU = traj_distr.dU
    dX = traj_distr.dX

    # Constants.
    idx_x = slice(dX)

    # Allocate space.
    sigma = np.zeros((T, dX+dU, dX+dU))
    mu = np.zeros((T, dX+dU))

    # Pull out dynamics.
    Fm = traj_info.dynamics.Fm
    fv = traj_info.dynamics.fv
    dyn_covar = traj_info.dynamics.dyn_covar

    # Set initial state covariance and mean
    sigma[0, idx_x, idx_x] = traj_info.x0sigma
    mu[0, idx_x] = traj_info.x0mu

    for t in range(T):
        sigma[t, :, :] = np.vstack([
                            np.hstack([sigma[t, idx_x, idx_x],
                                       sigma[t, idx_x, idx_x].dot(traj_distr.K[t, :, :].T)]),
                            np.hstack([traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]),
                                       traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(traj_distr.K[t, :, :].T)
                                       + traj_distr.pol_covar[t, :, :]])])

        # u_t = p(u_t | x_t)
        mu[t, :] = np.hstack([mu[t, idx_x], traj_distr.K[t, :, :].dot(mu[t, idx_x]) + traj_distr.k[t, :]])

        if t < T - 1:
            # x_t+1 = p(x_t+1 | x_t, u_t)
            sigma[t+1, idx_x, idx_x] = Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + dyn_covar[t, :, :]
            mu[t+1, idx_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]
    return mu, sigma

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

plt.show(block=False)

raw_input('Showing plots. Press a key to close...')
