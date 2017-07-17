import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import math
import os
from robolearn.utils.plot_utils import plot_sample_list, plot_sample_list_distribution
from robolearn.algos.gps.gps_utils import IterationData
import scipy.stats

# gps_directory_name = 'GPS_2017-07-13_11:30:33'
# gps_directory_name = 'GPS_2017-07-13_17:07:10'
# gps_directory_name = 'GPS_2017-07-14_10:05:47'
gps_directory_name = 'GPS_2017-07-14_16:49:21'

init_itr = 0
final_itr = 100

plot_eta = False
plot_step_mult = False
plot_cs = False
plot_sample_list_actions = False
plot_sample_list_states = False
plot_sample_list_obs = False
plot_policy_output = True
plot_traj_distr = False

eta_color = 'black'
cs_color = 'red'
step_mult_color = 'red'
sample_list_cols = 3
plot_sample_list_max_min = False

gps_path = '/home/desteban/workspace/robolearn/scenarios/' + gps_directory_name

iteration_data_list = list()
iteration_ids = list()
for pp in range(init_itr, final_itr):
    if os.path.isfile(gps_path+'/MDGPS_iteration_data_itr_'+str('%02d' % pp)+'.pkl'):
        print('Loading GPS iteration_data from iteration %d' % pp)
        iteration_data_list.append(pickle.load(open(gps_path+'/MDGPS_iteration_data_itr_'+str('%02d' % pp)+'.pkl',
                                                    'rb')))
        iteration_ids.append(pp)

# total_cond = len(pol_sample_lists_costs[0])
total_itr = len(iteration_data_list)
total_cond = len(iteration_data_list[0])
colormap = plt.cm.nipy_spectral  # nipy_spectral, Set1, Paired

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
        fig.canvas.set_window_title('Actions | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])
        for itr in range(total_itr):
            actions = iteration_data_list[itr][cond].sample_list.get_actions()
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("Action %d" % (ii+1))
                    ax.plot(actions.mean(axis=0)[:, ii], label=("itr %d" % iteration_ids[itr]))
                    if plot_sample_list_max_min:
                        ax.fill_between(range(actions.mean(axis=0).shape[0]), actions.min(axis=0)[:, ii],
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
        fig.canvas.set_window_title('States | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])
        for itr in range(total_itr):
            states = iteration_data_list[itr][cond].sample_list.get_states()
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("State %d" % (ii+1))
                    ax.plot(states.mean(axis=0)[:, ii], label=("itr %d" % iteration_ids[itr]))
                    if plot_sample_list_max_min:
                        ax.fill_between(range(states.mean(axis=0).shape[0]), states.min(axis=0)[:, ii],
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
        fig.canvas.set_window_title('Observations | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])
        for itr in range(total_itr):
            obs = iteration_data_list[itr][cond].sample_list.get_obs()
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("Observation %d" % (ii+1))
                    ax.plot(obs.mean(axis=0)[:, ii], label=("itr %d" % iteration_ids[itr]))
                    if plot_sample_list_max_min:
                        ax.fill_between(range(states.mean(axis=0).shape[0]), states.min(axis=0)[:, ii],
                                        states.max(axis=0)[:, ii], alpha=0.5)
                    legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                    legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)


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


def plot_3d_gaussian(self, i, mu, sigma, edges=100, linestyle='-.',
                     linewidth=1.0, color='black', alpha=0.1, label=''):
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

    sigma_xy = sigma[:, 0:2, 0:2]
    u, s, v = np.linalg.svd(sigma_xy)

    for t in range(T):
        xyz = np.repeat(mu[t, :].reshape((1, 3)), edges, axis=0)
        xyz[:, 0:2] += np.dot(xy_ellipse, np.dot(np.diag(
            np.sqrt(s[t, :])), u[t, :, :].T))
        self.plot_3d_points(i, xyz, linestyle=linestyle,
                            linewidth=linewidth, color=color, alpha=alpha, label=label)


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
            np.hstack([
                sigma[t, idx_x, idx_x],
                sigma[t, idx_x, idx_x].dot(traj_distr.K[t, :, :].T)
            ]),
            np.hstack([
                traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]),
                traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(
                    traj_distr.K[t, :, :].T
                ) + traj_distr.pol_covar[t, :, :]
            ])
        ])
        mu[t, :] = np.hstack([
            mu[t, idx_x],
            traj_distr.K[t, :, :].dot(mu[t, idx_x]) + traj_distr.k[t, :]
        ])
        if t < T - 1:
            sigma[t+1, idx_x, idx_x] = \
                Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + \
                dyn_covar[t, :, :]
            mu[t+1, idx_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]
    return mu, sigma

if plot_traj_distr:
    traj_distr_confidence = 0.95
    plot_confidence_interval = False
    plot_legend = False
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

plt.show(block=False)

print(iteration_data_list[-1][-1].traj_distr.chol_pol_covar.shape)

raw_input('Showing plots. Press a key to close...')
