import matplotlib.pyplot as plt
import numpy as np
import math


def plot_sample(sample, data_to_plot='actions', block=True, cols=3, color='blue'):

    if data_to_plot == 'actions':
        data = sample.get_acts()
        window_title = "Actions"
        ax_title = "Action"
    elif data_to_plot == 'states':
        data = sample.get_states()
        window_title = "States"
        ax_title = "State"
    elif data_to_plot == 'obs':
        data = sample.get_obs()
        window_title = "Observations"
        ax_title = "Observation"
    else:
        raise AttributeError("Wrong data to plot!")

    dData = data.shape[1]
    fig, axs = plt.subplots(int(math.ceil(float(dData)/cols)), cols)
    fig.subplots_adjust(hspace=0)
    fig.canvas.set_window_title(window_title)
    fig.set_facecolor((1, 1, 1))
    for ii in range(axs.size):
        ax = axs[ii/cols, ii % cols]
        if ii < dData:
            ax.set_title(ax_title + " %d" % (ii+1))
            ax.plot(data[:, ii], color=color)
        else:
            plt.setp(ax, visible=False)
    plt.show(block=block)


def plot_sample_list(sample_list, data_to_plot='actions', block=True, cols=3):

    if data_to_plot == 'actions':
        data = sample_list.get_actions()
        window_title = "Actions"
        ax_title = "Action"
    elif data_to_plot == 'states':
        data = sample_list.get_states()
        window_title = "States"
        ax_title = "State"
    elif data_to_plot == 'obs':
        data = sample_list.get_obs()
        window_title = "Observations"
        ax_title = "Observation"
    else:
        raise AttributeError("Wrong data to plot!")

    for nn in range(data.shape[0]):
        dData = data.shape[2]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/cols)), cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title(window_title + " from Sample %d" % nn)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/cols, ii % cols]
            if ii < dData:
                ax.set_title(ax_title + " %d" % (ii+1))
                ax.plot(data[nn, :, ii])
            else:
                plt.setp(ax, visible=False)
        plt.show(block=block)


def plot_sample_list_distribution(sample_list, data_to_plot='actions', block=True, cols=3):
    if data_to_plot == 'actions':
        data = sample_list.get_actions()
        window_title = "Actions"
        ax_title = "Action"
        data_color = 'b'
    elif data_to_plot == 'states':
        data = sample_list.get_states()
        window_title = "States"
        ax_title = "State"
        data_color = 'r'
    elif data_to_plot == 'obs':
        data = sample_list.get_obs()
        window_title = "Observations"
        ax_title = "Observation"
        data_color = 'g'
    else:
        raise AttributeError("Wrong data to plot!")

    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)

    dData = means.shape[1]
    fig, axs = plt.subplots(int(math.ceil(float(dData)/cols)), cols)
    fig.subplots_adjust(hspace=0)
    fig.canvas.set_window_title(window_title)
    fig.set_facecolor((1, 1, 1))
    for ii in range(axs.size):
        ax = axs[ii/cols, ii % cols]
        if ii < dData:
            ax.set_title(ax_title + " %d" % (ii+1))
            ax.plot(means[:, ii], color=data_color)
            ax.fill_between(range(means.shape[0]), mins[:, ii],
                            maxs[:, ii], alpha=0.5, color=data_color)
        else:
            plt.setp(ax, visible=False)
    plt.show(block=block)


def plot_training_costs(costs, block=True):
    t = np.arange(costs.shape[0])
    plt.plot(t, np.average(costs, axis=1))
    plt.fill_between(t, np.min(costs, axis=1), np.max(costs, axis=1), alpha=0.5)
    plt.show(block=block)


def plot_desired_sensed_torque(joints_to_plot, des_taus, sensed_taus,  joint_names, block=True, cols=3):
    # TODO: Check sizes
    dU = len(joints_to_plot)
    fig, axs = plt.subplots(int(math.ceil(float(dU)/cols)), cols)
    fig.canvas.set_window_title("Joint Torques")
    fig.set_facecolor((1, 1, 1))
    for ii in range(axs.size):
        ax1 = axs[ii/cols, ii % cols]
        if ii < dU:
            ax1.set_title("Joint %d: %s" % (ii+1, joint_names[ii]))
            ax1.plot(des_taus[:, joints_to_plot[ii]], 'k--')
            ax1.plot(sensed_taus[:, joints_to_plot[ii]], 'k')
            ax1.set_ylabel('Torque (Nm)', color='k')
            ax1.tick_params('y', colors='k')
            ax1.tick_params(direction='in')
        else:
            plt.setp(ax1, visible=False)

    fig.subplots_adjust(hspace=0)
    plt.show(block=block)

    return fig, axs


def plot_desired_sensed_data(joints_to_plot, des_qs, sensed_qs,  joint_names, data_type='position',
                             limits=None, block=True, cols=3, legend=True):
    # TODO: Check sizes
    dU = len(joints_to_plot)
    fig, axs = plt.subplots(int(math.ceil(float(dU)/cols)), cols)
    if data_type.lower() == 'position':
        fig.canvas.set_window_title("Joint Positions")
        des_color = 'limegreen'
        sensed_color = 'forestgreen'
    elif data_type.lower() == 'velocity':
        fig.canvas.set_window_title("Joint Velocities")
        des_color = 'lightcoral'
        sensed_color = 'red'
    elif data_type.lower() == 'acceleration':
        fig.canvas.set_window_title("Joint Accelerations")
        des_color = 'lightskyblue'
        sensed_color = 'blue'
    elif data_type.lower() == 'torque':
        fig.canvas.set_window_title("Joint Torques")
        des_color = 'gray'
        sensed_color = 'black'
    elif data_type.lower() == 'pose':
        fig.canvas.set_window_title("Operational point Pose")
        des_color = 'thistle'
        sensed_color = 'purple'
    elif data_type.lower() == 'pose-error':
        fig.canvas.set_window_title("Operational point Pose Error")
        des_color = 'thistle'
        sensed_color = 'purple'
    else:
        raise ValueError("Wrong data_type option:%s " % data_type)
    fig.set_facecolor((1, 1, 1))
    for ii in range(axs.size):
        ax1 = axs[ii/cols, ii % cols]
        if ii < dU:
            if limits is not None:
                ax1.plot(np.tile(limits[joints_to_plot[ii]][0], sensed_qs.shape[0]), color='orange', label='limit-min')
                ax1.plot(np.tile(limits[joints_to_plot[ii]][1], sensed_qs.shape[0]), color='orange', label='limit-max')

            if not data_type.lower() in ['pose', 'pose-error']:
                ax1.set_title("Joint %d: %s" % (ii+1, joint_names[ii]))
                #label = "Joint %d: %s" % (ii+1, joint_names[ii])
            else:
                ax1.set_title("%s" % joint_names[ii])
                #label = "%s" % joint_names[ii]
            ax1.plot(sensed_qs[:, joints_to_plot[ii]], color=sensed_color, label='Sensed')
            ax1.plot(des_qs[:, joints_to_plot[ii]], '--', color=des_color, label='Desired')

            if data_type.lower() == 'position':
                ax1.set_ylabel('Position (rad)', color='k')
            elif data_type.lower() == 'velocity':
                ax1.set_ylabel('Velocity (rad/s)', color='k')
            elif data_type.lower() == 'acceleration':
                ax1.set_ylabel('Acceleration (rad/s2)', color='k')
            elif data_type.lower() == 'torque':
                ax1.set_ylabel('Torque (N/m)', color='k')
            ax1.tick_params('y', colors='k')
            ax1.tick_params(direction='in')
            if legend:
                legend = ax1.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                #legend.get_frame().set_facecolor('#00FFCC')
        else:
            plt.setp(ax1, visible=False)

    fig.subplots_adjust(hspace=0)
    plt.show(block=block)

    return fig, axs


def plot_desired_sensed_torque_position(joints_to_plot, taus, sensed_taus, qs, sensed_qs,  joint_names, block=True, cols=3):
    # TODO: Check sizes
    dU = len(joints_to_plot)
    fig, axs = plt.subplots(int(math.ceil(float(dU)/cols)), cols)
    fig.canvas.set_window_title("Joint Torques/Positions")
    fig.set_facecolor((1, 1, 1))
    for ii in range(axs.size):
        ax1 = axs[ii/cols, ii % cols]
        if ii < dU:
            ax1.set_title("Joint %d: %s" % (ii+1, joint_names[ii]))
            #ax1.plot(taus[:, joints_to_plot[ii]], '--', color='gray')
            ax1.plot(sensed_taus[:, joints_to_plot[ii]], color='black')
            ax1.plot(taus[:, joints_to_plot[ii]], '--', color='gray')
            if ii % cols == 0:
                ax1.set_ylabel('Torque (Nm)', color='k')
            ax1.tick_params('y', colors='k')
            ax1.tick_params(direction='in')
            ax2 = ax1.twinx()
            #ax2.plot(qs[:, joints_to_plot[ii]], 'm--')
            ax2.plot(sensed_qs[:, joints_to_plot[ii]], color='forestgreen')
            ax2.plot(qs[:, joints_to_plot[ii]], '--', color="limegreen")
            if ii % cols == cols - 1:
                ax2.set_ylabel('Position (rad)', color='g')
            ax2.tick_params('y', colors='g')
            ax2.tick_params(direction='in')
        else:
            plt.setp(ax1, visible=False)

    fig.subplots_adjust(hspace=0)
    plt.show(block=block)

    return fig, axs


def plot_joint_info(joints_to_plot, data_to_plot,  joint_names, data='position', block=True, cols=3):
    # TODO: Check sizes
    dU = len(joints_to_plot)
    fig, axs = plt.subplots(int(math.ceil(float(dU)/cols)), cols)
    if data == 'position':
        fig.canvas.set_window_title("Joint Positions")
    elif data == 'velocity':
        fig.canvas.set_window_title("Joint Velocities")
    elif data == 'acceleration':
        fig.canvas.set_window_title("Joint Accelerations")
    elif data == 'torque':
        fig.canvas.set_window_title("Joint Torque")
    else:
        raise ValueError("Wrong plot option")

    fig.set_facecolor((1, 1, 1))
    for ii in range(axs.size):
        ax1 = axs[ii/cols, ii % cols]
        if ii < dU:
            ax1.set_title("Joint %d: %s" % (ii+1, joint_names[ii]))
            ax1.plot(data_to_plot[:, joints_to_plot[ii]], 'b')
            if data == 'position':
                ax1.set_ylabel('Position (rad)', color='k')
            elif data == 'velocity':
                ax1.set_ylabel('Velocity (rad/s)', color='k')
            elif data == 'acceleration':
                ax1.set_ylabel('Acceleration (rad/s2)', color='k')
            elif data == 'torque':
                ax1.set_ylabel('Torque (Nm)', color='k')
            ax1.tick_params('y', colors='k')
            ax1.tick_params(direction='in')
        else:
            plt.setp(ax1, visible=False)

    fig.subplots_adjust(hspace=0)
    plt.show(block=block)

    return fig, axs


def plot_joint_multi_info(joints_to_plot, data_to_plot,  joint_names, data='position', block=True, cols=3, legend=True,
                          labels=None):
    # TODO: Check sizes
    dU = len(joints_to_plot)
    fig, axs = plt.subplots(int(math.ceil(float(dU)/cols)), cols)
    if data == 'position':
        fig.canvas.set_window_title("Multi JJoint Positions")
    elif data == 'velocity':
        fig.canvas.set_window_title("Multi JJoint Velocities")
    elif data == 'acceleration':
        fig.canvas.set_window_title("Multi JJoint Accelerations")
    elif data == 'torque':
        fig.canvas.set_window_title("Multi Joint Torque")
    else:
        raise ValueError("Wrong plot option")

    fig.set_facecolor((1, 1, 1))
    for ii in range(axs.size):
        ax1 = axs[ii/cols, ii % cols]
        if ii < dU:
            ax1.set_title("Joint %d: %s" % (ii+1, joint_names[ii]))
            for jj in range(data_to_plot.shape[0]):
                if labels is None:
                    label_name = jj
                else:
                    label_name = labels[jj]
                ax1.plot(data_to_plot[jj, :, joints_to_plot[ii]], label=label_name)

            if data == 'position':
                ax1.set_ylabel('Position (rad)', color='k')
            elif data == 'velocity':
                ax1.set_ylabel('Velocity (rad/s)', color='k')
            elif data == 'acceleration':
                ax1.set_ylabel('Acceleration (rad/s2)', color='k')
            elif data == 'torque':
                ax1.set_ylabel('Torque (Nm)', color='k')
            ax1.tick_params('y', colors='k')
            ax1.tick_params(direction='in')
            if legend:
                legend = ax1.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                legend.get_frame().set_alpha(0.4)
                #legend.get_frame().set_facecolor('#00FFCC')
        else:
            plt.setp(ax1, visible=False)

    fig.subplots_adjust(hspace=0)
    plt.show(block=block)

    return fig, axs


def plot_multi_info(data_list, block=True, cols=3, legend=True, labels=None):
    dData = data_list[0].shape[1]
    fig, axs = plt.subplots(int(math.ceil(float(dData)/cols)), cols)
    fig.set_facecolor((1, 1, 1))
    lines = list()
    if labels is None:
        labels = list()
    for ii in range(axs.size):
        ax1 = axs[ii/cols, ii % cols]
        if ii < dData:
            ax1.set_title("Dimension %d" % (ii+1))
            for jj in range(len(data_list)):
                if len(labels) > jj:
                    label = labels[jj]
                else:
                    label = 'Data %d' % jj
                line = ax1.plot(data_list[jj][:, ii], label=label)[0]

                if ii == 0:
                    lines.append(line)
                    labels.append(label)

            ax1.tick_params('y', colors='k')
            ax1.tick_params(direction='in')
        else:
            plt.setp(ax1, visible=False)


    if legend:
        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        legend.get_frame().set_alpha(0.4)

    fig.subplots_adjust(hspace=0)
    plt.show(block=block)

    return fig, axs


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


def plot_3d_gaussian(ax, mu, sigma, edges=100, sigma_axes='XY', linestyle='-.', linewidth=1.0, color='black', alpha=0.1,
                     label='', markeredgewidth=1.0, marker=None, markersize=5.0):
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


def plot_sample_list_actions(iteration_data_list, samples_idx=None, sample_list_cols=3, colormap=None):
    """
    :param iteration_data_list: 
    :param samples_idx: None: plot all samples, else list of samples
    :return: 
    """
    if colormap is None:
        colormap = plt.cm.rainbow  # nipy_spectral, Set1, Paired, winter

    plot_sample_list_max_min = False

    total_cond = len(iteration_data_list)
    for cond in range(total_cond):
        dData = iteration_data_list[cond].sample_list.get_actions(samples_idx).shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('Actions | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            #ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        lines = list()
        labels = list()

        actions = iteration_data_list[cond].sample_list.get_actions(samples_idx)

        if samples_idx is None:
            samples_idx = range(actions.shape[0])

        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            if ii < dData:
                ax.set_title("Action %d" % (ii+1))
                for nn in samples_idx:
                    label = "Sample %d" % nn
                    line = ax.plot(actions[nn][:, ii], label=label)[0]

                    if ii == 0:
                        lines.append(line)
                        labels.append(label)

                    if nn == 0:
                        ax.tick_params(axis='both', direction='in')
                        #ax.set_xlim([0, actions.shape[2]])
                        #ax.set_ylim([ymin, ymax])
            else:
                plt.setp(ax, visible=False)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        legend.get_frame().set_alpha(0.4)
