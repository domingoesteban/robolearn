import matplotlib.pyplot as plt
import numpy as np
import math


def plot_sample(sample, data_to_plot='actions', block=True):

    if data_to_plot == 'actions':
        cols = 3
        actions = sample.get_acts()
        dU = actions.shape[1]
        fig, ax = plt.subplots(dU/cols+1, cols)
        fig.canvas.set_window_title("Actions from Sample")
        for ii in range(dU):
            plt.subplot(dU/cols+1, cols, ii+1)
            plt.title("Action %d" % (ii+1))
            fig.set_facecolor((0.5922, 0.6, 1))
            plt.plot(actions[:, ii])
        plt.show(block=block)

    elif data_to_plot == 'states':
        cols = 3
        states = sample.get_states()
        dX = states.shape[1]
        fig, ax = plt.subplots(dX/cols+1, cols)
        fig.canvas.set_window_title("States from Sample")
        for ii in range(dX):
            plt.subplot(dX/cols+1, cols, ii+1)
            plt.title("State %d" % (ii+1))
            fig.set_facecolor((0, 0.8, 0))
            plt.plot(states[:, ii])
        plt.show(block=block)

    else:
        raise AttributeError("Wrong data to plot!")


def plot_real_sensed_torque(joints_to_plot, taus, sensed_taus,  joint_names, block=True, cols=3):
    # TODO: Check sizes
    dU = len(joints_to_plot)
    fig, axs = plt.subplots(int(math.ceil(float(dU)/cols)), cols)
    fig.canvas.set_window_title("Joint Torques")
    fig.set_facecolor((1, 1, 1))
    for ii in range(axs.size):
        ax1 = axs[ii/cols, ii % cols]
        if ii < dU:
            ax1.set_title("Joint %d: %s" % (ii+1, joint_names[ii]))
            ax1.plot(taus[:, joints_to_plot[ii]], 'k--')
            ax1.plot(sensed_taus[:, joints_to_plot[ii]], 'k')
            ax1.set_ylabel('Torque (Nm)', color='k')
            ax1.tick_params('y', colors='k')
            ax1.tick_params(direction='in')
        else:
            plt.setp(ax1, visible=False)

    fig.subplots_adjust(hspace=0)
    plt.show(block=block)

    return fig, axs


def plot_real_sensed_position(joints_to_plot, qs, sensed_qs,  joint_names, block=True, cols=3):
    # TODO: Check sizes
    dU = len(joints_to_plot)
    fig, axs = plt.subplots(int(math.ceil(float(dU)/cols)), cols)
    fig.canvas.set_window_title("Joint Torques")
    fig.set_facecolor((1, 1, 1))
    for ii in range(axs.size):
        ax1 = axs[ii/cols, ii % cols]
        if ii < dU:
            ax1.set_title("Joint %d: %s" % (ii+1, joint_names[ii]))
            ax1.plot(qs[:, joints_to_plot[ii]], 'k--')
            ax1.plot(sensed_qs[:, joints_to_plot[ii]], 'k')
            ax1.set_ylabel('Position (rad)', color='k')
            ax1.tick_params('y', colors='k')
            ax1.tick_params(direction='in')
        else:
            plt.setp(ax1, visible=False)

    fig.subplots_adjust(hspace=0)
    plt.show(block=block)

    return fig, axs


def plot_real_sensed_torque_position(joints_to_plot, taus, sensed_taus, qs, sensed_qs,  joint_names, block=True, cols=3):
    # TODO: Check sizes
    dU = len(joints_to_plot)
    fig, axs = plt.subplots(int(math.ceil(float(dU)/cols)), cols)
    fig.canvas.set_window_title("Joint Torques/Positions")
    fig.set_facecolor((1, 1, 1))
    for ii in range(axs.size):
        ax1 = axs[ii/cols, ii % cols]
        if ii < dU:
            ax1.set_title("Joint %d: %s" % (ii+1, joint_names[ii]))
            ax1.plot(taus[:, joints_to_plot[ii]], 'r--')
            ax1.plot(sensed_taus[:, joints_to_plot[ii]], 'k')
            ax1.set_ylabel('Torque (Nm)', color='k')
            ax1.tick_params('y', colors='k')
            ax1.tick_params(direction='in')
            ax2 = ax1.twinx()
            ax2.plot(qs[:, joints_to_plot[ii]], 'm--')
            ax2.plot(sensed_qs[:, joints_to_plot[ii]], 'g')
            ax2.set_ylabel('Position (rad)', color='g')
            ax2.tick_params('y', colors='g')
            ax2.tick_params(direction='in')
        else:
            plt.setp(ax1, visible=False)

    fig.subplots_adjust(hspace=0)
    plt.show(block=block)

    return fig, axs
