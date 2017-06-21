import matplotlib.pyplot as plt
import numpy as np
import math


def plot_sample(sample, data_to_plot='actions', block=True, cols=3):

    if data_to_plot == 'actions':
        actions = sample.get_acts()
        dU = actions.shape[1]
        fig, axs = plt.subplots(int(math.ceil(float(dU)/cols)), cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title("Actions from Sample")
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/cols, ii % cols]
            if ii < dU:
                ax.set_title("Action %d" % (ii+1))
                ax.plot(actions[:, ii])
            else:
                plt.setp(ax, visible=False)
        plt.show(block=block)

    elif data_to_plot == 'states':
        states = sample.get_states()
        dX = states.shape[1]
        fig, axs = plt.subplots(int(math.ceil(float(dX)/cols)), cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title("States from Sample")
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/cols, ii % cols]
            if ii < dX:
                ax.set_title("State %d" % (ii+1))
                ax.plot(states[:, ii])
            else:
                plt.setp(ax, visible=False)
        plt.show(block=block)

    else:
        raise AttributeError("Wrong data to plot!")


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


def plot_desired_sensed_data(joints_to_plot, des_qs, sensed_qs,  joint_names, data_type='position', block=True, cols=3):
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
    else:
        raise ValueError("Wrong data_type option:%s " % data_type)
    fig.set_facecolor((1, 1, 1))
    for ii in range(axs.size):
        ax1 = axs[ii/cols, ii % cols]
        if ii < dU:
            ax1.set_title("Joint %d: %s" % (ii+1, joint_names[ii]))
            ax1.plot(sensed_qs[:, joints_to_plot[ii]], color=sensed_color)
            ax1.plot(des_qs[:, joints_to_plot[ii]], '--', color=des_color)
            ax1.set_ylabel('Position (rad)', color='k')
            ax1.tick_params('y', colors='k')
            ax1.tick_params(direction='in')
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
