import matplotlib.pyplot as plt
import numpy as np


def plot_sample(sample, data_to_plot='actions', block=True):

    if data_to_plot == 'actions':
        cols = 3
        actions = sample.get_acts()
        dU = actions.shape[1]
        fig, ax = plt.subplots(dU/cols, cols)
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
        fig, ax = plt.subplots(dX/cols, cols)
        fig.canvas.set_window_title("States from Sample")
        for ii in range(dX):
            plt.subplot(dX/cols+1, cols, ii+1)
            plt.title("State %d" % (ii+1))
            fig.set_facecolor((0, 0.8, 0))
            plt.plot(states[:, ii])
        plt.show(block=block)

    else:
        raise AttributeError("Wrong data to plot!")
