import numpy as np
from robolearn.utils.plot_utils import *
from robolearn.utils.sample import Sample

block=True
cols = 3
actions = np.ones([7, 20])
dU = actions.shape[0]
fig, ax = plt.subplots(dU/cols, cols)
for ii in range(dU):
    actions[ii, :] = ii
    plt.subplot(dU/cols+1, cols, ii+1)
    fig.canvas.set_window_title("Action"+str(ii))
    fig.set_facecolor((0.5922, 0.6, 1))
    plt.plot(actions[ii])
plt.show(block=block)
