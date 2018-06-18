import numpy as np
import matplotlib.pyplot as plt
import pickle
from robolearn.old_utils.plot_utils import plot_sample_list, plot_sample_list_distribution, plot_sample

gps_directory_name = 'LOG_2017-07-20_08:24:36'

sample_number = 0  # If None, plot all the samples and show their mean, min and max
cond = 0  # Condition number

plot_states = True
plot_actions = True
plot_obs = False

gps_path = '/home/desteban/workspace/robolearn/scenarios/' + gps_directory_name

sample = pickle.load(open(gps_path+'/cond_'+str('%02d' % cond)+'_sample_'+str('%02d' % sample_number)+'.pkl', 'rb'))

# for cond in range(total_conditions):
#     plot_sample_list(sample_list[cond], data_to_plot='actions', block=False, cols=3)
#     #plot_sample_list(sample_list[cond], data_to_plot='states', block=False, cols=3)
#     #plot_sample_list(sample_list[cond], data_to_plot='obs', block=False, cols=3)
# raw_input('Showing plots')

if plot_actions:
    plot_sample(sample, data_to_plot='actions', block=False, cols=3, color='black')
if plot_states:
    plot_sample(sample, data_to_plot='states', block=False, cols=3, color='green')
if plot_obs:
    plot_sample(sample, data_to_plot='obs', block=False, cols=3, color='blue')

raw_input('Showing plots. Press a key to close...')
