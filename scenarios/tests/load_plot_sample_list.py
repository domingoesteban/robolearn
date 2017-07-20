import numpy as np
import matplotlib.pyplot as plt
import pickle
from robolearn.utils.plot_utils import plot_sample_list, plot_sample_list_distribution, plot_sample

# gps_directory_name = 'GPS_2017-07-13_11:30:33'
gps_directory_name = 'GPS_2017-07-14_10:05:47'

gps_itr = 5  # GPS Iteration number
sample_number = 0  # If None, plot all the samples and show their mean, min and max
pol_sample = True  # If false, load the traj_sample

plot_states = True
plot_actions = True
plot_obs = False


gps_path = '/home/desteban/workspace/robolearn/scenarios/' + gps_directory_name

if pol_sample:
    sample_list = pickle.load(open(gps_path+'/pol_sample_itr_'+str('%02d' % gps_itr)+'.pkl', 'rb'))
else:
    sample_list = pickle.load(open(gps_path+'/traj_sample_itr_'+str('%02d' % gps_itr)+'.pkl', 'rb'))

total_conditions = len(sample_list)

# for cond in range(total_conditions):
#     plot_sample_list(sample_list[cond], data_to_plot='actions', block=False, cols=3)
#     #plot_sample_list(sample_list[cond], data_to_plot='states', block=False, cols=3)
#     #plot_sample_list(sample_list[cond], data_to_plot='obs', block=False, cols=3)
# raw_input('Showing plots')

for cond in range(total_conditions):
    if plot_actions:
        if sample_number is None:
            plot_sample_list_distribution(sample_list[cond], data_to_plot='actions', block=False, cols=3)
        else:
            plot_sample(sample_list[cond][sample_number], data_to_plot='actions', block=False, cols=3, color='black')
    if plot_states:
        if sample_number is None:
            plot_sample_list_distribution(sample_list[cond], data_to_plot='states', block=False, cols=3)
        else:
            plot_sample(sample_list[cond][sample_number], data_to_plot='states', block=False, cols=3, color='green')
    if plot_obs:
        if sample_number is None:
            plot_sample_list_distribution(sample_list[cond], data_to_plot='obs', block=False, cols=3)
        else:
            plot_sample(sample_list[cond][sample_number], data_to_plot='obs', block=False, cols=3, color='blue')

raw_input('Showing plots. Press a key to close...')
