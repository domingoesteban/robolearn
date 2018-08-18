import os
from robolearn.old_utils.plots.policy_final_distance import plot_policy_final_distance

method = 'gps'  # 'gps' or 'trajopt'
gps_directory_names = ['gps_log1', 'gps_log2', 'gps_log4']#, 'reacher_log2', 'reacher_log3']
gps_models_labels = ['MDGPS', 'MDGPS no 1/6 worst', 'MDGPS no 2/6 worst']
states_tuples = [(6, 9), (7, 10), (8, 11)]
states_tuples = [(6, 9), (7, 10)]
# itr_to_load = None  # list(range(8))
itr_to_load = None  # list(range(8))
block = False
per_state = False
latex_plot = True

dir_names = [os.path.dirname(os.path.realpath(__file__)) + '/../' + dir_name
             for dir_name in gps_directory_names]

plot_policy_final_distance(dir_names, states_tuples,
                           itr_to_load=itr_to_load, method=method,
                           per_element=per_state,
                           latex_plot=True,
                           gps_models_labels=gps_models_labels,
                           block=block)

input('Showing plots. Press a key to close...')
