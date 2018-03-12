import os
from builtins import input
import numpy as np
from robolearn.utils.plots.policy_final_safe_distance import plot_policy_final_safe_distance

method = 'gps'  # 'gps' or 'trajopt'
gps_directory_names = ['reacher_log']#, 'reacher_log2', 'reacher_log3']
gps_models_labels = ['gps1']#, 'gps2', 'gps3']
safe_distance = 0.15
safe_states_tuples = [(6, 12), (7, 13)]
itr_to_load = None  # list(range(8))
block = False

dir_names = [os.path.dirname(os.path.realpath(__file__)) + '/../' + dir_name
             for dir_name in gps_directory_names]

plot_policy_final_safe_distance(dir_names, safe_states_tuples,
                                itr_to_load=itr_to_load, method=method,
                                gps_models_labels=gps_models_labels,
                                safe_distance=safe_distance,
                                block=block)

input('Showing plots. Press a key to close...')
