import os
from robolearn.utils.plots.policy_cost import plot_policy_cost
from robolearn.utils.plots.specific_cost import plot_specific_cost
from robolearn.utils.plots.duals import plot_duals

method = 'gps'  # 'gps' or 'trajopt'
gps_directory_names = ['reacher_log']#, 'reacher_log2', 'reacher_log3']
gps_models_labels = ['gps1']#, 'gps2', 'gps3']
itr_to_load = None  # list(range(8))
block = False
specific_costs = None  #[4]  # None for all costs

dir_names = [os.path.dirname(os.path.realpath(__file__)) + '/../' + dir_name
             for dir_name in gps_directory_names]

plot_specific_cost(dir_names, itr_to_load=itr_to_load, method=method,
                   gps_models_labels=gps_models_labels, block=block,
                   specific_costs=specific_costs)

input('Showing plots. Press a key to close...')

