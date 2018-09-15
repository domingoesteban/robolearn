import os
from robolearn.old_utils.plots.specific_cost import plot_specific_cost

method = 'gps'  # 'gps' or 'trajopt'
gps_directory_names = ['gps_log1']
gps_models_labels = ['gps_log1']
itr_to_load = None  # list(range(8))
block = False
specific_costs = None  #[4]  # None for all costs

dir_names = [os.path.dirname(os.path.realpath(__file__)) + '/../' + dir_name
             for dir_name in gps_directory_names]

plot_specific_cost(dir_names, itr_to_load=itr_to_load, method=method,
                   gps_models_labels=gps_models_labels, block=block,
                   specific_costs=specific_costs)

input('Showing plots. Press a key to close...')
