import os
from robolearn.old_utils.plots.policy_cost import plot_policy_cost

method = 'gps'  # 'gps' or 'trajopt'
gps_directory_names = ['gps_log1']
gps_models_labels = ['gps_log1']
itr_to_load = None  # list(range(8))
block = False
plot_cs = True
plot_policy_costs = True
plot_cost_types = False

dir_names = [os.path.dirname(os.path.realpath(__file__)) + '/../' + dir_name
             for dir_name in gps_directory_names]

plot_policy_cost(dir_names, itr_to_load=itr_to_load, method=method,
                 gps_models_labels=gps_models_labels, block=block,
                 plot_cs=plot_cs, plot_policy_costs=plot_policy_costs,
                 plot_cost_types=plot_cost_types)

input('Showing plots. Press a key to close...')
