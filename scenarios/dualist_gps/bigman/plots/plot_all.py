import os
from robolearn.utils.plots.policy_cost import plot_policy_cost
from robolearn.utils.plots.specific_cost import plot_specific_cost
from robolearn.utils.plots.duals import plot_duals

method = 'gps'  # 'gps' or 'trajopt'
gps_directory_names = ['gps_log1']
gps_models_labels = ['gps_log1']
itr_to_load = None  # list(range(8))
block = False
plot_cs = True
plot_policy_costs = True
plot_cost_types = False
specific_costs = None  #[4]  # None for all costs

dir_names = [os.path.dirname(os.path.realpath(__file__)) + '/../' + dir_name
             for dir_name in gps_directory_names]

plot_policy_cost(dir_names, itr_to_load=itr_to_load, method=method,
                 gps_models_labels=gps_models_labels, block=block,
                 plot_cs=plot_cs, plot_policy_costs=plot_policy_costs,
                 plot_cost_types=plot_cost_types)

plot_specific_cost(dir_names, itr_to_load=itr_to_load, method=method,
                   gps_models_labels=gps_models_labels, block=block,
                   specific_costs=specific_costs)

plot_duals(dir_names, itr_to_load=itr_to_load, method=method,
           gps_models_labels=gps_models_labels, block=block)

input('Showing plots. Press a key to close...')
