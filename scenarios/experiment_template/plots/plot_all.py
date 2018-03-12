import os
from robolearn.utils.plots.policy_cost import plot_policy_cost
from robolearn.utils.plots.specific_cost import plot_specific_cost
from robolearn.utils.plots.duals import plot_duals
from robolearn.utils.plots.kl_multipliers import plot_kl_multipliers
from robolearn.utils.plots.policy_final_distance import plot_policy_final_distance
from robolearn.utils.plots.policy_final_safe_distance import plot_policy_final_safe_distance

method = 'gps'  # 'gps' or 'trajopt'
gps_directory_names = ['reacher_log']#, 'reacher_log2', 'reacher_log3']
gps_models_labels = ['gps1']#, 'gps2', 'gps3']
itr_to_load = None  # list(range(8))
states_tuples = [(6, 9), (7, 10)]
safe_distance = 0.15
safe_states_tuples = [(6, 12), (7, 13)]
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

plot_kl_multipliers(dir_names, itr_to_load=itr_to_load, method=method,
                    gps_models_labels=gps_models_labels, block=block)

plot_duals(dir_names, itr_to_load=itr_to_load, method=method,
           gps_models_labels=gps_models_labels, block=block)

plot_policy_final_distance(dir_names, states_tuples, itr_to_load=itr_to_load,
                           method=method, gps_models_labels=gps_models_labels,
                           block=block)

plot_policy_final_safe_distance(dir_names, safe_states_tuples,
                                itr_to_load=itr_to_load, method=method,
                                gps_models_labels=gps_models_labels,
                                safe_distance=safe_distance,
                                block=block)

input('Showing plots. Press a key to close...')
