import os
from robolearn.old_utils.plots.policy_cost import plot_policy_cost
from robolearn.old_utils.plots.avg_specific_costs import plot_avg_specific_costs
from robolearn.old_utils.plots.duals import plot_duals

method = 'gps'  # 'gps' or 'trajopt'
gps_directory_names = ['gps_log4', 'gps_log7', 'gps_log8']#, 'reacher_log2', 'reacher_log3']
gps_models_labels = ['MDGPS', 'MDGPS no 1/6 worst', 'MDGPS no 2/6 worst']
itr_to_load = None  # list(range(8))
block = False
specific_costs = [3, 4]  #None  # None for all costs
latex_plot = True

dir_names = [os.path.dirname(os.path.realpath(__file__)) + '/../' + dir_name
             for dir_name in gps_directory_names]

# conds_to_combine = list(range(12))
conds_to_combine = list(range(4))
plot_avg_specific_costs(dir_names, itr_to_load=itr_to_load, method=method,
                        gps_models_labels=gps_models_labels, block=block,
                        conds_to_combine=conds_to_combine,
                        specific_costs=specific_costs, latex_plot=latex_plot)

conds_to_combine = list(range(12, 15))
conds_to_combine = list([4])
plot_avg_specific_costs(dir_names, itr_to_load=itr_to_load, method=method,
                        gps_models_labels=gps_models_labels, block=block,
                        conds_to_combine=conds_to_combine,
                        specific_costs=specific_costs, latex_plot=latex_plot)

input('Showing plots. Press a key to close...')

