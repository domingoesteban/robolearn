import os
from robolearn.v010.utils.plots.policy_final_distance_new import plot_policy_final_distance_new
from builtins import input

method = 'gps'  # 'gps' or 'trajopt'


option = 0  # 0: plot mdgps-bmdgps-dmdgps | 1: plot remove_bad experiment
itr_to_load = None  # list(range(8))
block = False
per_state = False
latex_plot = True
plot_tgt = True  # False will plot safe distance

if option == 0:
    # Paper logs: Methods comparison, Distance to Tgt plot
    gps_directory_names = ['mdgps_log_CENTAURO']#, 'bmdgps_log', 'dmdgps_log']
    gps_models_labels = ['MDGPS']#, 'B-MDGPS', 'D-MDGPS']
elif option == 1:
    # Papers logs:
    gps_directory_names = ['mdgps_log_CENTAURO']#, 'mdgps_no1_log', 'mdgps_no2_log']
    gps_models_labels = ['MDGPS']#, 'MDGPS no 1/6 worst', 'MDGPS no 2/6 worst']
else:
    raise ValueError("Wrong script option '%s'" % str(option))

if plot_tgt:
    states_idxs = [14, 15, 16]
    tolerance = 0.1
else:
    states_idxs = [20, 21, 22]
    tolerance = 0.15

dir_names = [os.path.dirname(os.path.realpath(__file__)) + '/../' + dir_name
             for dir_name in gps_directory_names]

# GPS
# conds_to_combine = list(range(4))
conds_to_combine = list([0])
plot_policy_final_distance_new(dir_names, states_idxs,
                               itr_to_load=itr_to_load, method=method,
                               per_element=per_state,
                               conds_to_combine=conds_to_combine,
                               latex_plot=True,
                               gps_models_labels=gps_models_labels,
                               block=block, tolerance=tolerance,
                               plot_title='Training conditions')

conds_to_combine = list([0])
plot_policy_final_distance_new(dir_names, states_idxs,
                               itr_to_load=itr_to_load, method=method,
                               per_element=per_state,
                               conds_to_combine=conds_to_combine,
                               latex_plot=True,
                               gps_models_labels=gps_models_labels,
                               block=block, tolerance=tolerance,
                               plot_title='Test condition')

# # TRAJOPT
# conds_to_combine = list(range(5))
# conds_to_combine = None
# plot_policy_final_distance_new(dir_names, states_idxs,
#                                itr_to_load=itr_to_load, method=method,
#                                per_element=per_state,
#                                conds_to_combine=conds_to_combine,
#                                latex_plot=True,
#                                gps_models_labels=gps_models_labels,
#                                block=block, tolerance=tolerance)

input('Showing plots. Press a key to close...')
