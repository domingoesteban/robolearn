import os
from robolearn.utils.plots.sum_specific_costs import plot_sum_specific_costs

method = 'gps'  # 'gps' or 'trajopt'

option = 0  # 0: plot mdgps-bmdgps-dmdgps | 1: plot remove_bad experiment
itr_to_load = None  # list(range(8))
block = False
specific_costs = [3, 4]  #None  # None for all costs
latex_plot = True

if option == 0:
    # Paper logs: Methods comparison, Distance to Tgt plot
    gps_directory_names = ['mdgps_log', 'bmdgps_log', 'dmdgps_log']
    gps_models_labels = ['MDGPS', 'B-MDGPS', 'D-MDGPS']
elif option == 1:
    # Papers logs:
    gps_directory_names = ['mdgps_log', 'mdgps_no1_log', 'mdgps_no2_log']
    gps_models_labels = ['MDGPS', 'MDGPS no 1/6 worst', 'MDGPS no 2/6 worst']
else:
    raise ValueError("Wrong script option '%s'" % str(option))

dir_names = [os.path.dirname(os.path.realpath(__file__)) + '/../' + dir_name
             for dir_name in gps_directory_names]

conds_to_combine = list(range(4))
plot_sum_specific_costs(dir_names, itr_to_load=itr_to_load, method=method,
                        gps_models_labels=gps_models_labels, block=block,
                        conds_to_combine=conds_to_combine,
                        specific_costs=specific_costs, latex_plot=latex_plot,
                        plot_title='Training conditions')

conds_to_combine = list([4])
plot_sum_specific_costs(dir_names, itr_to_load=itr_to_load, method=method,
                        gps_models_labels=gps_models_labels, block=block,
                        conds_to_combine=conds_to_combine,
                        specific_costs=specific_costs, latex_plot=latex_plot,
                        plot_title='Test condition')

input('Showing plots. Press a key to close...')

