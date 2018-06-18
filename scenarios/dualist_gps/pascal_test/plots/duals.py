import os
from robolearn.old_utils.plots.duals import plot_duals

method = 'gps'  # 'gps' or 'trajopt'
gps_directory_names = ['gps_log1', 'gps_log3', 'gps_log5']
gps_models_labels = ['gps_log1', 'gps_log3', 'gps_log5']
itr_to_load = None  # list(range(8))
block = False

dir_names = [os.path.dirname(os.path.realpath(__file__)) + '/../' + dir_name
             for dir_name in gps_directory_names]

plot_duals(dir_names, itr_to_load=itr_to_load, method=method,
           gps_models_labels=gps_models_labels, block=block)

input('Showing plots. Press a key to close...')
