import os
from robolearn.old_utils.plots.kl_multipliers import plot_kl_multipliers

method = 'gps'  # 'gps' or 'trajopt'
gps_directory_names = ['reacher_log']#, 'reacher_log2', 'reacher_log3']
gps_models_labels = ['gps1']#, 'gps2', 'gps3']
itr_to_load = None  # list(range(8))
block = False

dir_names = [os.path.dirname(os.path.realpath(__file__)) + '/../' + dir_name
             for dir_name in gps_directory_names]

plot_kl_multipliers(dir_names, itr_to_load=itr_to_load, method=method,
                    gps_models_labels=gps_models_labels, block=block)

input('Showing plots. Press a key to close...')
