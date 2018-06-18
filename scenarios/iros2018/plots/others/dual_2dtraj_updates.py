import os
from builtins import input
from robolearn.old_utils.plots.dual_2dtraj_updates import plot_dual_2dtraj_updates

method = 'gps'  # 'gps' or 'trajopt'
gps_directory_names = ['gps_log1', 'gps_log2', 'gps_log3']#, 'reacher_log2', 'reacher_log3']
gps_models_labels = ['gps1', 'gps2', 'gps3']
idx_to_plot = [6, 7]  # dX + dU
tgt_positions = [(0.70, -0.15),
                 (0.63, -0.09),
                 (0.51, 0.05),
                 (0.51, -0.05),
                 (0.51, 0.05)]
obst_positions = [(0.74, 0.10),
                  (0.73,  -0.36),
                  (0.65,  -0.10),
                  (0.61,  0.10),
                  (0.62,  -0.08)]
safe_distance = 0.15
# itr_to_load = None  # list(range(8))
itr_to_load = None  # list(range(8))
block = False

# ---
itr_to_plot = [1]
itr_to_load = list(range(2))
# ---

dir_names = [os.path.dirname(os.path.realpath(__file__)) + '/../' + dir_name
             for dir_name in gps_directory_names]

plot_dual_2dtraj_updates(dir_names, idx_to_plot,
                         itr_to_load=itr_to_load,
                         itr_to_plot=itr_to_plot,
                         method=method,
                         gps_models_labels=gps_models_labels,
                         tgt_positions=tgt_positions,
                         obst_positions=obst_positions,
                         safe_distance=safe_distance,
                         block=block)

input('Showing plots. Press a key to close...')
