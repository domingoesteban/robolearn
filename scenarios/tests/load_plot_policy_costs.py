import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle

#gps_directory_name = 'GPS_2017-08-04_20:32:12'  # l1: 1.0, l2: 1.0e-3
#gps_directory_name = 'GPS_2017-08-07_16:05:32'  # l1: 1.0, l2: 0.0
#gps_directory_name = 'GPS_2017-08-07_19:35:58'  # l1: 1.0, l2: 1.0
gps_directory_name = 'GPS_2017-08-09_14:11:15'  # 2 arms

init_pol_sample_itr = 0
final_pol_sample_itr = 100
data_color = 'blue'

gps_path = '/home/desteban/workspace/robolearn/scenarios/robolearn_log/' + gps_directory_name

pol_sample_lists_costs = list()
print('Loading data from %s directory name.' % gps_directory_name)
for pp in range(init_pol_sample_itr, final_pol_sample_itr):
    if os.path.isfile(gps_path+'/pol_sample_cost_itr_'+str('%02d' % pp)+'.pkl'):
        print('Loading policy sample cost from iteration %d' % pp)
        pol_sample_lists_costs.append(pickle.load(open(gps_path+'/pol_sample_cost_itr_'+str('%02d' % pp)+'.pkl', 'rb')))

total_cond = len(pol_sample_lists_costs[0])
total_itr = len(pol_sample_lists_costs)
for cond in range(total_cond):
    fig, ax = plt.subplots(1, 1)
    fig.canvas.set_window_title('Policy Costs | Condition %d' % cond)
    fig.set_facecolor((1, 1, 1))
    mean_costs = np.zeros(total_itr)
    max_costs = np.zeros(total_itr)
    min_costs = np.zeros(total_itr)
    std_costs = np.zeros(total_itr)
    for itr in range(total_itr):
        total_samples = len(pol_sample_lists_costs[itr][cond])
        samples_cost_sum = pol_sample_lists_costs[itr][cond].sum(axis=1)
        mean_costs[itr] = samples_cost_sum.mean()
        max_costs[itr] = samples_cost_sum.max()
        min_costs[itr] = samples_cost_sum.min()
        std_costs[itr] = samples_cost_sum.std()
    ax.set_title('Policy Costs | Condition %d' % cond)
    ax.plot(mean_costs, color=data_color)
    ax.fill_between(range(total_itr), min_costs, max_costs, alpha=0.5, color=data_color)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show(block=False)

raw_input('Showing plots. Press a key to close...')
