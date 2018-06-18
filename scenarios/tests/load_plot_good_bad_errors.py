import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import pickle
import math
import os, sys
from robolearn.old_utils.plot_utils import plot_sample_list, plot_sample_list_distribution, lqr_forward, plot_3d_gaussian
from robolearn.old_algos.gps.gps_utils import IterationData
from robolearn.old_utils.iit.iit_robots_params import bigman_params
from robolearn.old_utils.traj_opt.traj_opt_utils import traj_distr_kl, traj_distr_kl_alt
import scipy.stats

gps_directory_name = 'GPS_2017-09-10_15:30:24'  # Normal Sunday 10/09 | new init_pos
#gps_directory_name = 'GPS_2017-09-10_19:10:07'  # G/B Sunday 10/09 | new init_pos

gps_directory_names = ['GPS_2017-09-12_07:01:16', 'GPS_2017-09-11_15:25:19', 'GPS_2017-09-13_07:24:42']
gps_models_labels = ['MDGPS', 'B-MDGPS', 'D-MDGPS']
gps_models_line_styles = [':', '--', '-']

init_itr = 0
final_itr = 2
#final_itr = 30
samples_idx = None  # List of samples / None: all samples
max_traj_plots = None  # None, plot all
last_n_iters = None  # None, plot all iterations
sensed_joints = 'RA'
method = 'MDGPS_MDREPS'

iteration_data_options = {
    'plot_errors': True,
}

eta_color = 'black'
cs_color = 'red'
step_mult_color = 'red'
sample_list_cols = 3
plot_sample_list_max_min = False
plot_joint_limits = True
gps_num = 0

load_iteration_data = True

#iteration_data_options = [value for key, value in options.items() if key not in duality_data_options+policy_different_options]

gps_path = '/home/desteban/workspace/robolearn/scenarios/robolearn_log/' + gps_directory_name

iteration_data_list = list()
good_duality_info_list = list()
good_trajectories_info_list = list()
bad_duality_info_list = list()
bad_trajectories_info_list = list()
iteration_ids = list()
pol_sample_lists_costs = list()
pol_sample_lists_cost_compositions = list()

max_available_itr = None
for pp in range(init_itr, final_itr):
    if os.path.isfile(gps_path+'/' + str('gps%02d_' % gps_num) + method.upper() + '_iteration_data_itr_'+str('%02d' % pp)+'.pkl'):
        if os.path.isfile(gps_path+'/' + str('gps%02d_' % gps_num) + method.upper() + '_iteration_data_itr_'+str('%02d' % pp)+'.pkl'):
            max_available_itr = pp

if max_available_itr is not None:
    print("Max available iterations: %d" % max_available_itr)

    if last_n_iters is not None:
        init_itr = max(max_available_itr - last_n_iters + 1, 0)

    if max_traj_plots is not None:
        if max_available_itr > max_traj_plots:
            itr_to_load = np.linspace(init_itr, max_available_itr, max_traj_plots, dtype=np.uint8)
        else:
            itr_to_load = range(init_itr, max_available_itr+1)

    else:
        itr_to_load = range(init_itr, max_available_itr+1)

    print("Iterations to load: %s" % itr_to_load)
    for pp in itr_to_load:
        if os.path.isfile(gps_path+'/' + str('gps%02d_' % gps_num) + method.upper() + '_iteration_data_itr_'+str('%02d' % pp)+'.pkl'):
            print('Loading GPS iteration_data from iteration %d' % pp)
            iteration_data_list.append(pickle.load(open(gps_path+'/' + str('gps%02d_' % gps_num) + method.upper() +'_iteration_data_itr_'+str('%02d' % pp)+'.pkl',
                                                            'rb')))
        iteration_ids.append(pp)

if load_iteration_data:
    data_list_with_data = iteration_data_list
    if not data_list_with_data:
        raise AttributeError("No data has been loaded. Check that files exist")
    T = iteration_data_list[-1][-1].sample_list.get_actions(samples_idx).shape[1]
else:
    raise ValueError("NO data has been loaded!")

# total_cond = len(pol_sample_lists_costs[0])
total_itr = len(data_list_with_data)
total_cond = len(data_list_with_data[0])
colormap = plt.cm.rainbow  # nipy_spectral, Set1, Paired, winter

joint_limits = [bigman_params['joints_limits'][ii] for ii in bigman_params['joint_ids'][sensed_joints]]

if False:
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].sample_list.get_actions(samples_idx).shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('Actions | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        lines = list()
        labels = list()
        for itr in range(total_itr):
            actions = iteration_data_list[itr][cond].sample_list.get_actions(samples_idx)
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("Action %d" % (ii+1))
                    label = "itr %d" % iteration_ids[itr]
                    line = ax.plot(actions.mean(axis=0)[:, ii], label=label)[0]

                    if ii == 0:
                        lines.append(line)
                        labels.append(label)

                    if itr == 0:
                        ax.tick_params(axis='both', direction='in')
                        #ax.set_xlim([0, actions.shape[2]])
                        #ax.set_ylim([ymin, ymax])

                    if plot_sample_list_max_min:
                        ax.fill_between(range(actions.mean(axis=0).shape[0]), actions.min(axis=0)[:, ii],
                                        actions.max(axis=0)[:, ii], alpha=0.5)
                    # # One legend for each ax
                    # legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                    # legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        legend.get_frame().set_alpha(0.4)

if True:
    error_x = 0.05
    error_y = 0.05
    error_z = 0.05
    error_R = 0.05
    error_P = 0.05
    error_Y = 0.05
    max_error_drill = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

    indeces_drill = np.array([27, 28, 29, 30, 31, 32])

    for cond in range(total_cond):
        N = iteration_data_list[0][cond].sample_list.get_states(samples_idx).shape[-3]
        dData = iteration_data_list[0][cond].sample_list.get_states(samples_idx).shape[-1]
        fig, axs = plt.subplots(1, 1,)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('States | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        #for ii in range(axs.size):
        #    ax = axs[ii/sample_list_cols, ii % sample_list_cols]
        #    ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        lines = list()
        labels = list()

        errors = np.zeros(total_itr)

        for itr in range(total_itr):
            states = iteration_data_list[itr][cond].sample_list.get_states(samples_idx)
            all_zs = states[:, :, indeces_drill[-1]]
            print(all_zs.shape)
            error_count = 0
            for nn in range(N):
                print(N)
                print(nn)
                if np.any(all_zs[nn, :] > max_error_drill[-1]):
                    error_count += 1
            errors[itr] = error_count*100./N

        axs.plot(errors)

        ## One legend for all figures
        #legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        #legend.get_frame().set_alpha(0.4)

plt.show(block=False)

raw_input('Showing plots. Press a key to close...')
