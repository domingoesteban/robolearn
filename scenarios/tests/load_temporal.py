import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import pickle
import math
import os, sys
from robolearn.utils.plot_utils import plot_sample_list, plot_sample_list_distribution, lqr_forward, plot_3d_gaussian
from robolearn.algos.gps.gps_utils import IterationData
from robolearn.utils.iit.iit_robots_params import bigman_params
from robolearn.utils.traj_opt.traj_opt_utils import traj_distr_kl, traj_distr_kl_alt
import scipy.stats
from matplotlib.patches import Ellipse

# gps_directory_name = 'GPS_2017-09-01_15:22:55'  # Test MDGPS | Weekend
# gps_directory_name = 'GPS_2017-09-04_10:45:00'  # Test MDGPS | New cov_bad
# gps_directory_name = 'GPS_2017-09-08_18:07:44'  # Only Bad Friday 08/09
# gps_directory_name = 'GPS_2017-09-09_15:49:05'  # Normal Saturday 09/09
# gps_directory_name = 'GPS_2017-09-10_15:30:24'  # Normal Sunday 10/09 | new init_pos
# gps_directory_name = 'GPS_2017-09-10_19:10:07'  # G/B Sunday 10/09 | xi = 2.5, chi = 1.0
# gps_directory_name = 'GPS_2017-09-11_15:25:19'  # Bad | 2 hidden | xi = 5
# gps_directory_name = 'GPS_2017-09-12_07:01:16'  # Normal | 2 hidden | xi = 5
# gps_directory_name = 'GPS_2017-09-12_17:30:19'  # G/B | 2 hidden | xi=4, chi=1.0, tol_good=0.3
# gps_directory_name = 'GPS_2017-09-12_20:08:37'  # G/B | 2 hidden | xi=3, chi=1.0, tol_good=0.6
# gps_directory_name = 'GPS_2017-09-13_07:24:42'  # G/B | 2 hidden | xi=3, chi=1.5, tol_good=0.6 | Ts=0.02
# gps_directory_name = 'GPS_2017-09-13_13:24:56'  # NORMAL MDGPS PREV (New reset?)
# gps_directory_name = 'GPS_2017-09-13_21:58:52'  # G/B | 2 hidden | xi=3, chi=1.5, tol_good=0.6 |always
# gps_directory_name = 'GPS_2017-09-14_08:37:37'  # G/B | 2 hidden | xi=3, chi=1.5, tol_good=0.6 | only_traj | max_omega=1e16
# gps_directory_name = 'GPS_2017-09-14_09:59:56'  # G/B | 2 hidden | xi=3, chi=1.5, tol_good=0.6 | only_traj | max_omega=1e2
# gps_directory_name = 'GPS_2017-09-14_11:32:56'  # G/B | 2 hidden | xi=3, chi=1.5, tol_good=0.6 | only_traj | max_omega=2e1
# gps_directory_name = 'GPS_2017-09-14_12:35:39'  # G/B | 2 hidden | xi=3, chi=1.5, tol_good=0.6 | only_traj | max_omega=2e1 | max_nu=5e0 | init_nu=0.4 init_omega=0.4
# gps_directory_name = 'GPS_2017-09-14_13:10:46'  # G/B | 2 hidden | xi=3, chi=1.5, tol_good=0.6 | only_traj | max_omega=2e0 | max_nu=2e0 | init_nu=0.2 init_omega=0.2
# gps_directory_name = 'GPS_2017-09-14_15:46:29'  # G/B | 2 hidden | xi=3, chi=1.5, tol_good=0.6 | only_traj | max_omega=5e-1 | max_nu=5e-1 | init_nu=0.001 init_omega=0.001
# gps_directory_name = 'GPS_2017-09-14_17:05:37'  # As before but MDGPS
# gps_directory_name = 'GPS_2017-09-15_06:58:05'  # As before but B-MDGPS | base_kl_bad=3 -> This reach at iter
# gps_directory_name = 'GPS_2017-09-15_08:44:42'  # As before but B-MDGPS | base_kl_bad=3.5
# gps_directory_name = 'GPS_2017-09-15_12:33:18'  # As before but B-MDGPS | base_kl_bad=4.0 -> The local are very noisy and very rude movs, but at itr=07 it lifts al lot the arm but lands almost tho the point. Itr 40 almost reaches the pose.
# gps_directory_name = 'GPS_2017-09-15_18:22:46'  # As before but D-MDGPS | bse_kl_good=1.5 | good_tol=0.6
# gps_directory_name = 'GPS_2017-09-15_18:22:46'  # As before but D-MDGPS | bse_kl_good=1.0 | good_tol=0.6 | base_kl_bad=4.0??-->
# gps_directory_name = 'GPS_2017-09-15_18:22:46'  # As before but always good and always bad
# gps_directory_name = 'GPS_2017-09-16_02:15:49'  # B-MDGPS | base_kl_bad=4.5  | only_traj??
# gps_directory_name = 'GPS_2017-09-16_02:51:51'  # B-MDGPS | base_kl_bad=4.2  bse_kl_good=1.5?? | only_traj??

# MDGPS, bad-MDGPS, dual-MDGPS
#gps_directory_names = ['GPS_2017-09-13_13:24:56', 'GPS_2017-09-13_21:58:52', 'GPS_2017-09-13_21:58:52']

# PAPER
gps_directory_names = ['GPS_2017-09-14_17:05:37', 'GPS_2017-09-15_12:33:18', 'GPS_2017-09-15_18:22:46']

#gps_directory_names = ['GPS_2017-09-14_17:05:37', 'GPS_2017-09-16_02:51:51', 'GPS_2017-09-15_18:22:46']
#gps_directory_names = ['GPS_2017-09-20_09:49:20', 'GPS_2017-09-20_13:33:50', 'GPS_2017-09-15_18:22:46']
##gps_directory_names = ['GPS_2017-09-16_02:51:51', 'GPS_2017-09-16_02:51:51', 'GPS_2017-09-16_02:51:51']

gps_models_labels = ['MDGPS', 'B-MDGPS', 'D-MDGPS']
#gps_models_line_styles = [':', '--', '-']
gps_models_line_styles = ['-', '-', '-']
gps_models_colors = ['black', 'magenta', 'orange']
gps_models_markers = ['^', 's', 'D']

indeces_drill = np.array([27, 28, 29, 30, 31, 32])
#failure_limits = np.array([0.1, 0.1, 0.1, 0.02, 0.02, 0.02])
#failure_limits = np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02])
failure_limits = np.array([0.1, 0.1, 0.1, 0.02, 0.02, 0.02])
failure_limits = np.array([0.8, 0.8, 0.8, 0.13, 0.13, 0.13])
max_t_acum_error = 10

init_itr = 0
final_itr = 100
#final_itr = 30
samples_idx = [-1]  # List of samples / None: all samples
max_traj_plots = None  # None, plot all
last_n_iters = None  # 5  # None, plot all iterations
itr_to_load = range(25)#[4, 9, 14, 19, 24] #range(6)#[5, 20, 29, 32, 42]#range(50)  #None  # range(17, 21)#[10, 11, 12, 13, 14]#[1, 5, 9, 12, 14]#, 5, 6]#, 20, 30, 40]
#itr_to_load = [8, 10, 12, 14, 18]  # Best dmdgps
#itr_to_load = [17, 19, 21, 23, 24]  # Best bmdgps
#itr_to_load = [8, 10, 12, 14, 18]  # mdgps

sensed_joints = 'RA'
method = 'MDGPS_MDREPS'

options = {
    'plot_eta': False,
    'plot_nu': False,
    'plot_omega': False,
    'plot_step_mult': False,  # If linearized policy(then NN policy) is worse, epsilon is reduced.
    'plot_cs': False,
    'plot_sample_list_actions': False,
    'plot_sample_list_states': False,
    'plot_sample_list_obs': False,
    'plot_sample_list_actions_dual': False,
    'plot_policy_output': False,
    'plot_policy_actions': False,
    'plot_policy_states': False,
    'plot_policy_obs': False,
    'plot_traj_distr': False,
    'plot_3d_traj': False,
    'plot_3d_duality_traj': False,
    # Important
    'plot_duality_influence': True,
    'plot_duality_traj_distr': False,
    'plot_policy_costs': False,
    'plot_3d_pol_traj': False,
    'plot_train_errors': False,
    #'plot_duality_influence': False,
    #'plot_duality_traj_distr': False,
    #'plot_policy_costs': False,
    #'plot_3d_pol_traj': True,
    #'plot_train_errors': False,
}

cs_color = 'red'
step_mult_color = 'red'
sample_list_cols = 3
plot_sample_list_max_min = False
plot_joint_limits = True
gps_num = 0
total_gps = len(gps_directory_names)


duality_data_options = ['plot_duality_traj_distr', 'plot_sample_list_actions_dual', 'plot_3d_duality_traj', 'plot_duality_influence']
policy_different_options = ['plot_policy_costs']
iteration_data_options = [key for key, value in options.items() if key not in duality_data_options+policy_different_options]

load_iteration_data = any([options[key] for key in iteration_data_options])
load_duality_data = any([options[key] for key in duality_data_options])
load_policy_different_data = any([options[key] for key in policy_different_options])

#iteration_data_options = [value for key, value in options.items() if key not in duality_data_options+policy_different_options]


iteration_data_list = [list() for _ in range(total_gps)]
good_duality_info_list = [list() for _ in range(total_gps)]
good_trajectories_info_list = [list() for _ in range(total_gps)]
bad_duality_info_list = [list() for _ in range(total_gps)]
bad_trajectories_info_list = [list() for _ in range(total_gps)]
iteration_ids = [list() for _ in range(total_gps)]
pol_sample_lists_costs = [list() for _ in range(total_gps)]
pol_sample_lists_cost_compositions = [list() for _ in range(total_gps)]

max_available_itr = None

for gps, gps_directory_name in enumerate(gps_directory_names):
    gps_path = '/home/desteban/workspace/robolearn/scenarios/robolearn_log/' + gps_directory_name
    for pp in range(init_itr, final_itr):
        if os.path.isfile(gps_path+'/' + str('gps%02d_' % gps_num) + method.upper() + '_iteration_data_itr_'+str('%02d' % pp)+'.pkl'):
            if os.path.isfile(gps_path+'/' + str('gps%02d_' % gps_num) + method.upper() + '_iteration_data_itr_'+str('%02d' % pp)+'.pkl'):
                max_available_itr = pp

    if max_available_itr is not None:
        print("Max available iterations: %d" % max_available_itr)

        if itr_to_load is None:
            if last_n_iters is not None:
                init_itr = max(max_available_itr - last_n_iters + 1, 0)

            if max_traj_plots is not None:
                if max_available_itr > max_traj_plots:
                    itr_to_load = np.linspace(init_itr, max_available_itr, max_traj_plots, dtype=np.uint8)
                else:
                    itr_to_load = range(init_itr, max_available_itr+1)

            else:
                itr_to_load = range(init_itr, max_available_itr+1)

        print("Iterations to load in %s: %s" % (gps_directory_name, itr_to_load))
        for pp in itr_to_load:
            if load_iteration_data or load_duality_data or load_policy_different_data:
                if os.path.isfile(gps_path+'/' + str('gps%02d_' % gps_num) + method.upper() + '_iteration_data_itr_'+str('%02d' % pp)+'.pkl'):
                    print('Loading GPS iteration_data from iteration %d' % pp)
                    iteration_data_list[gps].append(pickle.load(open(gps_path+'/' + str('gps%02d_' % gps_num) + method.upper() +'_iteration_data_itr_'+str('%02d' % pp)+'.pkl',
                                                                'rb')))
                    iteration_ids[gps].append(pp+1)
            if load_duality_data:
                if os.path.isfile(gps_path+'/' + str('gps%02d_' % gps_num) + 'bad_duality_info_itr_'+str('%02d' % pp)+'.pkl'):
                    print('Loading GPS good_data from iteration %d' % pp)
                    bad_duality_info_list[gps].append(pickle.load(open(gps_path+'/' + str('gps%02d_' % gps_num) + 'bad_duality_info_itr_'+str('%02d' % pp)+'.pkl',
                                                                  'rb')))
                    good_duality_info_list[gps].append(pickle.load(open(gps_path+'/' + str('gps%02d_' % gps_num) + 'good_duality_info_itr_'+str('%02d' % pp)+'.pkl',
                                                                   'rb')))
                    bad_trajectories_info_list[gps].append(pickle.load(open(gps_path+'/' + str('gps%02d_' % gps_num) + 'bad_trajectories_info_itr_'+str('%02d' % pp)+'.pkl',
                                                                       'rb')))
                    good_trajectories_info_list[gps].append(pickle.load(open(gps_path+'/' + str('gps%02d_' % gps_num) + 'good_trajectories_info_itr_'+str('%02d' % pp)+'.pkl',
                                                                        'rb')))
            if load_policy_different_data:
                if os.path.isfile(gps_path+'/' + str('gps%02d_' % gps_num)+'pol_sample_cost_itr_'+str('%02d' % pp)+'.pkl'):
                    print('Loading policy sample cost from iteration %d' % pp)
                    pol_sample_lists_costs[gps].append(pickle.load(open(gps_path+'/' + str('gps%02d_' % gps_num)+'pol_sample_cost_itr_'+str('%02d' % pp)+'.pkl', 'rb')))
                if os.path.isfile(gps_path+'/' + str('gps%02d_' % gps_num)+'pol_sample_cost_composition_itr_'+str('%02d' % pp)+'.pkl'):
                    print('Loading policy sample cost composition from iteration %d' % pp)
                    pol_sample_lists_cost_compositions[gps].append(pickle.load(open(gps_path+'/' + str('gps%02d_' % gps_num)+'pol_sample_cost_composition_itr_'+str('%02d' % pp)+'.pkl', 'rb')))

            # if load_iteration_data or load_duality_data or load_policy_different_data:
            #     iteration_ids[gps].append(pp)  --> Now it is done before


if load_iteration_data:
    data_list_with_data = iteration_data_list
    if not data_list_with_data:
        raise AttributeError("No data has been loaded. Check that files exist")
    T = iteration_data_list[0][-1][-1].sample_list.get_actions(samples_idx).shape[1]
elif load_duality_data:
    data_list_with_data = bad_duality_info_list
    if not data_list_with_data:
        raise AttributeError("No data has been loaded. Check that files exist")
    T = iteration_data_list[0][-1][-1].sample_list.get_actions(samples_idx).shape[1]
elif load_policy_different_data:
    data_list_with_data = pol_sample_lists_costs
else:
    raise ValueError("NO data has been loaded!")

# total_cond = len(pol_sample_lists_costs[0])
total_itr = len(data_list_with_data[0])
total_cond = len(data_list_with_data[0][0])
colormap = plt.cm.rainbow  # nipy_spectral, Set1, Paired, winter

joint_limits = [bigman_params['joints_limits'][ii] for ii in bigman_params['joint_ids'][sensed_joints]]

if options['plot_eta']:
    for cond in range(total_cond):
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Eta values | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        colormap_list = [colormap(i) for i in np.linspace(0, 1, total_gps)]
        ax.set_prop_cycle('color', colormap_list)
        lines = [None for _ in range(total_gps)]
        labels = [None for _ in range(total_gps)]
        for gps in range(total_gps):
            etas = np.zeros(total_itr)
            for itr in range(total_itr):
                etas[itr] = iteration_data_list[gps][itr][cond].eta
            lines[gps] = ax.plot(range(1, total_itr+1), etas, color=colormap_list[gps],
                                 linestyle=gps_models_line_styles[gps])[0]
            labels[gps] = gps_models_labels[gps]
        ax.set_title('Eta values | Condition %d' % cond)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        legend.get_frame().set_alpha(0.4)

if options['plot_nu']:
    for cond in range(total_cond):
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Nu values | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        colormap_list = [colormap(i) for i in np.linspace(0, 1, total_gps)]
        ax.set_prop_cycle('color', colormap_list)
        lines = [None for _ in range(total_gps)]
        labels = [None for _ in range(total_gps)]
        nus = np.zeros(total_itr)
        for gps in range(total_gps):
            for itr in range(total_itr):
                nus[itr] = iteration_data_list[gps][itr][cond].nu
            lines[gps] = ax.plot(range(1, total_itr+1), nus, color=colormap_list[gps],
                                 linestyle=gps_models_line_styles[gps])[0]
            labels[gps] = gps_models_labels[gps]
        ax.set_title('Nu values | Condition %d' % cond)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        legend.get_frame().set_alpha(0.4)

if options['plot_omega']:
    for cond in range(total_cond):
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Omega values | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        colormap_list = [colormap(i) for i in np.linspace(0, 1, total_gps)]
        ax.set_prop_cycle('color', colormap_list)
        lines = [None for _ in range(total_gps)]
        labels = [None for _ in range(total_gps)]
        omegas = np.zeros(total_itr)
        for gps in range(total_gps):
            for itr in range(total_itr):
                omegas[itr] = iteration_data_list[gps][itr][cond].omega
            lines[gps] = ax.plot(range(1, total_itr+1), omegas, color=colormap_list[gps],
                                 linestyle=gps_models_line_styles[gps])[0]
            labels[gps] = gps_models_labels[gps]
        ax.set_title('Omega values | Condition %d' % cond)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

if options['plot_step_mult']:
    for cond in range(total_cond):
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Step multiplier | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        etas = np.zeros(total_itr)
        for itr in range(total_itr):
            etas[itr] = iteration_data_list[itr][cond].step_mult
        ax.set_title('Step multiplier | Condition %d' % cond)
        ax.plot(range(1, total_itr+1), etas, color=eta_color)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

if options['plot_cs']:
    for cond in range(total_cond):
        fig, ax = plt.subplots(1, 1)
        fig.canvas.set_window_title('Samples Costs | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        mean_costs = np.zeros(total_itr)
        max_costs = np.zeros(total_itr)
        min_costs = np.zeros(total_itr)
        std_costs = np.zeros(total_itr)
        for itr in range(total_itr):
            samples_cost_sum = iteration_data_list[itr][cond].cs.sum(axis=1)
            mean_costs[itr] = samples_cost_sum.mean()
            max_costs[itr] = samples_cost_sum.max()
            min_costs[itr] = samples_cost_sum.min()
            std_costs[itr] = samples_cost_sum.std()
        ax.set_title('Samples Costs | Condition %d' % cond)
        ax.plot(range(1, total_itr+1), mean_costs, color=cs_color)
        ax.fill_between(range(1, total_itr+1), min_costs, max_costs, alpha=0.5, color=cs_color)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

if options['plot_sample_list_actions_dual']:
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

if options['plot_sample_list_actions']:
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

if options['plot_sample_list_states']:
    for cond in range(total_cond):
        dData = iteration_data_list[0][0][cond].sample_list.get_states(samples_idx).shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('States | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        lines = list()
        labels = list()
        for itr in range(total_itr):
            states = iteration_data_list[0][itr][cond].sample_list.get_states(samples_idx)
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("State %d" % (ii+1))
                    label = "itr %d" % iteration_ids[0][itr]
                    line = ax.plot(states.mean(axis=0)[:, ii], label=label)[0]

                    if ii == 0:
                        lines.append(line)
                        labels.append(label)

                    if itr == 0:
                        ax.tick_params(axis='both', direction='in')

                    if plot_sample_list_max_min:
                        ax.fill_between(range(states.mean(axis=0).shape[0]), states.min(axis=0)[:, ii],
                                        states.max(axis=0)[:, ii], alpha=0.5)
                    # # One legend for each ax
                    # legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                    # legend.get_frame().set_alpha(0.4)

                    if plot_joint_limits and ii < len(joint_limits):
                        ax.plot(np.tile(joint_limits[ii][0], [T]), color='black', linestyle='--', linewidth=1)
                        ax.plot(np.tile(joint_limits[ii][1], [T]), color='black', linestyle='--', linewidth=1)
                else:
                    plt.setp(ax, visible=False)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        legend.get_frame().set_alpha(0.4)

if options['plot_sample_list_obs']:
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].sample_list.get_obs(samples_idx).shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('Observations | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        lines = list()
        labels = list()
        for itr in range(total_itr):
            obs = iteration_data_list[itr][cond].sample_list.get_obs(samples_idx)
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("Observation %d" % (ii+1))
                    label = "itr %d" % iteration_ids[itr]
                    line = ax.plot(obs.mean(axis=0)[:, ii], label=label)[0]

                    if ii == 0:
                        lines.append(line)
                        labels.append(label)

                    if itr == 0:
                        ax.tick_params(axis='both', direction='in')

                    if plot_sample_list_max_min:
                        ax.fill_between(range(states.mean(axis=0).shape[0]), states.min(axis=0)[:, ii],
                                        states.max(axis=0)[:, ii], alpha=0.5)
                    # # One legend for each ax
                    # legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                    # legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        legend.get_frame().set_alpha(0.4)

if options['plot_policy_output']:
    pol_sample_to_vis = -1
    pol_confidence = 0.95
    plot_confidence_interval = False
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].pol_info.pol_mu.shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title("Policy's Actions | Condition %d" % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])
        for itr in range(total_itr):
            mus = iteration_data_list[itr][cond].pol_info.pol_mu[pol_sample_to_vis]
            sigs = iteration_data_list[itr][cond].pol_info.pol_sig[pol_sample_to_vis]
            mins = np.zeros_like(mus)
            maxs = np.zeros_like(mus)
            for tt in range(mins.shape[0]):
                for dd in range(mins.shape[1]):
                    mins[tt, dd], maxs[tt, dd] = scipy.stats.norm.interval(pol_confidence,
                                                                           loc=mus[tt, dd],
                                                                           scale=sigs[tt, dd, dd])
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("Action %d" % (ii+1))
                    ax.plot(mus[:, ii], label=("itr %d" % iteration_ids[itr]))
                    if plot_confidence_interval:
                        ax.fill_between(range(mus.shape[0]), mins[:, ii], maxs[:, ii], alpha=0.5)
                    legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                    legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

if options['plot_policy_actions']:
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].pol_info.policy_samples.get_actions(samples_idx).shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('Policy Actions | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        lines = list()
        labels = list()
        for itr in range(total_itr):
            actions = iteration_data_list[itr][cond].pol_info.policy_samples.get_actions(samples_idx)
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

if options['plot_policy_states']:
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].pol_info.policy_samples.get_states(samples_idx).shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('Policy States | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        lines = list()
        labels = list()
        for itr in range(total_itr):
            states = iteration_data_list[itr][cond].pol_info.policy_samples.get_states(samples_idx)
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("State %d" % (ii+1))
                    label = "itr %d" % iteration_ids[itr]
                    line = ax.plot(states.mean(axis=0)[:, ii], label=label)[0]

                    if ii == 0:
                        lines.append(line)
                        labels.append(label)

                    if itr == 0:
                        ax.tick_params(axis='both', direction='in')

                    if plot_sample_list_max_min:
                        ax.fill_between(range(states.mean(axis=0).shape[0]), states.min(axis=0)[:, ii],
                                        states.max(axis=0)[:, ii], alpha=0.5)
                        # # One legend for each ax
                        # legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                        # legend.get_frame().set_alpha(0.4)

                    if plot_joint_limits and ii < len(joint_limits):
                        ax.plot(np.tile(joint_limits[ii][0], [T]), color='black', linestyle='--', linewidth=1)
                        ax.plot(np.tile(joint_limits[ii][1], [T]), color='black', linestyle='--', linewidth=1)
                else:
                    plt.setp(ax, visible=False)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        legend.get_frame().set_alpha(0.4)

if options['plot_policy_obs']:
    for cond in range(total_cond):
        dData = iteration_data_list[0][cond].pol_info.policy_samples.get_obs(samples_idx).shape[-1]
        fig, axs = plt.subplots(int(math.ceil(float(dData)/sample_list_cols)), sample_list_cols)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('Policy Observations | Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        for ii in range(axs.size):
            ax = axs[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        lines = list()
        labels = list()
        for itr in range(total_itr):
            obs = iteration_data_list[itr][cond].pol_info.policy_samples.get_obs(samples_idx)
            for ii in range(axs.size):
                ax = axs[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dData:
                    ax.set_title("Observation %d" % (ii+1))
                    label = "itr %d" % iteration_ids[itr]
                    line = ax.plot(obs.mean(axis=0)[:, ii], label=label)[0]

                    if ii == 0:
                        lines.append(line)
                        labels.append(label)

                    if itr == 0:
                        ax.tick_params(axis='both', direction='in')

                    if plot_sample_list_max_min:
                        ax.fill_between(range(states.mean(axis=0).shape[0]), states.min(axis=0)[:, ii],
                                        states.max(axis=0)[:, ii], alpha=0.5)
                    # # One legend for each ax
                    # legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                    # legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0., borderaxespad=0.)
        legend.get_frame().set_alpha(0.4)

if options['plot_traj_distr']:
    traj_distr_confidence = 0.95
    plot_confidence_interval = False
    plot_legend = True
    for cond in range(total_cond):
        dX = iteration_data_list[-1][cond].traj_distr.dX
        dU = iteration_data_list[-1][cond].traj_distr.dU
        fig_act, axs_act = plt.subplots(int(math.ceil(float(dU)/sample_list_cols)), sample_list_cols)
        fig_act.subplots_adjust(hspace=0)
        fig_act.canvas.set_window_title("Trajectory Distribution's Actions | Condition %d" % cond)
        fig_act.set_facecolor((1, 1, 1))
        fig_state, axs_state = plt.subplots(int(math.ceil(float(dX)/sample_list_cols)), sample_list_cols)
        fig_state.subplots_adjust(hspace=0)
        fig_state.canvas.set_window_title("Trajectory Distribution's States | Condition %d" % cond)
        fig_state.set_facecolor((1, 1, 1))

        for ii in range(axs_act.size):
            ax = axs_act[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])
        for ii in range(axs_state.size):
            ax = axs_state[ii/sample_list_cols, ii % sample_list_cols]
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])

        for itr in range(total_itr):
            traj_distr = iteration_data_list[itr][cond].traj_distr
            traj_info = iteration_data_list[itr][cond].traj_info

            mu, sigma = lqr_forward(traj_distr, traj_info)
            T = traj_distr.T
            dU = traj_distr.dU
            dX = traj_distr.dX
            x_idxs = range(dX)
            u_idxs = range(dX, dX+dU)
            mins = np.zeros_like(mu)
            maxs = np.zeros_like(mu)
            if plot_confidence_interval:
                for tt in range(T):
                    sigma_diag = np.diag(sigma[tt, :, :])
                    mins[tt, :], maxs[tt, :] = scipy.stats.norm.interval(traj_distr_confidence, loc=mu[tt, :],
                                                                         scale=sigma_diag[:])

            for ii in range(axs_act.size):
                ax = axs_act[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dU:
                    ax.set_title("Action %d" % (ii+1))
                    ax.plot(mu[:, u_idxs[ii]], label=("itr %d" % iteration_ids[itr]))
                    if plot_confidence_interval:
                        ax.fill_between(T, mins[:, ii], maxs[:, ii], alpha=0.5)
                    if plot_legend:
                        legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                        legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

            for ii in range(axs_state.size):
                ax = axs_state[ii/sample_list_cols, ii % sample_list_cols]
                if ii < dX:
                    ax.set_title("State %d" % (ii+1))
                    ax.plot(mu[:, x_idxs[ii]], label=("itr %d" % iteration_ids[itr]))
                    if plot_confidence_interval:
                        ax.fill_between(T, mins[:, ii], maxs[:, ii], alpha=0.5)
                    if plot_legend:
                        legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                        legend.get_frame().set_alpha(0.4)
                else:
                    plt.setp(ax, visible=False)

if options['plot_duality_influence']:
    linewidth = 3.0
    state_to_plot_idx = 2
    traj_distr_confidence = 0.95
    plot_confidence_interval = False  # True
    alpha_conf_int = 0.3
    plot_legend = True
    max_var = .5  # None, do not fix variance
    for itr in range(1, total_itr):
        gps = -1
    
        fig_act, ax = plt.subplots(1, 1)
        fig_act.canvas.set_window_title("Dualist Constraints Influence %02d" % itr)
        ii = state_to_plot_idx
        cond = 0
        dX = iteration_data_list[0][0][0].traj_distr.dX
        T = iteration_data_list[0][0][0].pol_info.policy_samples.get_states().shape[1]
        time = np.arange(0, 5, 0.02)
        x_idxs = range(dX)


        traj_distr = iteration_data_list[gps][itr][cond].traj_distr
        traj_info = iteration_data_list[gps][itr][cond].traj_info

        mu, sigma = lqr_forward(traj_distr, traj_info)

        good_distr = good_duality_info_list[gps][itr-1][cond].traj_dist
        good_traj_info = good_trajectories_info_list[gps][itr-1][cond]
        bad_distr = bad_duality_info_list[gps][itr-1][cond].traj_dist
        bad_traj_info = bad_trajectories_info_list[gps][itr-1][cond]

        prev_traj_distr = iteration_data_list[gps][itr-1][cond].traj_distr
        prev_traj_info = iteration_data_list[gps][itr-1][cond].traj_info

        mu_good, sigma_good = lqr_forward(good_distr, good_traj_info)
        mu_bad, sigma_bad = lqr_forward(bad_distr, bad_traj_info)
        mu_prev, sigma_prev = lqr_forward(prev_traj_distr, prev_traj_info)

        kl_div_good_bad = traj_distr_kl_alt(mu_good, sigma_good, good_distr, bad_distr, tot=True)
        print("KL_div(g||b): %f" % kl_div_good_bad)

        #ax.plot(time, mu_prev[:, x_idxs[ii]], label=("Prev traj."), zorder=9, color='blue', linewidth=linewidth)
        ax.plot(time, mu_good[:, x_idxs[ii]], label=("Good traj."), zorder=7, color='green', linewidth=linewidth)
        ax.plot(time, mu_bad[:, x_idxs[ii]], label=("Bad traj."), zorder=8, color='red', linewidth=linewidth)

        ax.plot(time, mu[:, x_idxs[ii]], label=("Opt. traj."), zorder=10, color=gps_models_colors[gps],
                linestyle=gps_models_line_styles[gps], linewidth=linewidth)

        ax.set_xlabel("Time (s)", fontsize=30, weight='bold')
        ax.set_ylabel("Shoulder Yaw Angle (rad)", fontsize=30, weight='bold')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        #ax.set_xticks(range(min_iteration, max_iteration+1, 5))
        ax.set_xlim(0, 5)
        ax.set_ylim(-0.6, 0.6)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        if plot_legend:
            #legend = ax.legend(loc='lower right', ncol=1, fontsize=20)
            #legend = ax.legend(bbox_to_anchor=(1.00, 1), loc=2, borderaxespad=0., fontsize=20)
            legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.5, fontsize=20)
            #legend = ax.legend(loc='best', fontsize='x-small', borderaxespad=0.)
            #legend.get_frame().set_alpha(0.4)
        
        plt.show(block=False)
        raw_input('press a key')


if options['plot_duality_traj_distr']:
    traj_distr_confidence = 0.95
    plot_confidence_interval = False  # True
    alpha_conf_int = 0.3
    plot_legend = True
    max_var = .5  # None, do not fix variance

    for cond in range(total_cond):
        dX = iteration_data_list[0][-1][cond].traj_distr.dX
        dU = iteration_data_list[0][-1][cond].traj_distr.dU

        for gps in range(total_gps):
            fig_act, axs_act = plt.subplots(int(math.ceil(float(dU)/sample_list_cols)), sample_list_cols)
            fig_act.subplots_adjust(hspace=0)
            fig_act.canvas.set_window_title("Trajectory Distribution Actions for %s Condition %d" % (gps_models_labels[gps], cond))
            fig_act.set_facecolor((1, 1, 1))
            fig_state, axs_state = plt.subplots(int(math.ceil(float(dX)/sample_list_cols)), sample_list_cols)
            fig_state.subplots_adjust(hspace=0)
            fig_state.canvas.set_window_title("Trajectory Distribution States for %s Condition %d" % (gps_models_labels[gps], cond))
            fig_state.set_facecolor((1, 1, 1))

            for ii in range(axs_act.size):
                ax = axs_act[ii/sample_list_cols, ii % sample_list_cols]
                #ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, 3*total_itr)])
                ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, 4)]) # prev, current, prev_good, prev_bad
            for ii in range(axs_state.size):
                ax = axs_state[ii/sample_list_cols, ii % sample_list_cols]
                #ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, 3*total_itr)])
                ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, 4)]) # prev, current, prev_good, prev_bad

            #for itr in range(total_itr):
            for itr in [1]:
                traj_distr = iteration_data_list[gps][itr][cond].traj_distr
                traj_info = iteration_data_list[gps][itr][cond].traj_info

                good_distr = good_duality_info_list[gps][itr-1][cond].traj_dist
                good_traj_info = good_trajectories_info_list[gps][itr-1][cond]
                bad_distr = bad_duality_info_list[gps][itr-1][cond].traj_dist
                bad_traj_info = bad_trajectories_info_list[gps][itr-1][cond]

                prev_traj_distr = iteration_data_list[gps][itr-1][cond].traj_distr
                prev_traj_info = iteration_data_list[gps][itr-1][cond].traj_info

                mu, sigma = lqr_forward(traj_distr, traj_info)
                mu_good, sigma_good = lqr_forward(good_distr, good_traj_info)
                mu_bad, sigma_bad = lqr_forward(bad_distr, bad_traj_info)
                mu_prev, sigma_prev = lqr_forward(prev_traj_distr, prev_traj_info)

                kl_div_good_bad = traj_distr_kl_alt(mu_good, sigma_good, good_distr, bad_distr, tot=True)
                print("KL_div(g||b): %f" % kl_div_good_bad)

                T = traj_distr.T
                dU = traj_distr.dU
                dX = traj_distr.dX
                x_idxs = range(dX)
                u_idxs = range(dX, dX+dU)
                if plot_confidence_interval:
                    mins = np.zeros_like(mu)
                    maxs = np.zeros_like(mu)
                    mins_good = np.zeros_like(mu_good)
                    maxs_good = np.zeros_like(mu_good)
                    mins_bad = np.zeros_like(mu_bad)
                    maxs_bad = np.zeros_like(mu_bad)
                    mins_prev = np.zeros_like(mu_prev)
                    maxs_prev = np.zeros_like(mu_prev)
                    for tt in range(T):
                        sigma_diag = np.diag(sigma[tt, :, :])
                        mins[tt, :], maxs[tt, :] = scipy.stats.norm.interval(traj_distr_confidence, loc=mu[tt, :],
                                                                             scale=sigma_diag[:])
                        sigma_good_diag = np.diag(sigma_good[tt, :, :])
                        if max_var is not None:
                            sigma_good_diag = np.min(np.vstack((sigma_good_diag, np.ones_like(sigma_good_diag)*max_var)), axis=0)
                        mins_good[tt, :], maxs_good[tt, :] = scipy.stats.norm.interval(traj_distr_confidence, loc=mu_good[tt, :],
                                                                             scale=sigma_good_diag[:])
                        sigma_bad_diag = np.diag(sigma_bad[tt, :, :])
                        if max_var is not None:
                            sigma_bad_diag = np.min(np.vstack((sigma_bad_diag, np.ones_like(sigma_bad_diag)*max_var)), axis=0)
                        mins_bad[tt, :], maxs_bad[tt, :] = scipy.stats.norm.interval(traj_distr_confidence, loc=mu_bad[tt, :],
                                                                                     scale=sigma_bad_diag[:])
                        sigma_prev_diag = np.diag(sigma_prev[tt, :, :])
                        mins_prev[tt, :], maxs_prev[tt, :] = scipy.stats.norm.interval(traj_distr_confidence, loc=mu_prev[tt, :],
                                                                             scale=sigma_prev_diag[:])

                for ii in range(axs_act.size):
                    ax = axs_act[ii/sample_list_cols, ii % sample_list_cols]
                    if ii < dU:
                        ax.set_title("Action %d" % (ii+1))
                        ax.plot(mu[:, u_idxs[ii]], label=("itr %d" % iteration_ids[gps][itr]), zorder=10)
                        ax.plot(mu_prev[:, u_idxs[ii]], label=("Prev itr %d" % iteration_ids[gps][itr]), zorder=9)
                        ax.plot(mu_good[:, u_idxs[ii]], label=("Good itr %d" % iteration_ids[gps][itr]), zorder=7)
                        ax.plot(mu_bad[:, u_idxs[ii]], label=("Bad itr %d" % iteration_ids[gps][itr]), zorder=8)
                        if plot_confidence_interval:
                            ax.fill_between(range(T), mins[:, u_idxs[ii]], maxs[:, u_idxs[ii]], alpha=alpha_conf_int, zorder=4)
                            ax.fill_between(range(T), mins_prev[:, u_idxs[ii]], maxs_prev[:, u_idxs[ii]], alpha=alpha_conf_int, zorder=3)
                            ax.fill_between(range(T), mins_good[:, u_idxs[ii]], maxs_good[:, u_idxs[ii]], alpha=alpha_conf_int, zorder=1)
                            ax.fill_between(range(T), mins_bad[:, u_idxs[ii]], maxs_bad[:, u_idxs[ii]], alpha=alpha_conf_int, zorder=2)
                        if plot_legend:
                            legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                            legend.get_frame().set_alpha(0.4)
                    else:
                        plt.setp(ax, visible=False)

                for ii in range(axs_state.size):
                    ax = axs_state[ii/sample_list_cols, ii % sample_list_cols]
                    if ii < dX:
                        ax.set_title("State %d" % (ii+1))
                        ax.plot(mu[:, x_idxs[ii]], label=("itr %d" % iteration_ids[gps][itr]), zorder=10)
                        ax.plot(mu_prev[:, x_idxs[ii]], label=("Prev itr %d" % iteration_ids[gps][itr]), zorder=9)
                        ax.plot(mu_good[:, x_idxs[ii]], label=("Good itr %d" % iteration_ids[gps][itr]), zorder=7)
                        ax.plot(mu_bad[:, x_idxs[ii]], label=("Bad itr %d" % iteration_ids[gps][itr]), zorder=8)
                        if plot_confidence_interval:
                            ax.fill_between(range(T), mins[:, x_idxs[ii]], maxs[:, x_idxs[ii]], alpha=alpha_conf_int, zorder=4)
                            ax.fill_between(range(T), mins_prev[:, x_idxs[ii]], maxs_prev[:, x_idxs[ii]], alpha=alpha_conf_int, zorder=3)
                            ax.fill_between(range(T), mins_good[:, x_idxs[ii]], maxs_good[:, x_idxs[ii]], alpha=alpha_conf_int, zorder=1)
                            ax.fill_between(range(T), mins_bad[:, x_idxs[ii]], maxs_bad[:, x_idxs[ii]], alpha=alpha_conf_int, zorder=2)
                        if plot_legend:
                            legend = ax.legend(loc='lower right', fontsize='x-small', borderaxespad=0.)
                            legend.get_frame().set_alpha(0.4)
                    else:
                        plt.setp(ax, visible=False)

if options['plot_3d_duality_traj']:
    distance_idxs = [24, 25, 26]  # NOT TO USE -1, -2, etc because it will get the mu and variance of u !!!
    linestyle = '-'
    linewidth = 1.0
    marker = None
    markersize = 5.0
    markeredgewidth = 1.0
    alpha = 1.0

    gauss_linestyle = ':'
    gauss_linewidth = 0.2
    gauss_marker = None
    gauss_markersize = 2.0
    gauss_markeredgewidth = 0.2
    gauss_alpha = 0.3

    plot_gaussian = False

    views = ['XY', 'XZ']

    #des_colormap = [colormap(i) for i in np.linspace(0, 1, total_itr)]
    des_colormap = [colormap(i) for i in np.linspace(0, 1, 4)]  # prev, cur, bad_prev, good_prev

    for cond in range(total_cond):
        fig_3d_traj = plt.figure()
        lines = list()
        labels = list()

        for vv, view in enumerate(views):
            ax_traj = fig_3d_traj.add_subplot(1, len(views), vv+1, projection='3d')
            ax_traj.set_prop_cycle('color', des_colormap)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            plot = ax_traj.plot([0], [0], [0], color='green', marker='o', markersize=10)

            fig_3d_traj.canvas.set_window_title("Expected Trajectories | Condition %d" % cond)
            ax_traj.set_xlabel('X')
            ax_traj.set_ylabel('Y')
            ax_traj.set_zlabel('Z')

            if view == 'XY':
                azim = 0.
                elev = 90.
            elif view == 'XZ':
                azim = 90.
                elev = 0.
            elif view == 'YZ':
                azim = 90.
                elev = 90.
            else:
                raise AttributeError("Wrong view %s" % view)

            ax_traj.view_init(elev=elev, azim=azim)

            #for itr in range(total_itr):
            for itr in [-1]:
                traj_distr = iteration_data_list[itr][cond].traj_distr
                traj_info = iteration_data_list[itr][cond].traj_info

                prev_traj_distr = iteration_data_list[itr-1][cond].traj_distr
                prev_traj_info = iteration_data_list[itr-1][cond].traj_info

                good_traj_distr = good_duality_info_list[itr-1][cond].traj_dist
                good_traj_info = good_trajectories_info_list[itr-1][cond]
                bad_traj_distr = bad_duality_info_list[itr-1][cond].traj_dist
                bad_traj_info = bad_trajectories_info_list[itr-1][cond]

                mu, sigma = lqr_forward(traj_distr, traj_info)
                mu_prev, sigma_prev = lqr_forward(prev_traj_distr, prev_traj_info)
                mu_good, sigma_good = lqr_forward(good_traj_distr, good_traj_info)
                mu_bad, sigma_bad = lqr_forward(bad_traj_distr, bad_traj_info)

                label = "itr %d" % iteration_ids[itr]
                label_prev = "Prev itr %d" % iteration_ids[itr]
                label_good = "Good itr %d" % iteration_ids[itr]
                label_bad = "Bad itr %d" % iteration_ids[itr]

                xs = np.linspace(5, 0, 100)
                plot = ax_traj.plot(mu[:, distance_idxs[0]],
                                       mu[:, distance_idxs[1]],
                                       zs=mu[:, distance_idxs[2]],
                                       linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize,
                                       markeredgewidth=markeredgewidth, alpha=alpha, color=des_colormap[0],
                                       label=label)[0]
                plot_prev = ax_traj.plot(mu_prev[:, distance_idxs[0]],
                                            mu_prev[:, distance_idxs[1]],
                                            zs=mu_prev[:, distance_idxs[2]],
                                            linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize,
                                            markeredgewidth=markeredgewidth, alpha=alpha, color=des_colormap[1],
                                            label=label)[0]
                plot_good = ax_traj.plot(mu_good[:, distance_idxs[0]],
                                            mu_good[:, distance_idxs[1]],
                                            zs=mu_good[:, distance_idxs[2]],
                                            linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize,
                                            markeredgewidth=markeredgewidth, alpha=alpha, color=des_colormap[2],
                                            label=label)[0]
                plot_bad = ax_traj.plot(mu_bad[:, distance_idxs[0]],
                                       mu_bad[:, distance_idxs[1]],
                                       zs=mu_bad[:, distance_idxs[2]],
                                       linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize,
                                       markeredgewidth=markeredgewidth, alpha=alpha, color=des_colormap[3],
                                       label=label)[0]

                if vv == 0:
                    lines.append(plot)
                    lines.append(plot_prev)
                    lines.append(plot_good)
                    lines.append(plot_bad)
                    labels.append(label)
                    labels.append(label_prev)
                    labels.append(label_good)
                    labels.append(label_bad)

                sigma_idx = np.ix_(distance_idxs, distance_idxs)
                plot_3d_gaussian(ax_traj, mu[:, distance_idxs], sigma[:, sigma_idx[0], sigma_idx[1]],
                                 sigma_axes=view, edges=100, linestyle=gauss_linestyle, linewidth=gauss_linewidth,
                                 color=des_colormap[0], alpha=gauss_alpha, label='',
                                 markeredgewidth=gauss_markeredgewidth, marker=marker, markersize=markersize)

                # plot_3d_gaussian(ax_traj, mu_prev[:, distance_idxs], sigma_prev[:, sigma_idx[0], sigma_idx[1]],
                #                  sigma_axes=view, edges=100, linestyle=gauss_linestyle, linewidth=gauss_linewidth,
                #                  color=des_colormap[1], alpha=gauss_alpha, label='',
                #                  markeredgewidth=gauss_markeredgewidth, marker=marker, markersize=markersize)

                if plot_gaussian:
                    plot_3d_gaussian(ax_traj, mu_good[:, distance_idxs], sigma_good[:, sigma_idx[0], sigma_idx[1]],
                                     sigma_axes=view, edges=100, linestyle=gauss_linestyle, linewidth=gauss_linewidth,
                                     color=des_colormap[2], alpha=gauss_alpha, label='',
                                     markeredgewidth=gauss_markeredgewidth, marker=marker, markersize=markersize)

                    plot_3d_gaussian(ax_traj, mu_bad[:, distance_idxs], sigma_bad[:, sigma_idx[0], sigma_idx[1]],
                                     sigma_axes=view, edges=100, linestyle=gauss_linestyle, linewidth=gauss_linewidth,
                                     color=des_colormap[3], alpha=gauss_alpha, label='',
                                     markeredgewidth=gauss_markeredgewidth, marker=marker, markersize=markersize)

                X = np.append(mu[:, distance_idxs[0]], 0)
                Y = np.append(mu[:, distance_idxs[1]], 0)
                Z = np.append(mu[:, distance_idxs[2]], 0)
                mid_x = (X.max() + X.min()) * 0.5
                mid_y = (Y.max() + Y.min()) * 0.5
                mid_z = (Z.max() + Z.min()) * 0.5
                max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

                ax_traj.set_xlim(mid_x - max_range, mid_x + max_range)
                ax_traj.set_ylim(mid_y - max_range, mid_y + max_range)
                ax_traj.set_zlim(mid_z - max_range, mid_z + max_range)

                X_prev = np.append(mu_prev[:, distance_idxs[0]], 0)
                Y_prev = np.append(mu_prev[:, distance_idxs[1]], 0)
                Z_prev = np.append(mu_prev[:, distance_idxs[2]], 0)
                mid_x_prev = (X_prev.max() + X_prev.min()) * 0.5
                mid_y_prev = (Y_prev.max() + Y_prev.min()) * 0.5
                mid_z_prev = (Z_prev.max() + Z_prev.min()) * 0.5
                max_range_prev = np.array([X_prev.max()-X_prev.min(), Y_prev.max()-Y_prev.min(), Z_prev.max()-Z_prev.min()]).max() / 2.0

                ax_traj.set_xlim(mid_x_prev - max_range_prev, mid_x_prev + max_range_prev)
                ax_traj.set_ylim(mid_y_prev - max_range_prev, mid_y_prev + max_range_prev)
                ax_traj.set_zlim(mid_z_prev - max_range_prev, mid_z_prev + max_range_prev)

                X_good = np.append(mu_good[:, distance_idxs[0]], 0)
                Y_good = np.append(mu_good[:, distance_idxs[1]], 0)
                Z_good = np.append(mu_good[:, distance_idxs[2]], 0)
                mid_x_good = (X_good.max() + X_good.min()) * 0.5
                mid_y_good = (Y_good.max() + Y_good.min()) * 0.5
                mid_z_good = (Z_good.max() + Z_good.min()) * 0.5
                max_range_good = np.array([X_good.max()-X_good.min(), Y_good.max()-Y_good.min(), Z_good.max()-Z_good.min()]).max() / 2.0

                ax_traj.set_xlim(mid_x_good - max_range_good, mid_x_good + max_range_good)
                ax_traj.set_ylim(mid_y_good - max_range_good, mid_y_good + max_range_good)
                ax_traj.set_zlim(mid_z_good - max_range_good, mid_z_good + max_range_good)

                X_bad = np.append(mu_bad[:, distance_idxs[0]], 0)
                Y_bad = np.append(mu_bad[:, distance_idxs[1]], 0)
                Z_bad = np.append(mu_bad[:, distance_idxs[2]], 0)
                mid_x_bad = (X_bad.max() + X_bad.min()) * 0.5
                mid_y_bad = (Y_bad.max() + Y_bad.min()) * 0.5
                mid_z_bad = (Z_bad.max() + Z_bad.min()) * 0.5
                max_range_bad = np.array([X_bad.max()-X_bad.min(), Y_bad.max()-Y_bad.min(), Z_bad.max()-Z_bad.min()]).max() / 2.0

                ax_traj.set_xlim(mid_x_bad - max_range_bad, mid_x_bad + max_range_bad)
                ax_traj.set_ylim(mid_y_bad - max_range_bad, mid_y_bad + max_range_bad)
                ax_traj.set_zlim(mid_z_bad - max_range_bad, mid_z_bad + max_range_bad)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0.)
        legend.get_frame().set_alpha(0.4)

if options['plot_3d_traj']:
    distance_idxs = [24, 25, 26]  # NOT TO USE -1, -2, etc because it will get the mu and variance of u !!!
    linestyle = '-'
    linewidth = 2.5
    marker = None
    markersize = 5.0
    markeredgewidth = 1.0
    alpha = 1.0

    gauss_linestyle = ':'
    gauss_linewidth = 0.2
    gauss_marker = None
    gauss_markersize = 2.0
    gauss_markeredgewidth = 0.2
    gauss_alpha = 0.3

    views = ['XY', 'XZ']

    des_colormap = [colormap(i) for i in np.linspace(0, 1, total_itr)]

    for cond in range(total_cond):
        fig_3d_traj = plt.figure()
        lines = list()
        labels = list()

        for vv, view in enumerate(views):
            ax_traj = fig_3d_traj.add_subplot(1, len(views), vv+1, projection='3d')
            ax_traj.set_prop_cycle('color', des_colormap)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            plot = ax_traj.plot([0], [0], [0], color='green', marker='o', markersize=10)

            fig_3d_traj.canvas.set_window_title("Expected Trajectories | Condition %d" % cond)
            ax_traj.set_xlabel('X')
            ax_traj.set_ylabel('Y')
            ax_traj.set_zlabel('Z')

            if view == 'XY':
                azim = 0.
                elev = 90.
            elif view == 'XZ':
                azim = 90.
                elev = 0.
            elif view == 'YZ':
                azim = 90.
                elev = 90.
            else:
                raise AttributeError("Wrong view %s" % view)

            ax_traj.view_init(elev=elev, azim=azim)

            for itr in range(total_itr):
                traj_distr = iteration_data_list[itr][cond].traj_distr
                traj_info = iteration_data_list[itr][cond].traj_info

                mu, sigma = lqr_forward(traj_distr, traj_info)

                label = "itr %d" % iteration_ids[itr]

                xs = np.linspace(5, 0, 100)
                plot = ax_traj.plot(mu[:, distance_idxs[0]],
                                       mu[:, distance_idxs[1]],
                                       zs=mu[:, distance_idxs[2]],
                                       linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize,
                                       markeredgewidth=markeredgewidth, alpha=alpha, color=des_colormap[itr],
                                       label=label)[0]

                if vv == 0:
                    lines.append(plot)
                    labels.append(label)

                sigma_idx = np.ix_(distance_idxs, distance_idxs)
                plot_3d_gaussian(ax_traj, mu[:, distance_idxs], sigma[:, sigma_idx[0], sigma_idx[1]],
                                 sigma_axes=view, edges=100, linestyle=gauss_linestyle, linewidth=gauss_linewidth,
                                 color=des_colormap[itr], alpha=gauss_alpha, label='',
                                 markeredgewidth=gauss_markeredgewidth)

                X = np.append(mu[:, distance_idxs[0]], 0)
                Y = np.append(mu[:, distance_idxs[1]], 0)
                Z = np.append(mu[:, distance_idxs[2]], 0)
                mid_x = (X.max() + X.min()) * 0.5
                mid_y = (Y.max() + Y.min()) * 0.5
                mid_z = (Z.max() + Z.min()) * 0.5
                max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

                ax_traj.set_xlim(mid_x - max_range, mid_x + max_range)
                ax_traj.set_ylim(mid_y - max_range, mid_y + max_range)
                ax_traj.set_zlim(mid_z - max_range, mid_z + max_range)

        # One legend for all figures
        legend = plt.figlegend(lines, labels, loc='lower center', ncol=5, labelspacing=0.)
        legend.get_frame().set_alpha(0.4)

if options['plot_3d_pol_traj']:
    last_n_iters = 5

    distance_idxs = [24, 25, 26]  # NOT TO USE -1, -2, etc because it will get the mu and variance of u !!!
    plot_type = '2d'  # 3d or 2d
    linestyle = '-'
    linestyle_failure = ':'
    linewidth = 3.0
    marker = None
    markersize = 5.0
    markeredgewidth = 1.0
    alpha = 1.0

    #ax_mins = [-0.15, -0.2, -0.2]
    #ax_maxs = [0.15, 0.3, 0.4]
    ax_mins = [-0.0, -0.2, -0.2]
    ax_maxs = [0.0, 0.3, 0.4]
    ax_ticks = [[-0.1, 0.0, 0.1, 0.2],
                [-0.1, 0.0, 0.1, 0.2],
                [-0.1, 0.0, 0.1, 0.2, 0.3]]

    ellipse_size = [0.05, 0.02, 0.1]


    background_dir = 'drill_icons'  # 'drill_icons'  #  'drill_3d', 'drill_icons', 'drill_icons_svg'
    image_format = 'png'  # 'png'
    fig_scale = 0.255/660.
    fig_size = np.array([700, 700, 700])*fig_scale

    size_x = 0.1
    size_y = 0.1
    size_z = 0.3
    relative_zero = np.array([0.00, -size_y/2-0.02, size_z/2+0.02])

    gauss_linestyle = ':'
    gauss_linewidth = 0.2
    gauss_marker = None
    gauss_markersize = 2.0
    gauss_markeredgewidth = 0.2
    gauss_alpha = 0.3

    views = ['YZ', 'XZ']
    #views = ['XZ']


    des_colormap = [colormap(i) for i in np.linspace(0, 1, total_itr)]
    if last_n_iters is not None:
        des_colormap[-last_n_iters:] = [colormap(i) for i in np.linspace(0, 1, last_n_iters)]

    samples_idx = -1

    #fig_3d_trajs = plt.figure()
    for cond in range(total_cond):
        for gps in range(total_gps):
            #fig_3d_traj = plt.figure()
            fig_3d_traj, ax_trajs = plt.subplots(1, len(views), sharex=False, sharey=True)

            #fig_3d_trajs = [plt.figure() for _ in range(len(views))]

            # ax_traj = fig_3d_traj.add_subplot(1, len(views), vv+1, projection='3d')
            lines = list()
            labels = list()

            fig_3d_traj.canvas.set_window_title("Expected Trajectories for %s Condition %d" % (gps_models_labels[gps], cond))

            if plot_type == '3d':
                projection = '3d'
            else:
                projection = None

            for vv, view in enumerate(views):
                #if vv > 0:
                #    ax_traj[vv] = fig_3d_traj.add_subplot(1, len(views), vv+1, projection=projection, sharey=ax_traj)
                #else:
                #    ax_traj[vv] = fig_3d_traj.add_subplot(1, len(views), vv+1, projection=projection)
                #ax_traj = fig_3d_trajs[vv].add_subplot(1, 1, 1, projection=projection)

                ax_trajs[vv].set_prop_cycle('color', des_colormap)
                if plot_type == '3d':
                    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
                    plot = ax_traj.plot([0], [0], [0], color='pink', marker='o', markersize=30)
                else:
                    #plot = ax_traj.plot([0], [0], color='green', marker='o', markersize=10)
                    #plot = ax_trajs[vv].plot([0], [0], color='lightgreen', marker='o', markersize=30, markeredgecolor='limegreen')
                    ax_names = ['X', 'Y', 'Z']
                    x_index = ax_names.index(view[0])
                    y_index = ax_names.index(view[1])
                    e = Ellipse(xy=[0, 0], width=ellipse_size[x_index], height=ellipse_size[y_index], angle=0)
                    ax_trajs[vv].add_artist(e)
                    e.set_facecolor('lightgreen')
                    e.set_edgecolor('limegreen')


                #fig_3d_trajs[vv].canvas.set_window_title("Expected Trajectories for %s| Condition %d | view %s" % (gps_models_labels[gps], cond, view))
                if plot_type == '3d':
                    ax_traj.set_xlabel('X')
                    ax_traj.set_ylabel('Y')
                    ax_traj.set_zlabel('Z')

                    if view == 'XY':
                        azim = 0.
                        elev = 90.
                    elif view == 'XZ':
                        azim = 90.
                        elev = 0.
                    elif view == 'YZ':
                        azim = 90.
                        elev = 90.
                    else:
                        raise AttributeError("Wrong view %s" % view)

                    ax_traj.view_init(elev=elev, azim=azim)
                else:
                    ax_names = ['X', 'Y', 'Z']
                    x_index = ax_names.index(view[0])
                    y_index = ax_names.index(view[1])
                    ax_trajs[vv].set_xlabel(ax_names[x_index] + ' (m)', fontsize=20, weight='bold')
                    ax_trajs[vv].set_ylabel(ax_names[y_index] + ' (m)', fontsize=20, weight='bold')
                    ax_trajs[vv].tick_params(axis='x', labelsize=15)
                    ax_trajs[vv].tick_params(axis='y', labelsize=15)

                    if vv > 0:
                        ax_trajs[vv].yaxis.label.set_visible(False)

                    # Image
                    if background_dir is not None:
                        print('Using background: /images/'+background_dir+'/'+view+'_plot.'+image_format)
                        current_path = os.path.dirname(os.path.realpath(__file__))
                        img = plt.imread(current_path+'/images/'+background_dir+'/'+view+'_plot.'+image_format)
                        # Image
                        # if view == 'YZ':
                        #     center_image = np.array([215, 680])
                        # elif view == 'XZ':
                        #     center_image = np.array([315, 680])
                        center_image = np.array([315, 215, 20])
                        #center_image = np.array([315, 215, 680])

                if last_n_iters is None or last_n_iters > len(iteration_ids[gps]):
                    iteration_range = range(len(iteration_ids[gps]))
                else:
                    iteration_range = range(-last_n_iters, 0)

                for itr in iteration_range:
                    # traj_distr = iteration_data_list[itr][cond].traj_distr
                    # traj_info = iteration_data_list[itr][cond].traj_info
                    # mu, sigma = lqr_forward(traj_distr, traj_info)

                    mu = iteration_data_list[gps][itr][cond].pol_info.policy_samples.get_states()[samples_idx, :, :]

                    if itr == iteration_range[0]:
                        init = mu[0, distance_idxs]
                        plot = ax_trajs[vv].plot([init[x_index]], [init[y_index]], color='pink', marker='o', markersize=10,
                                                 markeredgecolor='mediumvioletred')


                    T = mu.shape[0]
                    dX = mu.shape[1]
                    max_good = None
                    for t in range(T):
                        if np.any(np.abs(mu[t, indeces_drill]) > failure_limits):
                            max_good = t
                            break

                    label = "Itr. %d" % iteration_ids[gps][itr]

                    xs = np.linspace(5, 0, 100)
                    if plot_type == '3d':
                        #plot = ax_traj.plot(mu[:, distance_idxs[0]],
                        plot = ax_trajs[vv].plot(mu[:, distance_idxs[0]],
                                                 mu[:, distance_idxs[1]],
                                                 zs=mu[:, distance_idxs[2]],
                                                 linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize,
                                                 markeredgewidth=markeredgewidth, alpha=alpha, color=des_colormap[itr],
                                                 label=label)[0]

                    else:
                        #plot = ax_traj.plot(mu[:, distance_idxs[x_index]],
                        #print('Plotting itr %d' % itr)
                        plot = ax_trajs[vv].plot(mu[:max_good, distance_idxs[x_index]],
                                                 mu[:max_good, distance_idxs[y_index]],
                                                 linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize,
                                                 markeredgewidth=markeredgewidth, alpha=alpha, color=des_colormap[itr],
                                                 label=label)[0]


                        if max_good is not None:
                            ax_trajs[vv].plot(mu[max_good:, distance_idxs[x_index]],
                                              mu[max_good:, distance_idxs[y_index]],
                                              linestyle=linestyle_failure, linewidth=linewidth, marker=marker, markersize=markersize,
                                              markeredgewidth=markeredgewidth, alpha=alpha, color=des_colormap[itr],
                                              label=label)[0]

                        if background_dir is not None and itr == iteration_range[0]:
                            # extent=[left, right, bottom, top]
                            extent = [-relative_zero[x_index]-center_image[x_index]*fig_scale,
                                      -relative_zero[x_index]+fig_size[1]-center_image[x_index]*fig_scale,
                                      -relative_zero[y_index]-center_image[y_index]*fig_scale,
                                      -relative_zero[y_index]+fig_size[1]-center_image[y_index]*fig_scale]
                            #ax_traj.imshow(img, zorder=0, extent=extent)
                            ax_trajs[vv].imshow(img, zorder=0, extent=extent)

                        plot2 = ax_trajs[vv].plot(mu[-1, distance_idxs[x_index]],
                                                 mu[-1, distance_idxs[y_index]],
                                                 linestyle=linestyle, linewidth=linewidth, marker='P', markersize=10,
                                                 markeredgewidth=markeredgewidth, alpha=alpha, color=des_colormap[itr],
                                                 label=label)[0]


                    if vv == 0:
                        lines.append(plot)
                        labels.append(label)

                    # sigma_idx = np.ix_(distance_idxs, distance_idxs)
                    # plot_3d_gaussian(ax_traj, mu[:, distance_idxs], sigma[:, sigma_idx[0], sigma_idx[1]],
                    #                  sigma_axes=view, edges=100, linestyle=gauss_linestyle, linewidth=gauss_linewidth,
                    #                  color=des_colormap[itr], alpha=gauss_alpha, label='',
                    #                  markeredgewidth=gauss_markeredgewidth)

                    # Set axis limits
                    if plot_type == '3d':
                        X = np.append(mu[:, distance_idxs[0]], 0)
                        Y = np.append(mu[:, distance_idxs[1]], 0)
                        Z = np.append(mu[:, distance_idxs[2]], 0)
                        mid_x = (X.max() + X.min()) * 0.5
                        mid_y = (Y.max() + Y.min()) * 0.5
                        mid_z = (Z.max() + Z.min()) * 0.5
                        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

                        ax_traj.set_xlim(mid_x - max_range, mid_x + max_range)
                        ax_traj.set_ylim(mid_y - max_range, mid_y + max_range)
                        ax_traj.set_zlim(mid_z - max_range, mid_z + max_range)
                    else:
                        if background_dir is None:
                            X = np.append(mu[:, distance_idxs[x_index]], 0)
                            Y = np.append(mu[:, distance_idxs[y_index]], 0)
                            mid_x = (X.max() + X.min()) * 0.5
                            mid_y = (Y.max() + Y.min()) * 0.5
                            max_range = np.array([X.max()-X.min(), Y.max()-Y.min()]).max() / 2.0
                            #ax_traj.set_xlim(mid_x - max_range, mid_x + max_range)
                            #ax_traj.set_ylim(mid_y - max_range, mid_y + max_range)
                            ax_trajs[vv].set_xlim(min(-relative_zero[x_index], mid_x - max_range), max(-relative_zero[x_index], mid_x + max_range))
                            ax_trajs[vv].set_ylim(min(-relative_zero[y_index], mid_y - max_range), max(-relative_zero[y_index], mid_y + max_range))
                            plt.axis('equal')
                            plt.axis([min(-relative_zero[x_index], mid_x - max_range),
                                      max(-relative_zero[x_index], mid_x + max_range),
                                      min(-relative_zero[y_index], mid_y - max_range),
                                      max(-relative_zero[y_index], mid_y + max_range)])
                        else:
                            #ymin, ymax = ax_traj.get_ylim()np.
                            #xmin, xmax = ax_traj.get_xlim()
                            ax_trajs[vv].set_xlim(ax_mins[x_index], ax_maxs[x_index])
                            ax_trajs[vv].set_ylim(ax_mins[y_index], ax_maxs[y_index])
                            #print(np.arange(ax_mins[x_index], ax_maxs[x_index], 0.1))
                            #print(np.arange(ax_mins[y_index], ax_maxs[y_index], 0.1))
                            ax_trajs[vv].set_xticks(ax_ticks[x_index])
                            ax_trajs[vv].set_yticks(ax_ticks[y_index])

                plt.tight_layout()
                # One legend for all figures
                legend = plt.figlegend(lines, labels, loc='lower center', ncol=len(itr_to_load), labelspacing=0., fontsize=20)
                #legend = plt.figlegend(lines, labels, loc='center right', ncol=5, labelspacing=0.)
                #legend.get_frame().set_alpha(0.4)
                legend.get_frame().set_alpha(1.0)
                legend.get_frame().set_edgecolor('k')


if options['plot_policy_costs']:
    plots_type = 'iteration'  # 'iteration' or 'episode'
    include_last_T = False  # Only in iteration
    iteration_to_plot = -1
    plot_cost_types = False
    colormap = plt.cm.rainbow  # nipy_spectral, Set1, Paired, winter, rainbow

    total_cond = len(pol_sample_lists_costs[0][0])
    total_itr = len(pol_sample_lists_costs[0])

    temp_array = np.zeros(len(iteration_ids[-1]))

    if plots_type.lower() == 'iteration':
        #marker = 'o'
        marker = None
        for cond in range(total_cond):
            fig, ax = plt.subplots(1, 1)
            fig.subplots_adjust(hspace=0)
            fig.canvas.set_window_title('Policy Costs Condition %d' % cond)
            fig.set_facecolor((1, 1, 1))
            des_colormap = [colormap(i) for i in np.linspace(0, 1, total_gps)]

            lines = list()
            labels = list()

            min_iteration = np.inf
            max_iteration = -np.inf
            for gps in range(total_gps):
                mean_costs = np.zeros(len(iteration_ids[gps]))
                max_costs = np.zeros(len(iteration_ids[gps]))
                min_costs = np.zeros(len(iteration_ids[gps]))
                std_costs = np.zeros(len(iteration_ids[gps]))

                for itr in range(len(iteration_ids[gps])):
                    total_samples = len(pol_sample_lists_costs[gps][itr][cond])
                    samples_cost_sum = pol_sample_lists_costs[gps][itr][cond].sum(axis=1)

                    temp_array[itr] = samples_cost_sum[-1]

                    mean_costs[itr] = samples_cost_sum.mean()
                    max_costs[itr] = samples_cost_sum.max()
                    min_costs[itr] = samples_cost_sum.min()
                    std_costs[itr] = samples_cost_sum.std()

                    if iteration_ids[gps][itr] < min_iteration:
                        min_iteration = iteration_ids[gps][itr]

                    if iteration_ids[gps][itr] > max_iteration:
                        max_iteration = iteration_ids[gps][itr]

                line = ax.plot(iteration_ids[gps], mean_costs, marker=gps_models_markers[gps],
                               label=gps_models_labels[gps], linestyle=gps_models_line_styles[gps],
                               color=gps_models_colors[gps])[0]
                ax.fill_between(iteration_ids[gps], min_costs, max_costs, alpha=0.5, color=gps_models_colors[gps])
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                lines.append(line)
                labels.append(gps_models_labels[gps])


            #ax.set_title('Policy Costs | Condition %d' % cond)

                # Composition cost
                if plot_cost_types:
                    total_cost_types = len(pol_sample_lists_cost_compositions[-1][-1][-1])
                    mean_cost_types = np.zeros([len(iteration_ids[gps]), total_cost_types])
                    max_cost_types = np.zeros([len(iteration_ids[gps]), total_cost_types])
                    min_cost_types = np.zeros([len(iteration_ids[gps]), total_cost_types])
                    std_cost_types = np.zeros([len(iteration_ids[gps]), total_cost_types])
                    for itr in range(total_itr):
                        total_samples = len(pol_sample_lists_cost_compositions[itr][cond])
                        for c in range(total_cost_types):
                            cost_type_sum = np.zeros(total_samples)
                            for n in range(total_samples):
                                cost_type_sum[n] = np.sum(pol_sample_lists_cost_compositions[itr][cond][n][c])
                            mean_cost_types[itr, c] = cost_type_sum.mean()
                            max_cost_types[itr, c] = cost_type_sum.max()
                            min_cost_types[itr, c] = cost_type_sum.min()
                            std_cost_types[itr, c] = cost_type_sum.std()

                    for c in range(total_cost_types):
                        label = 'Cost type %d' % c
                        line = ax.plot(mean_cost_types[:, c], marker=marker, label=label)[0]
                        ax.fill_between(range(total_itr), min_cost_types[:, c], max_cost_types[:, c], alpha=0.5)
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        lines.append(line)
                        labels.append(label)

                print('%'*10)
                print('%d' % gps)
                #least_best_index = np.argpartition(temp_array, len(iteration_ids[gps]))[:len(iteration_ids[gps])]
                best_index = np.argpartition(temp_array, len(iteration_ids[gps])-1)[:len(iteration_ids[gps])-1]
                best_index = best_index[np.argsort(temp_array[best_index])]
                worst_index = np.argpartition(temp_array, -(len(iteration_ids[gps])-1))[-(len(iteration_ids[gps])-1):]
                worst_index = worst_index[np.argsort(-temp_array[worst_index])]
                print(best_index)
                print('%'*10)

            #ax.set_xticks(range(min_iteration, max_iteration+1))
            ax.set_xlim(min_iteration, max_iteration)
            ax.set_xticks(range(0, 26, 5))
            ax.set_xlim(0, 25)
            ax.set_xlabel("Iterations", fontsize=30, weight='bold')
            ax.set_ylabel("Average Cost", fontsize=30, weight='bold')
            ax.tick_params(axis='x', labelsize=25)
            ax.tick_params(axis='y', labelsize=25)

            #legend = ax.legend(loc='best', ncol=1, fontsize=20)
            legend = ax.legend(ncol=1, fontsize=25)
            #legend = plt.figlegend(lines, labels, loc='center right', ncol=1, labelspacing=0., borderaxespad=1.)
            #legend.get_frame().set_alpha(0.4)

    else:
        T = pol_sample_lists_costs[0][0].shape[1]
        if include_last_T is False:
            T = T - 1

        if iteration_to_plot is not None:
            if iteration_to_plot == -1:
                iteration_to_plot = total_itr - 1
            itr_to_plot = [iteration_to_plot]
        else:
            itr_to_plot = range(total_itr)

        for cond in range(total_cond):
            lines = list()
            labels = list()

            total_cost_types = len(pol_sample_lists_cost_compositions[-1][-1][-1])

            fig, ax = plt.subplots(1, 1)
            fig.canvas.set_window_title('Policy Costs | Condition %d' % cond)
            fig.set_facecolor((1, 1, 1))
            if plot_cost_types:
                colormap_list = [colormap(i) for i in np.linspace(0, 1, (len(itr_to_plot)*total_cost_types)+1)]
            else:
                colormap_list = [colormap(i) for i in np.linspace(0, 1, len(itr_to_plot))]
            ax.set_prop_cycle('color', colormap_list)
            ax.set_title('Policy Costs | Condition %d' % cond)

            mean_costs = np.zeros([total_itr, T])
            max_costs = np.zeros([total_itr, T])
            min_costs = np.zeros([total_itr, T])
            std_costs = np.zeros([total_itr, T])
            for itr in itr_to_plot:
                total_samples = len(pol_sample_lists_costs[itr][cond])
                samples_cost = pol_sample_lists_costs[itr][cond][:, :T]
                mean_costs[itr, :] = samples_cost.mean(axis=0)
                max_costs[itr, :] = samples_cost.max(axis=0)
                min_costs[itr, :] = samples_cost.min(axis=0)
                std_costs[itr, :] = samples_cost.std(axis=0)
                label = 'Total Cost (itr%d)' % itr
                line = ax.plot(mean_costs[itr, :], label=label)[0]
                ax.fill_between(range(T), min_costs[itr, :], max_costs[itr, :], alpha=0.5)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                lines.append(line)
                labels.append(label)

            # Composition cost
            if plot_cost_types:
                mean_cost_types = np.zeros([total_itr, total_cost_types, T])
                max_cost_types = np.zeros([total_itr, total_cost_types, T])
                min_cost_types = np.zeros([total_itr, total_cost_types, T])
                std_cost_types = np.zeros([total_itr, total_cost_types, T])
                for itr in itr_to_plot:
                    total_samples = len(pol_sample_lists_cost_compositions[itr][cond])
                    for c in range(total_cost_types):
                        cost_type = np.zeros([total_samples, T])
                        for n in range(total_samples):
                            cost_type[n, :] = pol_sample_lists_cost_compositions[itr][cond][n][c][:T]
                        mean_cost_types[itr, c, :] = cost_type.mean(axis=0)
                        max_cost_types[itr, c, :] = cost_type.max(axis=0)
                        min_cost_types[itr, c, :] = cost_type.min(axis=0)
                        std_cost_types[itr, c, :] = cost_type.std(axis=0)

                for c in range(total_cost_types):
                    label = 'Cost type %d (itr%d)' % (c, itr)
                    line = ax.plot(mean_cost_types[itr, c, :], label=label)[0]
                    ax.fill_between(range(T), min_cost_types[itr, c, :], max_cost_types[itr, c, :], alpha=0.5)
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

                    lines.append(line)
                    labels.append(label)

            plt.xlabel('Time')
            plt.ylabel('Cost')
            ax.set_xlim([0, T])
            ax.set_xticks(np.arange(0, T+2, 50))

            legend = plt.figlegend(lines, labels, loc='center right', ncol=1, labelspacing=0., borderaxespad=1.)
            legend.get_frame().set_alpha(0.4)


if options['plot_train_errors']:
    linewidth = 2.5
    for cond in range(total_cond):
        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(hspace=0)
        fig.canvas.set_window_title('Errors per Iteration Condition %d' % cond)
        fig.set_facecolor((1, 1, 1))
        #for ii in range(axs.size):
        #    ax = axs[ii/sample_list_cols, ii % sample_list_cols]
        #    ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, total_itr)])
        des_colormap = [colormap(i) for i in np.linspace(0, 1, total_gps)]

        lines = list()
        labels = list()

        min_iteration = np.inf
        max_iteration = -np.inf
        for gps in range(total_gps):
            N = iteration_data_list[gps][0][cond].sample_list.get_states().shape[0]
            costs = np.zeros((len(iteration_ids[gps]), N))
            errors = np.zeros(len(iteration_ids[gps]))
            print('%'*20)
            for itr in range(len(iteration_ids[gps])):
                #print("%d | %d" % (gps, itr))
                costs[itr, :] = iteration_data_list[gps][itr][cond].cs.sum(axis=1)
                worst_index = np.argsort(-costs[itr, :])
                print('itr%02d | %s | %f' % (itr, worst_index, costs[itr, worst_index[0]]))

                states = iteration_data_list[gps][itr][cond].sample_list.get_states()
                drill_distance = states[:, :, indeces_drill]
                N = drill_distance.shape[0]
                T = drill_distance.shape[1]

                error_count = 0
                for nn in range(N):
                    # # max_t_errors:
                    # acum_t_error = 0
                    # for t in range(T):
                    #     if np.any(np.abs(drill_distance[nn, t, :]) > failure_limits):
                    #         acum_t_error += 1
                    #     if acum_t_error > max_t_acum_error:
                    #         error_count += 1
                    #         break

                    # only last error
                    if np.any(np.abs(drill_distance[nn, -1, :]) > failure_limits):
                        error_count += 1


                errors[itr] = float(error_count)/N

                if iteration_ids[gps][itr] < min_iteration:
                    min_iteration = iteration_ids[gps][itr]

                if iteration_ids[gps][itr] > max_iteration:
                    max_iteration = iteration_ids[gps][itr]

            #line = ax.plot(iteration_ids[gps], errors, color=des_colormap[gps], label=gps_models_labels[gps],
            line = ax.plot(iteration_ids[gps], errors, label=gps_models_labels[gps], color=gps_models_colors[gps],
                           linestyle=gps_models_line_styles[gps], linewidth=linewidth, marker=gps_models_markers[gps])[0]

            lines.append(line)
            labels.append(gps_models_labels[gps])

            worst_ranking = np.dstack(np.unravel_index(np.argsort(-costs.ravel()), costs.shape))
            print('&'*10)
            print('%s' % worst_ranking[0, :15, :])
            for ww in range(15):
                print('%s' % costs[worst_ranking[0, ww, 0], worst_ranking[0, ww, 1]])

        ax.set_xlabel("Iterations", fontsize=30, weight='bold')
        ax.set_ylabel("Bad Samples / Total Samples", fontsize=30, weight='bold')
        ax.tick_params(axis='x', labelsize=25)
        ax.tick_params(axis='y', labelsize=25)
        #ax.set_xticks(range(min_iteration, max_iteration+1, 5))
        ax.set_xlim(min_iteration, max_iteration)
        ax.set_xticks(range(0, 26, 5))
        ax.set_xlim(0, 25)
        #ax_trajs[vv].set_ylim(ax_mins[y_index], ax_maxs[y_index])


        #legend = plt.figlegend(lines, labels, loc='center right', ncol=1, labelspacing=0., #borderaxespad=1.,
        #                       fontsize=20,  bbox_to_anchor=(5.0, 0.0))
        #legend = ax.legend(lines, labels, loc='center right', ncol=1, labelspacing=0., #borderaxespad=1.,
        #                       fontsize=20,  bbox_to_anchor=(-5.0, 0.0))
        #legend = ax.legend(loc='center right', bbox_to_anchor=(1.5, 0), ncol=1, fontsize=20)
        legend = ax.legend(ncol=1, fontsize=25)
        legend = ax.legend(borderaxespad=0.5, fontsize=20)
        box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #legend.get_frame().set_alpha(0.4)
        #legend.get_frame().set_edgecolor('k')

plt.show(block=False)

raw_input('Showing plots. Press a key to close...')
