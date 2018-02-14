import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import os, sys

method = 'trajopt'  # 'gps' or 'trajopt'
gps_directory_names = ['trajopt_log1']
gps_directory_names = ['reacher_log']
gps_models_labels = ['test']#, 'gps2', 'gps3', 'gps4']

#gps_models_line_styles = [':', '--', '-']
gps_models_line_styles = ['-', '-', '-', '-', '-']
gps_models_colors = ['black', 'magenta', 'orange', 'blue', 'green']
gps_models_markers = ['^', 's', 'D', '^', 's', 'D']

init_itr = 0
final_itr = 200
samples_idx = [-1]  # List of samples / None: all samples
max_traj_plots = None  # None, plot all
last_n_iters = None  # 5  # None, plot all iterations
itr_to_load = list(range(5))
# itr_to_load = [0, 4, 8]

colormap = plt.cm.rainbow  # nipy_spectral, Set1, Paired, winter
plot_sample_list_max_min = False
plot_joint_limits = True
gps_num = 0
total_gps = len(gps_directory_names)

duals_list = [list() for _ in range(total_gps)]
iteration_ids = [list() for _ in range(total_gps)]


# Get the data
for gps, gps_directory_name in enumerate(gps_directory_names):
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/../' + gps_directory_name

    max_available_runs = 0
    for rr in range(100):
        if os.path.exists(dir_path + ('/run_%02d' % rr)):
            max_available_runs += 1

    if max_available_runs == 0:
        print("There is not any runs data. Is the path '%s' correct?"
              % dir_path)
        exit(-1)

    print("Max available runs: %d in file %s"
          % (max_available_runs, gps_directory_name))

    duals_list[gps] = [list() for _ in range(max_available_runs)]
    iteration_ids[gps] = [list() for _ in range(max_available_runs)]

    for rr in range(max_available_runs):

        max_available_itr = 0
        for pp in range(init_itr, final_itr):
            if os.path.exists(dir_path + ('/run_%02d' % rr)
                                  + ('/itr_%02d' % pp)):
                max_available_itr += 1

        if max_available_itr == 0:
            print("There is not any iteration data. Is the path '%s' correct?"
                  % dir_path)
            exit(-1)

        print("Max available iterations: %d in file %s/run_%02d"
              % (max_available_itr, gps_directory_name, rr))

        if itr_to_load is None:
            if last_n_iters is not None:
                init_itr = max(max_available_itr - last_n_iters + 1, 0)

            if max_traj_plots is not None:
                if max_available_itr > max_traj_plots:
                    itr_to_load = np.linspace(init_itr, max_available_itr,
                                              max_traj_plots, dtype=np.uint8)
                else:
                    itr_to_load = list(range(init_itr, max_available_itr+1))

            else:
                itr_to_load = list(range(init_itr, max_available_itr+1))

        print("Desired iterations to load in %s: %s" % (gps_directory_name,
                                                        itr_to_load))

        first_itr_data = True
        total_itr = len(itr_to_load)
        for ii, itr_idx in enumerate(itr_to_load):
            itr_path = dir_path + str('/run_%02d' % rr) + \
                       str('/itr_%02d/' % itr_idx)
            # Duals
            file_to_load = itr_path + 'iteration_data_itr_' + \
                           str('%02d' % itr_idx)+'.pkl'
            if os.path.isfile(file_to_load):
                print('Loading GPS iteration_data from iteration %02d'
                      % itr_idx)
                iter_data = pickle.load(open(file_to_load, 'rb'))
                iteration_ids[gps][rr].append(itr_idx+1)
                n_cond = len(iter_data)
                T = iter_data[-1].cs.shape[1]
                if first_itr_data:
                    duals = np.zeros((n_cond, total_itr, 3))
                    first_itr_data = False

                for cc in range(n_cond):
                    duals[cc, ii, 0] = iter_data[cc].eta
                    duals[cc, ii, 1] = iter_data[cc].nu
                    duals[cc, ii, 2] = iter_data[cc].omega
            else:
                raise ValueError('ItrData does not exist! | gps[%02d] run[%02d]'
                                 ' itr[%02d]' % (gps, rr, itr_idx))

        # Clear all the loaded data
        duals_list[gps][rr] = duals
        del iter_data


total_runs = len(duals_list[-1])
total_cond = duals_list[-1][-1].shape[0]
total_itr = duals_list[-1][-1].shape[1]

for cond in range(total_cond):
    fig, ax = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0)
    fig.canvas.set_window_title('Duals Condition %02d '
                                '(over %02d runs)'
                                % (cond, total_runs))
    fig.set_facecolor((1, 1, 1))
    des_colormap = [colormap(i) for i in np.linspace(0, 1, total_gps)]

    lines = list()
    labels = list()

    for gps in range(total_gps):
        print('&&&'*5)
        print('&&&'*5)
        print('TODO: WE ARE USING ONLY ONE RUN')
        print('&&&'*5)
        rr = -1

        duals = duals_list[gps][rr][cond, :, :]

        # Eta
        ax[0].set_title('Eta (step)')
        line = ax[0].plot(iteration_ids[gps][rr], duals[:, 0],
                          marker=gps_models_markers[gps],
                          label=gps_models_labels[gps],
                          linestyle=gps_models_line_styles[gps],
                          color=gps_models_colors[gps])[0]

        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        lines.append(line)
        labels.append(gps_models_labels[gps])
        legend = ax[0].legend(ncol=1, fontsize=12)

        # Nu
        ax[1].set_title('Nu (bad)')
        line = ax[1].plot(iteration_ids[gps][rr], duals[:, 1],
                          marker=gps_models_markers[gps],
                          label=gps_models_labels[gps],
                          linestyle=gps_models_line_styles[gps],
                          color=gps_models_colors[gps])[0]

        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        lines.append(line)
        labels.append(gps_models_labels[gps])
        legend = ax[1].legend(ncol=1, fontsize=12)

        # Omega
        ax[2].set_title('Omega (good)')
        line = ax[2].plot(iteration_ids[gps][rr], duals[:, 2],
                          marker=gps_models_markers[gps],
                          label=gps_models_labels[gps],
                          linestyle=gps_models_line_styles[gps],
                          color=gps_models_colors[gps])[0]

        ax[2].xaxis.set_major_locator(MaxNLocator(integer=True))
        lines.append(line)
        labels.append(gps_models_labels[gps])
        legend = ax[2].legend(ncol=1, fontsize=12)


plt.show(block=False)

input('Showing plots. Press a key to close...')
