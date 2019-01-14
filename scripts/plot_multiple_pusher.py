import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from robolearn.utils.plots import plot_multiple_process_iu_returns
from robolearn.utils.plots import plot_process_iu_policies
from robolearn.utils.plots import plot_process_iu_values_errors
from robolearn.utils.plots import plot_process_general_data
from robolearn.utils.plots.learning_process_plots import plot_process_haarnoja
import json

# SEEDS = [610, 710, 810, 1010]
SEEDS = [610]
# MAX_ITER = 590
MAX_ITER = 500
# MAX_ITER = 50
# STEPS_PER_ITER = 3e3
STEPS_PER_ITER = None
LOG_PREFIX = '/home/desteban/logs/objective_test/pusher'


hiu_performance_dict = dict()
# Subtask 01
hiu_performance_dict['Subtask 01'] = dict()
hiu_performance_dict['Subtask 01']['HIU-SAC-W'] = dict(
    dir='sub-1',
    prefix='hiu_sacB_1_',
    ius=[0],
    r_scales=[1.e-0],
)
hiu_performance_dict['Subtask 01']['HIU-SAC-E'] = dict(
    dir='sub-1',
    prefix='hiu_sac_prompB_1_',
    ius=[0],
    r_scales=[1.e-0],
)
# hiu_performance_dict['Subtask 01']['W1-0'] = dict(
#     dir='sub-1',
#     prefix='hiu_sac_new_0_',
#     ius=[0],
#     r_scales=[1.e-0],
# )
# hiu_performance_dict['Subtask 01']['W1-5'] = dict(
#     dir='sub-1',
#     prefix='hiu_sac_new5_0_',
#     ius=[0],
#     r_scales=[1.e-0],
# )
# hiu_performance_dict['Subtask 01']['E1-0'] = dict(
#     dir='sub-1',
#     prefix='hiu_sac_new_promp_0_',
#     ius=[0],
#     r_scales=[1.e-0],
# )
# hiu_performance_dict['Subtask 01']['M2-5'] = dict(
#     dir='sub-1',
#     prefix='hiu_sac_new_mixture2_5_',
#     ius=[0],
#     r_scales=[1.e-0],
# )

# Subtask 02
hiu_performance_dict['Subtask 02'] = dict()
# hiu_performance_dict['Subtask 02']['W1-0'] = dict(
#     dir='sub-1',
#     prefix='hiu_sac_new_0_',
#     ius=[1],
#     r_scales=[1.e-0],
# )
# hiu_performance_dict['Subtask 02']['W1-5'] = dict(
#     dir='sub-1',
#     prefix='hiu_sac_new5_0_',
#     ius=[1],
#     r_scales=[1.e-0],
# )
# hiu_performance_dict['Subtask 02']['E1-0'] = dict(
#     dir='sub-1',
#     prefix='hiu_sac_new_promp_0_',
#     ius=[1],
#     r_scales=[1.e-0],
# )
# hiu_performance_dict['Subtask 02']['M2-5'] = dict(
#     dir='sub-1',
#     prefix='hiu_sac_new_mixture2_5_',
#     ius=[1],
#     r_scales=[1.e-0],
# )
hiu_performance_dict['Subtask 02']['HIU-SAC-W'] = dict(
    dir='sub-1',
    prefix='hiu_sacB_1_',
    ius=[1],
    r_scales=[1.e-0],
)
hiu_performance_dict['Subtask 02']['HIU-SAC-E'] = dict(
    dir='sub-1',
    prefix='hiu_sac_prompB_1_',
    ius=[1],
    r_scales=[1.e-0],
)

# Maintask
hiu_performance_dict['Main Task'] = dict()
# hiu_performance_dict['Main Task']['W1-0'] = dict(
#     dir='sub-1',
#     prefix='hiu_sac_new_0_',
#     ius=[-1],
#     r_scales=[1.e-0],
# )
# # hiu_performance_dict['Main Task']['W1-5'] = dict(
# #     dir='sub-1',
# #     prefix='hiu_sac_new5_0_',
# #     ius=[-1],
# #     r_scales=[1.e-0],
# # )
# hiu_performance_dict['Main Task']['E1-0'] = dict(
#     dir='sub-1',
#     prefix='hiu_sac_new_promp_0_',
#     ius=[-1],
#     r_scales=[1.e-0],
# )
# hiu_performance_dict['Main Task']['M2-5'] = dict(
#     dir='sub-1',
#     prefix='hiu_sac_new_mixture2_5_',
#     ius=[-1],
#     r_scales=[1.e-0],
# )
# # hiu_performance_dict['Main Task']['W1-1'] = dict(
# #     dir='sub-1',
# #     prefix='hiu_sac_new_1_',
# #     ius=[-1],
# #     r_scales=[1.e-0],
# # )
# hiu_performance_dict['Main Task']['M1-0'] = dict(
#     dir='sub-1',
#     prefix='hiu_sac_new_mixture1_0_',
#     ius=[-1],
#     r_scales=[1.e-0],
# )
# # hiu_performance_dict['Main Task']['M1-10'] = dict(
# #     dir='sub-1',
# #     prefix='hiu_sac_new_mixture1_10_',
# #     ius=[-1],
# #     r_scales=[1.e-0],
# # )
# hiu_performance_dict['Main Task']['M3-0'] = dict(
#     dir='sub-1',
#     prefix='hiu_sac_new_mixture3_0_',
#     ius=[-1],
#     r_scales=[1.e-0],
#
hiu_performance_dict['Main Task']['HIU-SAC-W'] = dict(
    dir='sub-1',
    prefix='hiu_sacB_1_',
    ius=[-1],
    r_scales=[1.e-0],
)
hiu_performance_dict['Main Task']['HIU-SAC-E'] = dict(
    dir='sub-1',
    prefix='hiu_sac_prompB_1_',
    ius=[-1],
    r_scales=[1.e-0],
)


def get_full_seed_paths(full_dict):
    categories = list(full_dict.keys())

    for cc, cate in enumerate(categories):
        expt_dict = full_dict[cate]
        expts = list(expt_dict)
        # print(expt_dict)
        expt_counter = 0
        for ee, expt in enumerate(expts):
            # print(expt['dir'])
            run_dict = expt_dict[expt]
            expt_dir = os.path.join(LOG_PREFIX, run_dict['dir'])
            if len(list_files_startswith(expt_dir, run_dict['prefix'])) > 0:
                expt_counter += 1
                dirs_and_iu = list()
                dir_prefix = os.path.join(expt_dir, run_dict['prefix'])
                # print(dir_prefix)
                for seed in SEEDS:
                    full_seed_dir = dir_prefix + str(seed)
                    # print('- ', full_seed_dir)
                    if os.path.exists(full_seed_dir):
                        # print('YES DATA IN: %s' % full_seed_dir)
                        dirs_and_iu.append((
                            full_seed_dir,
                            run_dict['ius'],
                            run_dict['r_scales'],
                        ))
                full_dict[cate][expt] = dirs_and_iu
        if expt_counter == 0:
            full_dict.pop(cate)
    return full_dict


def list_files_startswith(directory, prefix):
    return list(f for f in os.listdir(directory) if f.startswith(prefix))


def list_files_endswith(directory, suffix):
    return list(f for f in os.listdir(directory) if f.endswith(suffix))


def main():
    directories_dict = get_full_seed_paths(hiu_performance_dict)

    plot_multiple_process_iu_returns(
        directories_dict,
        max_iter=MAX_ITER,
        steps_per_iter=STEPS_PER_ITER,
    )


if __name__ == '__main__':
    main()
    input('Press a key to close script')
