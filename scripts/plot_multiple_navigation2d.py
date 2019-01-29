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

# POSSIBLE_SEEDS = [110, 210, 310, 410, 510, 610, 710, 810, 910, 1010]
# SEEDS = [610, 710, 810, 910, 1010]
# MAX_ITER = 500
# STEPS_PER_ITER = 1e2
# LOG_PREFIX = '/home/desteban/logs/objective_test/navigation2d'

# SEEDS = [610, 710, 810, 1010]
SEEDS = [610]
MAX_ITER = 100
# STEPS_PER_ITER = 3e3
STEPS_PER_ITER = None
LOG_PREFIX = '/home/desteban/logs/objective_test/navigation2d'

fig_name_prefix = 'Navigation2D_'


# Compo: Ient:2, Uents:2
# Compo2: Ient:2, Uents:1


hiu_performance_dict = dict()
# Subtask 01
hiu_performance_dict['Subtask 01'] = dict()
hiu_performance_dict['Subtask 01']['SAC'] = dict(
    dir='sub0',
    prefix='sacX_',
    ius=[-1],
    r_scales=[1.e-0],
)
hiu_performance_dict['Subtask 01']['HIU-SAC-W'] = dict(
    dir='sub-1',
    prefix='hiu_sacX_',
    ius=[0],
    r_scales=[1.e-0],
)
hiu_performance_dict['Subtask 01']['HIU-SAC-W2'] = dict(
    dir='sub-1',
    prefix='hiu_sacY_',
    ius=[0],
    r_scales=[1.e-0],
)



# Subtask 02
hiu_performance_dict['Subtask 02'] = dict()
hiu_performance_dict['Subtask 02']['SAC'] = dict(
    dir='sub1',
    prefix='sacX_',
    ius=[-1],
    r_scales=[1.e-0],
)
hiu_performance_dict['Subtask 02']['HIU-SAC-W'] = dict(
    dir='sub-1',
    prefix='hiu_sacX_',
    ius=[1],
    r_scales=[1.e-0],
)
hiu_performance_dict['Subtask 02']['HIU-SAC-W2'] = dict(
    dir='sub-1',
    prefix='hiu_sacY_',
    ius=[1],
    r_scales=[1.e-0],
)


# Maintask
hiu_performance_dict['Main Task'] = dict()
hiu_performance_dict['Main Task']['SAC'] = dict(
    dir='sub-1',
    prefix='sacX_',
    ius=[-1],
    r_scales=[1.e-0],
)
hiu_performance_dict['Main Task']['HIU-SAC-W'] = dict(
    dir='sub-1',
    prefix='hiu_sacX_',
    ius=[-1],
    r_scales=[1.e-0],
)
hiu_performance_dict['Main Task']['HIU-SAC-W2'] = dict(
    dir='sub-1',
    prefix='hiu_sacY_',
    ius=[-1],
    r_scales=[1.e-0],
)
# hiu_performance_dict['Main Task']['HIU-SAC-E'] = dict(
#     dir='sub-1',
#     prefix='hiu_sac_new_promp_',
#     ius=[-1],
#     r_scales=[1.e-0],
# )
# hiu_performance_dict['Main Task']['HIU-SAC-M'] = dict(
#     dir='sub-1',
#     prefix='hiu_sac_new_mixture_',
#     ius=[-1],
#     r_scales=[1.e-0],
# )


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


def main(args):
    directories_dict = get_full_seed_paths(hiu_performance_dict)

    # directories_dict = get_subtask_and_seed_idxs()

    plot_multiple_process_iu_returns(
        directories_dict,
        max_iter=MAX_ITER,
        steps_per_iter=STEPS_PER_ITER,
        fig_name_prefix=fig_name_prefix,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('file', type=str, default='./progress.csv',
    #                     help='path to the progress.csv file')
    parser.add_argument('--un', type=int, default=-1,
                        help='Unintentional id')
    parser.add_argument('--no_in', action='store_false')
    args = parser.parse_args()

    main(args)
    input('Press a key to close script')
