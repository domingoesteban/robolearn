import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from robolearn.utils.plots import plot_process_iu_returns
from robolearn.utils.plots import plot_process_iu_avg_rewards
from robolearn.utils.plots import plot_process_iu_policies
from robolearn.utils.plots import plot_process_iu_values_errors
from robolearn.utils.plots import plot_process_iu_alphas
from robolearn.utils.plots import plot_process_general_data
from robolearn.utils.plots.learning_process_plots import plot_process_haarnoja
import json


def main(args):
    # Load environment
    dirname = os.path.dirname(args.file)
    with open(os.path.join(dirname, 'variant.json')) as json_data:
        algo_name = json.load(json_data)['algo_name']

    # Plot according to RL algorithm
    if algo_name in ['HIUSAC', 'HIUSACNEW', 'SAC', 'HIUSACEpisodic']:
        plot_process_iu_values_errors(csv_file=args.file, n_unintentional=args.un,
                                      block=False)
        plot_process_iu_policies(csv_file=args.file, n_unintentional=args.un,
                                 block=False, plot_intentional=args.no_in,
                                 deterministic=False)
        plot_process_iu_alphas(csv_file=args.file, n_unintentional=args.un,
                               block=False)
        plot_process_iu_returns(csv_file=args.file, n_unintentional=args.un,
                                block=False)
        plot_process_iu_avg_rewards(csv_file=args.file,
                                    n_unintentional=args.un,
                                    block=False)

    elif algo_name in ['HIUDDPG']:
        plot_process_iu_policies(csv_file=args.file, n_unintentional=args.un,
                                 block=False, plot_intentional=args.no_in,
                                 deterministic=True)
        plot_process_iu_returns(csv_file=args.file, n_unintentional=args.un,
                                block=False)
    else:
        plot_process_general_data(csv_file=args.file, block=False)

    # plot_process_haarnoja(csv_file=args.file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default='./progress.csv',
                        help='path to the progress.csv file')
    parser.add_argument('--un', type=int, default=-1,
                        help='Unintentional id')
    parser.add_argument('--no_in', action='store_false')
    args = parser.parse_args()

    main(args)
    input('Press a key to close script')
