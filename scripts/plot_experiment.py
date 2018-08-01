import argparse
import numpy as np
import matplotlib.pyplot as plt
from robolearn.utils.plots import plot_process_iu_returns
from robolearn.utils.plots import plot_process_iu_policies
from robolearn.utils.plots import plot_process_iu_values_errors
from robolearn.utils.plots import plot_process_general_data


def main(args):
    # plot_process_general_data(csv_file=args.file, n_unintentional=args.un,
    #                           block=False)

    plot_process_iu_values_errors(csv_file=args.file, n_unintentional=args.un,
                                  block=False)

    plot_process_iu_policies(csv_file=args.file, n_unintentional=args.un,
                             block=False)

    plot_process_iu_returns(csv_file=args.file, n_unintentional=args.un,
                            block=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default='./parameters.pkl',
                        help='path to the progress.csv file')
    parser.add_argument('--un', type=int, default=-1,
                        help='Unintentional id')
    args = parser.parse_args()

    main(args)
    input('Press a key to close script')
