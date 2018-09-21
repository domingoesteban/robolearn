import numpy as np
import argparse
import joblib
# import IPython

from robolearn.envs.simple_envs.multigoal_deprecated.multigoal_q_plot_ import QFPolicyPlotter


def main(args):
    data = joblib.load(args.file)
    qfs = data['qfs']

    if args.deterministic:
        print('Using the deterministic version of the _i_policy.')
        policies = [data['_i_policy'] for _ in range(len(qfs))]
    else:
        print('Using the stochastic _i_policy.')
        policies = data['trained_policies']

    # q_fcn_positions = [
    #     (-2.5, 0.0),
    #     (0.0, 0.0),
    #     (2.5, 2.5)
    # ]
    q_fcn_positions = [
        (5, 5),
        (0, 0),
        (-5, 5)
    ]

    # QF Plot
    plotter = QFPolicyPlotter(
        qf=qfs,
        policy=policies,
        obs_lst=q_fcn_positions,
        default_action=[np.nan, np.nan],
        n_samples=100,
        render=True,
        )

    plotter.draw()

    # IPython.embed()
    return plotter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--deterministic', action="store_true")

    args = parser.parse_args()
    plotter = main(args)

    input('Press a key to close the script...')
