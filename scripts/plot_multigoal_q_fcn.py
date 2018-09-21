import numpy as np
import argparse
import joblib
import IPython

from robolearn.envs.simple_envs.multigoal_deprecated.multigoal_q_plot_ import QFPolicyPlotter


def main(args):
    data = joblib.load(args.file)
    if args.deterministic:
        print('Using the deterministic version of the _i_policy.')
        policy = data['_i_policy']
    else:
        print('Using the stochastic _i_policy.')
        policy = data['exploration_policy']

    qf = data['_i_qf']

    # QF Plot
    plotter = QFPolicyPlotter(
        qf=qf,
        policy=policy,
        obs_lst=np.array([[-2.5, 0.0], [0.0, 0.0], [2.5, 2.5]]),
        default_action=[np.nan, np.nan],
        n_samples=100)

    plotter.draw()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--deterministic', action="store_true")

    args = parser.parse_args()
    main(args)

    input('Press a key to close the script...')

    IPython.embed()
