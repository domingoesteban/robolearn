import numpy as np
import matplotlib.pyplot as plt
import argparse
import joblib
import json

from robolearn.envs.normalized_box_env import NormalizedBoxEnv
from robolearn.envs.simple_envs.navigation2d import Navigation2dGoalCompoEnv


def plot_v_fcn(i_vf, u_vf):
    xlim = (-7, 7)
    ylim = (-7, 7)
    delta = 0.05
    x_min, x_max = tuple(1.1 * np.array(xlim))
    y_min, y_max = tuple(1.1 * np.array(ylim))
    all_x = np.arange(x_min, x_max, delta)
    all_y = np.arange(y_min, y_max, delta)
    xy_mesh = np.meshgrid(all_x, all_y)
    all_obs = np.array(xy_mesh).transpose(1, 2, 0).reshape(-1, 2)

    def plot_v_contours(ax, values):
        values = values.reshape(len(all_x), len(all_y))

        contours = ax.contour(xy_mesh[0], xy_mesh[1], values, 20,
                              colors='dimgray')
        ax.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
        ax.imshow(values, extent=(x_min, x_max, y_min, y_max), origin='lower')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
        ax.set_aspect('equal', 'box')

    # Compute and ploy Sub-tasks Value-Fcn
    n_unintentions = u_vf.n_heads if u_vf is not None else 0
    n_cols = 3 if u_vf is not None else 1
    n_rows = int(np.ceil((n_unintentions+1)/n_cols))
    subgoals_fig, subgoals_axs = plt.subplots(n_rows, n_cols)
    subgoals_axs = np.atleast_2d(subgoals_axs)

    subgoals_fig.suptitle('V-values')

    # Compute and plot Main Task Value-fcn
    values, _ = i_vf.get_values(all_obs)
    plot_v_contours(subgoals_axs[0, 0], values)
    subgoals_axs[0, 0].set_title("Main Task")

    for aa in range(n_unintentions):
        row = (aa+1) // n_cols
        col = (aa+1) % n_cols
        subgo_ax = subgoals_axs[row, col]
        values, _ = u_vf.get_values(all_obs, val_idxs=[aa])
        values = values[0]

        subgo_ax.set_title("Sub-Task %02d" % aa)
        plot_v_contours(subgo_ax, values)


def plot_q_fcn(i_qf, i_qf2, u_qf, u_qf2, obs, policy):
    # Load environment
    with open('variant.json') as json_data:
        env_params = json.load(json_data)['env_params']

    env = NormalizedBoxEnv(
        Navigation2dGoalCompoEnv(**env_params),
        # normalize_obs=True,
        normalize_obs=False,
        online_normalization=False,
        obs_mean=None,
        obs_var=None,
        obs_alpha=0.001,
    )
    env.reset()
    env.render()

    obs = np.array(obs)
    n_action_samples = 100
    x_min, y_min = env.action_space.low
    x_max, y_max = env.action_space.high
    delta = 0.05
    xlim = (1.1*x_min, 1.1*x_max)
    ylim = (1.1*y_min, 1.1*y_max)
    all_x = np.arange(x_min, x_max, delta)
    all_y = np.arange(y_min, y_max, delta)
    xy_mesh = np.meshgrid(all_x, all_y)

    all_acts = np.zeros((len(all_x)*len(all_y), 2))
    all_acts[:, 0] = xy_mesh[0].ravel()
    all_acts[:, 1] = xy_mesh[1].ravel()

    n_unintentions = u_qf.n_heads if u_qf is not None else 0

    def plot_q_contours(ax, values):
        values = values.reshape(len(all_x), len(all_y))

        contours = ax.contour(xy_mesh[0], xy_mesh[1], values, 20,
                              colors='dimgray')
        ax.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
        ax.imshow(values, extent=(x_min, x_max, y_min, y_max), origin='lower',
                  alpha=0.5)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
        ax.set_aspect('equal', 'box')

    def plot_action_samples(ax, actions):
        x, y = actions[:, 0], actions[:, 1]
        ax.scatter(x, y, c='b', marker='*', zorder=5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    for ob in obs:
        all_obs = np.broadcast_to(ob, (all_acts.shape[0], 2))

        fig, all_axs = plt.subplots(1, n_unintentions + 1)
        fig.suptitle('Q-val Observation: ' + str(ob))

        all_axs = np.atleast_1d(all_axs)

        all_axs[0].set_title('Main Task')
        q_vals = i_qf.get_values(all_obs, all_acts)[0]
        if i_qf2 is not None:
            q2_vals = i_qf2.get_values(all_obs, all_acts)[0]
            q_vals = np.concatenate([q_vals, q2_vals], axis=1)
            q_vals = np.min(q_vals, axis=1, keepdims=True)

        plot_q_contours(all_axs[0], q_vals)

        if u_qf is None:
            pol_kwargs = dict(
            )
        else:
            pol_kwargs = dict(
                pol_idx=None,
            )

        # Compute and plot Main Task Q Value
        action_samples = policy.get_actions(all_obs[:n_action_samples, :],
                                            deterministic=False,
                                            **pol_kwargs
                                            )[0]
        plot_action_samples(all_axs[0], action_samples)

        for aa in range(n_unintentions):
            subgo_ax = all_axs[aa + 1]
            subgo_ax.set_title('Sub-Task %02d' % aa)

            q_vals = u_qf.get_values(all_obs, all_acts, val_idxs=[aa])[0]
            q_vals = q_vals[0]
            if u_qf2 is not None:
                q2_vals = u_qf2.get_values(all_obs, all_acts)[0]
                q2_vals = q2_vals[0]
                q_vals = np.concatenate([q_vals, q2_vals], axis=1)
                q_vals = np.min(q_vals, axis=1, keepdims=True)

            plot_q_contours(subgo_ax, q_vals)

            if u_qf is None:
                pol_kwargs = dict(
                )
            else:
                pol_kwargs = dict(
                    pol_idx=aa,
                )

            # Compute and plot Sub-Task Q Value
            action_samples = policy.get_actions(all_obs[:n_action_samples, :],
                                                deterministic=False,
                                                **pol_kwargs
                                                )[0]
            plot_action_samples(subgo_ax, action_samples)


def main(args):
    data = joblib.load(args.file)

    policy = data['policy']
    if 'u_qf' in data.keys():
        u_qf = data['u_qf']
    else:
        u_qf = None
    if 'u_qf2' in data.keys():
        u_qf2 = data['u_qf2']
    else:
        u_qf2 = None
    if 'u_vf' in data.keys():
        u_vf = data['u_vf']
    else:
        u_vf = None
    i_qf = data['qf']
    i_qf2 = data['qf2']
    i_vf = data['vf']

    q_fcn_obs = [
        (-2.0, 4),
        (-2.0, -2.0),
        (4, -2.0),
        (-7, -7.00),
    ]

    # QF Plot
    plot_q_fcn(i_qf, i_qf2, u_qf, u_qf2, q_fcn_obs, policy)

    # VF Plot
    plot_v_fcn(i_vf, u_vf)

    plt.show()

    epoch = data['epoch']

    print('Data for epoch: %02d' % epoch)

    # IPython.embed()
    # return plotter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default='./params.pkl',
                        help='path to the snapshot file')

    args = parser.parse_args()
    plotter = main(args)

    input('Press a key to close the script...')
