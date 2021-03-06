import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import os
import torch
# from torch.autograd import Variable
# import robolearn.torch.pytorch_util as ptu
from robolearn.utils.plots import canvas_draw


class QFPolicyPlotter:
    def __init__(self, i_qf, i_policy, u_qfs, u_policies, obs_lst,
                 default_action, n_samples,
                 render=False, save_path=None):
        self._i_qf = i_qf
        self._i_policy = i_policy
        self._u_qfs = u_qfs
        self._u_policies = u_policies

        # Convert to cuda if they didn't
        if ptu.gpu_enabled():
            self._i_qf.cuda()
            self._i_policy.cuda()
            for u_qf, u_policy in zip(self._u_qfs, self._u_policies):
                u_qf.cuda()
                u_policy.cuda()

        self._obs_lst = np.array(obs_lst)
        self._default_action = default_action
        self._n_samples = n_samples

        self._var_inds = np.where(np.isnan(default_action))[0]
        assert len(self._var_inds) == 2

        self._n_demons = len(self._u_qfs)

        n_plots = self._n_demons + 1

        x_size = 5 * n_plots
        y_size = 5

        self._fig = plt.figure(figsize=(x_size, y_size))
        self._ax_lst = []
        self._line_objects = list()

        # plt.subplots_adjust(left=0.3)
        plt.subplots_adjust(left=0.10)
        for i in range(n_plots):
            ax = self._fig.add_subplot(100 + n_plots * 10 + i + 1)
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            ax.grid(True)
            if i == 0:
                ax.set_title('Intentional')
            else:
                ax.set_title('Unintentional %02d' % (i))
            ax.set_xlabel('Xvel')
            ax.set_ylabel('Yvel')
            self._ax_lst.append(ax)

        self._current_obs_idx = 1
        self._obs_labels = [str(obs) for obs in self._obs_lst]
        self._fig.canvas.set_window_title('Observation ' +
                                          self._obs_labels[self._current_obs_idx])
        self._plot_level_curves()
        self._plot_action_samples()

        self._radio_ax = \
            self._fig.add_axes([0.01, 0.48, 0.06, 0.05*len(self._obs_lst)])
        self._radio_button = RadioButtons(self._radio_ax, self._obs_labels,
                                          active=self._current_obs_idx)
        self._radio_button.on_clicked(self.radio_update_plots)

        if save_path is None:
            self._save_path = '/home/desteban/logs/q_plots'
        else:
            self._save_path = save_path

        if render:
            plt.show(block=False)
        canvas_draw(self._fig.canvas, 0.05)

    def radio_update_plots(self, label):
        idx = self._obs_labels.index(label)
        self._current_obs_idx = idx

        self._fig.canvas.set_window_title('Observation ' +
                                          self._obs_labels[self._current_obs_idx])

        self.draw()

    def save_figure(self, itr=0):
        fig_title = self._fig.suptitle("Iteration %02d" % itr, fontsize=14)
        prev_obs_idx = self._current_obs_idx
        self._radio_ax.set_visible(False)
        canvas_draw(self._fig.canvas, 0.01)

        for oo, label in enumerate(self._obs_labels):
            self._current_obs_idx = oo
            self.draw()

            fig_log_path = os.path.join(self._save_path,
                                        'obs%02d' % oo,
                                        )

            if not os.path.isdir(fig_log_path):
                os.makedirs(fig_log_path)

            fig_log_name = os.path.join(fig_log_path,
                                        ('%02d' % itr).zfill(4)
                                        )

            self._fig.savefig(fig_log_name)

        self._current_obs_idx = prev_obs_idx
        self._radio_ax.set_visible(True)
        fig_title.set_visible(False)
        canvas_draw(self._fig.canvas, 0.01)

    def draw(self):
        # noinspection PyArgumentList
        [h.remove() for h in self._line_objects]
        self._line_objects = list()

        self._plot_level_curves()
        self._plot_action_samples()

        canvas_draw(self._fig.canvas, 0.01)

    def _plot_level_curves(self):
        # Create mesh grid.
        xs = np.linspace(-1, 1, 50)
        ys = np.linspace(-1, 1, 50)
        xgrid, ygrid = np.meshgrid(xs, ys)
        N = len(xs)*len(ys)

        # Copy default values along the first axis and replace nans with
        # the mesh grid points.
        actions = np.tile(self._default_action, (N, 1)).astype(np.float32)
        actions[:, self._var_inds[0]] = xgrid.ravel()
        actions[:, self._var_inds[1]] = ygrid.ravel()

        obs = self._obs_lst[self._current_obs_idx]
        for dd in range(self._n_demons+1):
            ax = self._ax_lst[dd]

            if dd == 0:
                qf = self._i_qf
            else:
                qf = self._u_qfs[dd-1]

            obs = obs.astype(np.float32)
            obs_torch = torch.tensor(torch.from_numpy(obs).unsqueeze(0).expand(N, 2))
            actions_torch = torch.tensor(torch.from_numpy(actions))

            # if ptu.gpu_enabled():
            #     qs = ptu.get_numpy(_i_qf(obs_torch, actions_torch).squeeze())#.data.cpu().numpy()
            # else:
            qs = ptu.get_numpy(qf(obs_torch, actions_torch).squeeze())#.data.numpy()

            qs = qs.reshape(xgrid.shape)

            cs = ax.contour(xgrid, ygrid, qs, 20)
            self._line_objects += cs.collections
            self._line_objects += ax.clabel(
                cs, inline=1, fontsize=10, fmt='%.2f')

    def _plot_action_samples(self):
        obs = self._obs_lst[self._current_obs_idx]
        for dd in range(self._n_demons+1):
            ax = self._ax_lst[dd]

            if dd == 0:
                policy = self._i_policy
            else:
                policy = self._u_policies[dd-1]
            actions = policy.get_actions(
                np.ones((self._n_samples, 1)) * obs[None, :])
            x, y = actions[:, 0], actions[:, 1]
            self._line_objects += ax.plot(x, y, 'b0*')
