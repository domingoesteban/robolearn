"""
Based on Haarnoja sac's multigoal_env.py file

https://github.com/haarnoja/sac
"""

import numpy as np
import matplotlib.pyplot as plt
from robolearn.utils.plots import plt_pause

from robolearn.core.serializable import Serializable
from gym.spaces.box import Box


class MultiCompositionEnv(Serializable):
    """
    Move a 2D point mass to one of the goal positions. Cost is the distance to
    the closest goal.

    State: position.
    Action: velocity.
    """
    def __init__(self, goal_reward=10, actuation_cost_coeff=30,
                 distance_cost_coeff=1, init_sigma=0.1, goal_positions=None):
        Serializable.__init__(self)
        Serializable.quick_init(self, locals())

        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.array((0, 0), dtype=np.float32)
        self.init_sigma = init_sigma
        if goal_positions is None:
            self.goal_positions = np.array(
                [
                    [5, 0],
                    [-5, 0],
                    [0, 5],
                    [0, -5]
                ],
                dtype=np.float32
            )
        else:
            self.goal_positions = np.array(goal_positions, dtype=np.float32)

        self.goal_threshold = .1
        self.goal_reward = goal_reward
        self.action_cost_coeff = actuation_cost_coeff
        self.distance_cost_coeff = distance_cost_coeff
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1.
        self.observation = None

        # Rendering
        self._fig = None
        self._ax = None
        self._env_lines = list()
        self.fixed_plots = None
        self.dynamic_plots = []

        # New rendering
        self._goals_fig = None
        self._goals_ax = None
        self._dynamic_line = None
        self._dynamic_goals_lines = list()

        # self.reset()

    def reset(self):
        self.clear_all_plots()

        unclipped_observation = self.init_mu + self.init_sigma * \
            np.random.normal(size=self.dynamics.s_dim)
        o_lb, o_ub = self.observation_space.low, self.observation_space.high
        self.observation = np.clip(unclipped_observation, o_lb, o_ub)
        return self.observation

    @property
    def observation_space(self):
        return Box(
            low=np.array((self.xlim[0], self.ylim[0])),
            high=np.array((self.xlim[1], self.ylim[1])),
            shape=None,
            dtype=np.float32
        )

    @property
    def action_space(self):
        return Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(self.dynamics.a_dim,),
            dtype=np.float32
        )

    def get_current_obs(self):
        return np.copy(self.observation)

    def step(self, action):
        action = action.ravel()

        a_lb, a_ub = self.action_space.low, self.action_space.high
        action = np.clip(action, a_lb, a_ub).ravel()

        next_obs = self.dynamics.forward(self.observation, action)
        o_lb, o_ub = self.observation_space.low, self.observation_space.high
        next_obs = np.clip(next_obs, o_lb, o_ub)

        self.observation = np.copy(next_obs)

        # compute running/stage reward
        reward, reward_composition = self.compute_reward(self.observation,
                                                         action)
        # compute final reward
        cur_position = self.observation
        dist_to_all_goals = np.array(
            [np.linalg.norm(cur_position - goal_position)
             for goal_position in self.goal_positions]
        )
        dist_to_goal = np.amin(dist_to_all_goals)
        done = dist_to_goal < self.goal_threshold
        if done:
            reward += self.goal_reward
        any_done = dist_to_all_goals < self.goal_threshold
        if np.any(any_done):
            reward_composition += self.goal_reward*any_done

        return next_obs, reward, done, \
            {'pos': next_obs, 'reward_vector': reward_composition}

    def compute_reward(self, observation, action):
        # penalize the L2 norm of acceleration
        # noinspection PyTypeChecker
        action_cost = self.action_cost_coeff * np.sum(action ** 2)

        # penalize squared dist to goal
        cur_position = observation
        # noinspection PyTypeChecker
        dist_all_goals = np.array(
            [np.sum((cur_position - goal_position) ** 2)
             for goal_position in self.goal_positions]
        )
        goal_costs = self.distance_cost_coeff * dist_all_goals

        dist_cost = np.amin(dist_all_goals)
        goal_cost = self.distance_cost_coeff * dist_cost

        # penalize staying with the log barriers ???
        costs = [action_cost, goal_cost]
        reward = -np.sum(costs)
        reward_composition = -(action_cost + goal_costs)
        return reward, reward_composition

    def render(self, paths=None):
        if self._ax is None:
            self._init_plot()

        if self._goals_ax is None:
            self._init_goals_plot()

        if not self._fig_exist and not self._goal_fig_exist:  # Figures closed
            exit(-1)

        if paths is not None:
            # noinspection PyArgumentList
            [line.remove() for line in self._env_lines]
            self._env_lines = list()
            for path in paths:
                positions = path["env_infos"]["pos"]
                xx = positions[:, 0]
                yy = positions[:, 1]
                self._env_lines += self._ax.plot(xx, yy, 'b0')
        else:
            if self.observation is not None:
                if self._dynamic_line is None:
                    self._dynamic_line, = self._ax.plot(self.observation[0],
                                                        self.observation[1],
                                                        color='b0',
                                                        marker='o',
                                                        markersize=2)

                    n_cols = 2
                    n_goals = self.goal_positions.shape[0]
                    for aa in range(n_goals):
                        row = aa // n_cols
                        col = aa % n_cols
                        ax = self._goals_ax[row, col]
                        line, = ax.plot(self.observation[0], self.observation[1],
                                        color='b0', marker='o', markersize=2)
                        self._dynamic_goals_lines.append(line)

                else:
                    line = self._dynamic_line
                    line.set_xdata(np.append(line.get_xdata(), self.observation[0]))
                    line.set_ydata(np.append(line.get_ydata(), self.observation[1]))

                    n_goals = self.goal_positions.shape[0]
                    for aa in range(n_goals):
                        line = self._dynamic_goals_lines[aa]
                        line.set_xdata(np.append(line.get_xdata(),
                                                 self.observation[0]))
                        line.set_ydata(np.append(line.get_ydata(),
                                                 self.observation[1]))

        plt.draw()
        # plt.pause(0.01)
        plt_pause(0.001)

    def clear_all_plots(self):
        if self._dynamic_line is not None:
            self._dynamic_line.remove()
            plt.draw()
            # plt.pause(0.01)
            plt_pause(0.01)
            self._dynamic_line = None

        if self._dynamic_goals_lines:
            for ll in self._dynamic_goals_lines:
                ll.remove()
            plt.draw()
            # plt.pause(0.01)
            plt_pause(0.005)
            self._dynamic_goals_lines = list()

    def _init_plot(self):
        plt.ion()
        self._fig = plt.figure(figsize=(7, 7))
        self._ax = self._fig.add_subplot(111)
        self._ax.axis('equal')

        self._env_lines = list()
        self._ax.set_xlim((-7, 7))
        self._ax.set_ylim((-7, 7))

        self._ax.set_title('Multigoal Environment')
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y0')

        self._plot_position_cost(self._ax)

        plt.show(block=False)

    def _init_goals_plot(self):
        plt.ion()
        n_cols = 2
        n_goals = self.goal_positions.shape[0]
        n_rows = int(np.ceil(n_goals/n_cols))
        self._goals_fig, self._goals_ax = plt.subplots(n_rows, n_cols)

        self._goals_ax = np.atleast_2d(self._goals_ax)

        for aa in range(n_goals):
            row = aa // n_cols
            col = aa % n_cols
            self._goals_ax[row, col].set_xlim((-7, 7))
            self._goals_ax[row, col].set_ylim((-7, 7))
            self._goals_ax[row, col].set_title('Multigoal Env. | Goal %d' % aa)
            self._goals_ax[row, col].set_xlabel('x')
            self._goals_ax[row, col].set_ylabel('y0')
            self._goals_ax[row, col].set_aspect('equal', 'box')

        self._plot_goal_position_cost(self._goals_ax)

    def _plot_position_cost(self, ax):
        delta = 0.01
        x_min, x_max = tuple(1.1 * np.array(self.xlim))
        y_min, y_max = tuple(1.1 * np.array(self.ylim))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )
        goal_costs = \
            np.amin([(X - goal_x) ** 2 + (Y - goal_y) ** 2
                    for goal_x, goal_y in self.goal_positions],
                    axis=0)
        costs = goal_costs

        contours = ax.contour(X, Y, costs, 20)
        ax.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        goal = ax.plot(self.goal_positions[:, 0],
                       self.goal_positions[:, 1], 'ro')
        return [contours, goal]

    def _plot_goal_position_cost(self, axs):
        delta = 0.01
        x_min, x_max = tuple(1.1 * np.array(self.xlim))
        y_min, y_max = tuple(1.1 * np.array(self.ylim))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )

        n_cols = 2
        n_goals = self.goal_positions.shape[0]

        goals = [None for _ in range(n_goals)]
        contours = [None for _ in range(n_goals)]

        for aa in range(n_goals):
            row = aa // n_cols
            col = aa % n_cols
            ax = axs[row, col]

            (goal_x, goal_y) = self.goal_positions[aa]
            goal_costs = (X - goal_x) ** 2 + (Y - goal_y) ** 2
            costs = goal_costs

            contours[aa] = ax.contour(X, Y, costs, 20)
            ax.clabel(contours[aa], inline=1, fontsize=10, fmt='%.0f')
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])

            goals[aa] = ax.plot(self.goal_positions[:, 0],
                                self.goal_positions[:, 1], 'ro')

        return [contours, goals]

    @property
    def _fig_exist(self):
        return plt.fignum_exists(self._fig.number)

    @property
    def _goal_fig_exist(self):
        return plt.fignum_exists(self._goals_fig.number)


class PointDynamics(object):
    """
    State: position
    Action: velocity
    """
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action):
        mu_next = state + action
        state_next = mu_next + self.sigma * \
            np.random.normal(size=self.s_dim)
        return state_next