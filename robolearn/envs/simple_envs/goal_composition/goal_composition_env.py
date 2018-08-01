"""
Based on Haarnoja sac's multigoal_env.py file

https://github.com/haarnoja/sac
"""

import numpy as np
import matplotlib.pyplot as plt
from robolearn.utils.plots import plt_pause

from robolearn.core.serializable import Serializable
from gym.spaces.box import Box


class GoalCompositionEnv(Serializable):
    """
    Move a 2D point mass to one goal position.

    State: position.
    Action: velocity.
    """
    def __init__(self, goal_reward=10, actuation_cost_coeff=30,
                 distance_cost_coeff=1, init_position=None,
                 init_sigma=0.1, goal_position=None,
                 dynamics_sigma=0, goal_threshold=0.1, horizon=None,
                 log_distance_cost_coeff=1, alpha=1e-6):
        Serializable.__init__(self)
        Serializable.quick_init(self, locals())

        # Point Dynamics
        self._dynamics = PointDynamics(dim=2, sigma=dynamics_sigma)

        # Initial Position
        if init_position is None:
            init_position = (0, 0)
        self.init_mu = np.array(init_position, dtype=np.float32)
        self.init_sigma = init_sigma

        # Goal Position
        if goal_position is None:
            self.goal_position = np.array(
                [5, 5],
                dtype=np.float32
            )
        else:
            self.goal_position = np.array(goal_position, dtype=np.float32)

        # Masks
        self.goal_masks = [[True, True],
                           [True, False],
                           [False, True]]

        # Reward-related Variables
        self._goal_threshold = goal_threshold
        self._goal_reward = goal_reward
        self._action_cost_coeff = actuation_cost_coeff
        self._distance_cost_coeff = distance_cost_coeff
        self._alpha = alpha
        self._log_distance_cost_coeff = log_distance_cost_coeff

        # Maximum Reward
        self._max_rewards = [0, 0, 0]
        temp_output = self.compute_reward(self.goal_position, np.zeros(2))
        self._max_rewards[0] = temp_output[0]
        self._max_rewards[1] = temp_output[2][0]
        self._max_rewards[2] = temp_output[2][1]

        # # Reward: (3.5, 4.5)
        # print(self.compute_reward(np.array([3.5-5.0, 4.5-5.0]),
        #                           np.zeros(2)))
        # input("waaa")

        # Bounds
        self._xlim = (-7, 7)
        self._ylim = (-7, 7)
        self._vel_bound = 1.
        self._observation = None

        # Main Rendering
        self._fig = None
        self._ax = None
        self._env_lines = list()
        self._fixed_plots = None
        self._dynamic_plots = []

        # Subgoals rendering
        self._goals_fig = None
        self._goals_ax = None
        self._dynamic_line = None
        self._dynamic_goals_lines = list()

        # Time-related variables
        self._t_counter = 0
        self._horizon = horizon

        # self.reset()

    def reset(self):
        self._t_counter = 0
        self.clear_all_plots()

        unclipped_observation = self.init_mu + self.init_sigma * \
            np.random.normal(size=self._dynamics.s_dim)
        o_lb, o_ub = self.observation_space.low, self.observation_space.high
        self._observation = np.clip(unclipped_observation, o_lb, o_ub)

        return self._observation

    @property
    def observation_space(self):
        return Box(
            low=np.array((self._xlim[0], self._ylim[0])),
            high=np.array((self._xlim[1], self._ylim[1])),
            shape=None,
            dtype=np.float32
        )

    @property
    def action_space(self):
        return Box(
            low=-self._vel_bound,
            high=self._vel_bound,
            shape=(self._dynamics.a_dim,),
            dtype=np.float32
        )

    def get_current_obs(self):
        return np.copy(self._observation)

    def step(self, action):
        # if sum(action) != 0 and action[0] == action[1] and sum(action**2) != 2:
        #     print(action)
        #     raise ValueError
        # print(action)

        action = action.ravel()

        a_lb, a_ub = self.action_space.low, self.action_space.high
        action = np.clip(action, a_lb, a_ub).ravel()

        next_obs = self._dynamics.forward(self._observation, action)
        o_lb, o_ub = self.observation_space.low, self.observation_space.high
        next_obs = np.clip(next_obs, o_lb, o_ub)

        self._observation = np.copy(next_obs)

        # Compute reward
        reward, reward_composition, reward_multigoal = \
            self.compute_reward(self._observation, action)

        # Compute Done
        done, done_multigoal = self._check_termination()

        return next_obs, reward, done, \
            {'pos': next_obs,
             'reward_vector': reward_composition,
             'reward_multigoal': reward_multigoal,
             'terminal_multigoal': done_multigoal}

    def _check_termination(self):
        if self._horizon is None:
            goal_pos = self.goal_position
            tgt_pos = self._observation

            goal_poses_masked = [goal_pos[mask] for mask in self.goal_masks]
            tgt_poses_masked = [tgt_pos[mask] for mask in self.goal_masks]

            goal_tgt_dists = [np.linalg.norm(goal_mask - tgt_mask)
                              for goal_mask, tgt_mask in zip(goal_poses_masked,
                                                             tgt_poses_masked)]

            done_vect = [goal_tgt_dist <= self._goal_threshold
                         for goal_tgt_dist in goal_tgt_dists]

            return done_vect[0], done_vect[1:]
        else:
            if self._t_counter >= self._horizon:
                return True, [True for _ in self.goal_masks[1:]]
            else:
                return False, [False for _ in self.goal_masks[1:]]

    def compute_reward(self, observation, action):
        # Penalize the L2 norm of acceleration
        action_cost = - self._action_cost_coeff * np.sum(action ** 2)

        # Penalize squared dist to goal
        cur_position = observation

        if cur_position.ndim == 1:
            cur_position_mask = [cur_position[mask] for mask in self.goal_masks]
        elif cur_position.ndim == 2:
            cur_position_mask = [cur_position[mask, mask] for mask in self.goal_masks]
        else:
            raise NotImplementedError

        goal_position_mask = [self.goal_position[mask]
                              for mask in self.goal_masks]
        dist_all_goals = np.array(
            [np.sum((cur_pos - goal_pos) ** 2)
             for cur_pos, goal_pos in zip(cur_position_mask,
                                          goal_position_mask)]
        )
        goal_costs = - self._distance_cost_coeff * dist_all_goals

        # Penalize log dist to goal
        log_goal_costs = - self._log_distance_cost_coeff * \
                         np.log(dist_all_goals + self._alpha)

        # Bonus for being inside threshold area
        dist_all_goals = np.array(
            [np.linalg.norm(goal_pos - cur_pos)
             for cur_pos, goal_pos in zip(cur_position_mask,
                                          goal_position_mask)]
        )
        any_done = dist_all_goals < self._goal_threshold
        bonus_goal_rewards = self._goal_reward * any_done

        # TODO:penalize staying with the log barriers ???

        # Compute Multigoal reward
        reward_multigoal = [goal_cost + log_goal_cost + action_cost + bonus_goal_reward
                            for goal_cost, log_goal_cost, bonus_goal_reward
                            in zip(goal_costs[1:], log_goal_costs[1:],
                                   bonus_goal_rewards[1:])]

        # Subtract Maximum reward
        reward_multigoal = [reward_multi - max_reward
                            for reward_multi, max_reward
                            in zip(reward_multigoal, self._max_rewards[1:])]

        # Compute Main-Task Reward
        reward_composition = [goal_costs[0], log_goal_costs[0], action_cost,
                              bonus_goal_rewards[0], -self._max_rewards[0]]
        reward = np.sum(reward_composition)

        return reward, reward_composition, reward_multigoal

    def render(self, paths=None):
        if self._ax is None:
            self._init_goal_plot()

        if self._goals_ax is None:
            self._init_subgoals_plot()

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
            if self._observation is not None:
                if self._dynamic_line is None:
                    self._dynamic_line, = self._ax.plot(self._observation[0],
                                                        self._observation[1],
                                                        color='b',
                                                        marker='o',
                                                        markersize=2)
                    n_cols = 2
                    n_goals = len(self.goal_masks) - 1
                    for aa in range(n_goals):
                        row = aa // n_cols
                        col = aa % n_cols
                        ax = self._goals_ax[row, col]
                        line, = ax.plot(self._observation[0], self._observation[1],
                                        color='b', marker='o', markersize=2)
                        self._dynamic_goals_lines.append(line)

                else:
                    line = self._dynamic_line
                    line.set_xdata(np.append(line.get_xdata(), self._observation[0]))
                    line.set_ydata(np.append(line.get_ydata(), self._observation[1]))

                    n_goals = len(self.goal_masks) - 1
                    for aa in range(n_goals):
                        line = self._dynamic_goals_lines[aa]
                        line.set_xdata(np.append(line.get_xdata(),
                                                 self._observation[0]))
                        line.set_ydata(np.append(line.get_ydata(),
                                                 self._observation[1]))

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

    def _init_goal_plot(self):
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

        self._plot_goal_position_cost(self._ax)

        plt.show(block=False)

    def _init_subgoals_plot(self):
        plt.ion()
        n_cols = 2
        n_goals = len(self.goal_masks) - 1
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

        self._plot_subgoals_position_cost(self._goals_ax)

    def _plot_goal_position_cost(self, ax):
        delta = 0.01
        x_min, x_max = tuple(1.1 * np.array(self._xlim))
        y_min, y_max = tuple(1.1 * np.array(self._ylim))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )

        dists = (X - self.goal_position[0]) ** 2 + \
                (Y - self.goal_position[1]) ** 2

        goal_costs = -self._distance_cost_coeff * dists

        log_goal_costs = -self._log_distance_cost_coeff * \
                         np.log(dists + self._alpha)

        # Bonus for being inside area
        all_dist_together = np.concatenate(
            (np.expand_dims((X - self.goal_position[0]), axis=-1),
             np.expand_dims((Y - self.goal_position[1]), axis=-1)),
            axis=-1
        )
        dist_goal = np.linalg.norm(all_dist_together, axis=-1)
        done = dist_goal < self._goal_threshold
        bonus_goal_reward = self._goal_reward * done

        costs = goal_costs + log_goal_costs + bonus_goal_reward - \
                self._max_rewards[0]

        contours = ax.contour(X, Y, costs, 20)
        ax.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        goal = ax.plot(self.goal_position[0],
                       self.goal_position[1], 'ro')

        goal_threshold = plt.Circle(self.goal_position,
                                    self._goal_threshold, color='r', alpha=0.5)
        ax.add_artist(goal_threshold)

        x_line = ax.plot([self.goal_position[0], self.goal_position[0]],
                         [y_min, y_max], 'r')
        y_line = ax.plot([x_min, x_max],
                         [self.goal_position[1], self.goal_position[1]], 'r')

        return [contours, goal, goal_threshold]

    def _plot_subgoals_position_cost(self, axs):
        delta = 0.01
        x_min, x_max = tuple(1.1 * np.array(self._xlim))
        y_min, y_max = tuple(1.1 * np.array(self._ylim))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )

        n_cols = 2
        n_goals = len(self.goal_masks) - 1

        goals = [None for _ in range(n_goals)]
        contours = [None for _ in range(n_goals)]

        for aa in range(n_goals):
            row = aa // n_cols
            col = aa % n_cols
            ax = axs[row, col]

            goal_mask = self.goal_masks[aa+1]

            goal_x = self.goal_position[0]
            goal_y = self.goal_position[1]
            dists = ((X - goal_x)*goal_mask[0]) ** 2 + \
                    ((Y - goal_y)*goal_mask[1]) ** 2

            goal_costs = dists * self._distance_cost_coeff

            log_goal_costs = np.log(dists + self._alpha) * \
                             self._log_distance_cost_coeff

            # Bonus for being inside area
            all_together = np.concatenate(
                (np.expand_dims((X - goal_x)*goal_mask[0], axis=-1),
                 np.expand_dims((Y - goal_y)*goal_mask[1], axis=-1)),
                axis=-1
            )
            dist_goal = np.linalg.norm(all_together, axis=-1)
            done = dist_goal < self._goal_threshold
            bonus_goal_reward = done * self._goal_reward

            costs = goal_costs + log_goal_costs - bonus_goal_reward - \
                    self._max_rewards[aa+1]

            contours[aa] = ax.contour(X, Y, -costs, 20)
            ax.clabel(contours[aa], inline=1, fontsize=10, fmt='%.0f')
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])

            goals[aa] = ax.plot(self.goal_position[0],
                                self.goal_position[1], 'ro')

            x_line = ax.plot([self.goal_position[0], self.goal_position[0]],
                             [y_min, y_max], 'r')
            y_line = ax.plot([x_min, x_max],
                             [self.goal_position[1], self.goal_position[1]],
                             'r')

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