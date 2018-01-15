import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import sys
import os

from gym import utils
from robolearn.envs.discrete_env import DiscreteEnv

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x12": [
        "FFFFFFFFFFFF",
        "FFFFFFFFFFFF",
        "FFFFFFFFFFFF",
        "SCCCCCCCCCCG"
    ],
    "6x12": [
        "FFFFFFFFFFFF",
        "FFFFFFFFFFFF",
        "FFFFFFFFFFFF",
        "FFFFFFFFFFFF",
        "FFFFFFFFFFFF",
        "SCCCCCCCCCCG"
    ],
}

MAP_BG_COLORS = {'S': 'lightblue', 'G': 'green', 'F': 'white', 'C': 'black'}
MAP_BG_TXT_COLORS = {'S': 'black', 'G': 'black', 'F': 'white', 'C': 'white'}

COLOR_DICT = dict(zip(mcolors.CSS4_COLORS.keys(),
                      [mcolors.hex2color(color)
                       for color in mcolors.CSS4_COLORS.values()]))

REWARD_DICT = {
               'S': -1,
               'G': -1,
               'F': -1,
               'C': -100
               }

ABSORVING_STATES = ['G', 'C']

ACT_NAMES = ['L', 'LD', 'D', 'DR', 'R', 'RU', 'U', 'UL']

IMG_HEIGHT = 240
IMG_WIDTH = 240

SLIPPERY_PERCENTAGE = 0.2


class CliffEnv(DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi', 'rgb_array']}

    def __init__(self, desc=None, map_name="4x4", is_slippery=True,
                 reward_dict=None, nA=4):
        """

        :param desc: 2D array specifying what each grid cell means
                     (used for plotting)
        :param map_name: '4x4' or '8x8'
        :param is_slippery: Surface is slippery or not
        """
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]

        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        if reward_dict is not None:
            reward_dict = REWARD_DICT
        self.reward_dict = reward_dict

        nS = nrow * ncol

        if nA == 4:
            action_names = [ACT_NAMES[ii] for ii in range(4)]
        elif nA == 8:
            action_names = ACT_NAMES
        else:
            raise ValueError

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == 0:  # left
                col = max(col-1, 0)
            elif a == 1:  # down
                row = min(row+1, nrow-1)
            elif a == 2:  # right
                col = min(col+1, ncol-1)
            elif a == 3:  # up
                row = max(row-1, 0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = self.get_letter(row, col)
                    if letter in ABSORVING_STATES:
                        rew = self.get_reward(letter)

                        li.append((1.0, s, rew, True))

                    else:
                        if is_slippery:
                            for b in [(a-1) % 4, a, (a+1) % 4]:
                                new_row, new_col = inc(row, col, b)
                                new_state = to_s(new_row, new_col)
                                new_letter = self.get_letter(new_row, new_col)
                                done = new_letter in ABSORVING_STATES

                                rew = self.get_reward(new_letter)

                                li.append((1 - SLIPPERY_PERCENTAGE
                                           if b == a else SLIPPERY_PERCENTAGE/2,
                                           new_state, rew, done))
                        else:
                            new_row, new_col = inc(row, col, a)
                            new_state = to_s(new_row, new_col)
                            new_letter = self.get_letter(new_row, new_col)
                            done = new_letter in ABSORVING_STATES

                            rew = self.get_reward(new_letter)

                            li.append((1.0, new_state, rew, done))

        super(CliffEnv, self).__init__(nS, nA, P, isd)

        self.fig = None
        self.ax = None
        self.s_draw = None

    def get_letter(self, row, col):
        return self.desc[row, col].decode()

    def get_reward(self, new_letter):
        reward_keys = list(self.reward_dict.keys())

        if new_letter not in reward_keys:
            raise ValueError('Wrong new letter:',
                             new_letter)

        return float(self.reward_dict[new_letter])

    def to_row_col(self, s):
        row = int(s // self.ncol)
        col = int(s % self.ncol)
        return row, col

    def _render(self, mode='human', close=False):
        if close:
            if self.fig is not None:
                plt.close(self.fig)
                self.fig = None
                self.ax = None
                self.s_draw = None
            return
        if mode == 'human':
            if self.fig is None:
                self._plot_backgound()
                self._plot_env()
                plt.ion()
                plt.show()
            else:
                self._plot_env()
            # self.fig.suptitle('Iter %d' % self.internal_counter)
            self.fig.canvas.set_window_title('Frozen Lake environment')
            plt.pause(0.0001)
            return
        else:
            plt.ioff()
            matplotlib.use('Agg')
            self._plot_backgound()
            self._plot_env()
            dpi = self.fig.get_dpi()
            scale_h = float(self.ncol) / max(self.ncol, self.nrow)
            scale_w = float(self.nrow) / max(self.ncol, self.nrow)
            self.fig.set_size_inches(float(IMG_HEIGHT*scale_h)/float(dpi),
                                     float(IMG_WIDTH*scale_w)/float(dpi))
            self.fig.subplots_adjust(bottom=0., left=0., right=1., top=1.)
            extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            self.fig.savefig('/tmp/temporal_frozen_lake_img',
                             format='png', bbox_inches=extent)
            self._render(close=True)
            plt.ion()
            return plt.imread('/tmp/temporal_frozen_lake_img')[:, :, :3]

    def _plot_env(self):
        row, col = self.to_row_col(self.s)
        self._robot_marker(col, row)
        self.fig.canvas.draw()

    def _plot_backgound(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.fig.canvas.draw()

        self.env_color = np.ones((self.nrow, self.ncol, 3))
        for row in range(self.nrow):
            for col in range(self.ncol):
                letter = self.get_letter(row, col)
                self.env_color[row, col, :] = \
                    COLOR_DICT[MAP_BG_COLORS[letter]]

        square_size = 0.5
        self.env_image = self.ax.imshow(self.env_color, interpolation='nearest')
        self.ax.set_xticks(np.arange(self.ncol)-square_size)
        self.ax.set_yticks(np.arange(self.nrow)-square_size)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.xaxis.set_ticks_position('none')
        self.ax.yaxis.set_ticks_position('none')
        for row in range(self.nrow):
            for col in range(self.ncol):
                letter = self.get_letter(row, col)
                self.ax.text(col, row, str(self.get_letter(row, col)),
                             color=COLOR_DICT[MAP_BG_TXT_COLORS[letter]],
                             size=10,  verticalalignment='center',
                             horizontalalignment='center', fontweight='bold')
                if letter == b'S':
                    self._robot_marker(col, row)
        self.ax.grid(color='k', lw=2, ls='-')
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def _robot_marker(self, x, y, color='red'):
        if self.s_draw is not None:
            self.s_draw.remove()

        if self.ncol == 4:
            zoom = 0.03
        else:
            zoom = 0.015
        image = plt.imread(os.path.join(os.path.dirname(__file__),
                                        'robotio.png'))

        for cc in range(3):
            image[:, :, cc] = COLOR_DICT[color][cc]

        im = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
        self.s_draw = self.ax.add_artist(ab)
