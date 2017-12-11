"""
This project was developed by Peter Chen, Rocky Duan, Pieter Abbeel for the
Berkeley Deep RL Bootcamp, August 2017. Bootcamp website with slides and lecture
videos: https://sites.google.com/view/deep-rl-bootcamp/.

Code adapted from Berkeley Deep RL Class [HW2]
(https://github.com/berkeleydeeprlcourse/homework/blob/c1027d83cd542e67ebed982d44666e0d22a00141/hw2/HW2.ipynb) [(license)](https://github.com/berkeleydeeprlcourse/homework/blob/master/LICENSE).

Copyright 2017 Deep RL Bootcamp Organizers.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


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
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}

MAP_BG_COLORS = {b'S': 'lightblue', b'G': 'green', b'F': 'white', b'H': 'black'}
MAP_BG_TXT_COLORS = {b'S': 'black', b'G': 'black', b'F': 'white', b'H': 'white'}

COLOR_DICT = dict(zip(mcolors.CSS4_COLORS.keys(),
                      [mcolors.hex2color(color)
                       for color in mcolors.CSS4_COLORS.values()]))

IMG_HEIGHT = 240
IMG_WIDTH = 240


class FrozenLakeEnv(DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the
    park when you made a wild throw that left the frisbee out in the middle of
    the lake. The water is mostly frozen, but there are a few holes where the
    ice has melted. If you step into one of those holes, you'll fall into the
    freezing water. At this time, there's an international frisbee shortage,
    so it's absolutely imperative that you navigate across the lake and retrieve
    the disc. However, the ice is slippery, so you won't always move in the
    direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi', 'rgb_array']}

    def __init__(self, desc=None, map_name="4x4", is_slippery=True,
                 reward_dict=None):
        """

        :param desc: 2D array specifying what each grid cell means
                     (used for plotting)
        :param map_name: '4x4' or '8x8'
        :param is_slippery: Frozen surface is slippery or not
        """
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]

        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        nA = 4
        nS = nrow * ncol

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
                    letter = desc[row, col]
                    if letter in b'GH':
                        if reward_dict is not None:
                            if letter == b'G':
                                rew = float(reward_dict['G'])
                            elif letter == b'H':
                                rew = float(reward_dict['H'])
                            else:
                                raise ValueError('Wrong key error. It should be G, S, F or H')
                        else:
                            rew = 0
                        li.append((1.0, s, rew, True))

                    else:
                        if is_slippery:
                            for b in [(a-1) % 4, a, (a+1) % 4]:
                                new_row, new_col = inc(row, col, b)
                                new_state = to_s(new_row, new_col)
                                new_letter = desc[new_row, new_col]
                                done = bytes(new_letter) in b'GH'
                                if reward_dict is not None:
                                    if new_letter == b'G':
                                        rew = float(reward_dict['G'])
                                    elif new_letter == b'S':
                                        rew = float(reward_dict['S'])
                                    elif new_letter == b'H':
                                        rew = float(reward_dict['H'])
                                    elif new_letter == b'F':
                                        rew = float(reward_dict['F'])
                                    else:
                                        raise ValueError('Wrong key error. It'
                                                         'should be '
                                                         'G, S, F or H')
                                else:
                                    rew = float(new_letter == b'G')
                                li.append((0.8 if b == a else 0.1,
                                           new_state, rew, done))
                        else:
                            new_row, new_col = inc(row, col, a)
                            new_state = to_s(new_row, new_col)
                            new_letter = desc[new_row, new_col]
                            done = bytes(new_letter) in b'GH'
                            if reward_dict is not None:
                                if new_letter == b'G':
                                    rew = float(reward_dict['G'])
                                elif new_letter == b'S':
                                    rew = float(reward_dict['S'])
                                elif new_letter == b'H':
                                    rew = float(reward_dict['H'])
                                elif new_letter == b'F':
                                    rew = float(reward_dict['F'])
                                else:
                                    raise ValueError('Wrong key error. It'
                                                     'should be G, S, F or H')
                            else:
                                rew = float(new_letter == b'G')
                            li.append((1.0, new_state, rew, done))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

        self.fig = None
        self.ax = None
        self.s_draw = None

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
            self.fig.set_size_inches(float(IMG_HEIGHT)/float(dpi),
                                     float(IMG_WIDTH)/float(dpi))
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
                letter = self.desc[row, col]
                self.env_color[row, col, :] = COLOR_DICT[MAP_BG_COLORS[letter]]

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
                letter = self.desc[row, col]
                self.ax.text(col, row, str(self.desc[row, col].item().decode()),
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
