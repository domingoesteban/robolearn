import numpy as np


class CostAction(object):
    def __init__(self, wu, target=None):
        self._wu = wu
        self._target = target

    def eval(self, path):
        actions = path['actions']
        T = len(actions)
        Du = path['actions'][-1].shape[0]
        Dx = path['observations'][-1].shape[0]

        # Code from superball_gps
        if self._target is None:
            target = 0
        else:
            target = np.tile(self._target, (T, 1))
            # target = np.tile(self._hyperparams['target'], (Du, 1))

        # l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1)
        l = 0.5 * np.sum(self._wu * ((actions - target) ** 2),
                         axis=1)

        # lu = self._hyperparams['wu'] * sample_u
        lu = self._wu * (actions - target)
        lx = np.zeros((T, Dx))
        luu = np.tile(np.diag(self._wu), [T, 1, 1])
        lxx = np.zeros((T, Dx, Dx))
        lux = np.zeros((T, Du, Dx))

        return l, lx, lu, lxx, luu, lux
