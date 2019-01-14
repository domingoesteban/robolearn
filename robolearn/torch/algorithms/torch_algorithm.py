import abc
from typing import Iterable

from robolearn.torch.core import PyTorchModule


class TorchAlgorithm(object):
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def torch_models(self) -> Iterable[PyTorchModule]:
        # # type: (None) -> Iterable[PyTorchModule]
        pass

    def training_mode(self, mode):
        for model in self.torch_models:
            model.train(mode)

    def cuda(self, device=None):
        for model in self.torch_models:
            model.cuda(device)

    def cpu(self):
        for model in self.torch_models:
            model.cpu()
