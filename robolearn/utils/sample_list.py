import pickle
import logging

import numpy as np

#LOGGER = logging.getLogger(__name__)


class SampleList(object):
    """
    Class that handles writes and reads to sample data.
    Inspired by C. Finn code in github.com:cbfinn/gps.git
    """
    def __init__(self, sample_list=None):
        if sample_list is None:
            self._samples = []
        else:
            self._samples = sample_list

    def set_sample_list(self, sample_list):
        self._samples = sample_list

    def add_sample(self, sample):
        self._samples.append(sample)

    def remove_sample(self, idx):
        self._samples.pop(idx)

    def get_obs(self, idx=None, obs_name=None, t=None):
        """ Returns N x T x dX numpy array of states. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_obs(obs_name=obs_name, t=t) for i in idx])

    def get_states(self, idx=None, state_name=None, t=None):
        """ Returns N x T x dX numpy array of states. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_states(state_name=state_name, t=t) for i in idx])

    def get_actions(self, idx=None, t=None):
        """ Returns N x T x dU numpy array of actions. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_acts(t=t) for i in idx])

    #def get_noise(self, idx=None):
    #    """ Returns N x T x dU numpy array of noise generated during rollouts. """
    #    if idx is None:
    #        idx = range(len(self._samples))
    #    return np.asarray([self._samples[i].get(NOISE) for i in idx])

    def get_samples(self, idx=None):
        """ Returns N sample objects. """
        if idx is None:
            idx = range(len(self._samples))
        return [self._samples[i] for i in idx]

    def num_samples(self):
        """ Returns number of samples. """
        return len(self._samples)

    # Convenience methods.
    def __len__(self):
        return self.num_samples()

    def __getitem__(self, idx):
        return self.get_samples([idx])[0]


class PickleSampleWriter(object):
    """
    Pickles samples into data_file.
    Author: C.Finn et al. in github.com:cbfinn/gps.git
    """
    def __init__(self, data_file):
        self._data_file = data_file

    def write(self, samples):
        """ Write samples to data file. """
        with open(self._data_file, 'wb') as data_file:
            pickle.dump(data_file, samples)


#class SysOutWriter(object):
#    """
#    Writes notifications to sysout on sample writes.
#    Author: C.Finn et al. in github.com:cbfinn/gps.git
#    """
#    def __init__(self):
#        pass
#
#    def write(self, samples):
#        """ Write number of samples to sysout. """
#        LOGGER.debug('Collected %d samples', len(samples))
