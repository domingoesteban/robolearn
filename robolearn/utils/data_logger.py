""" Data logger.
    Author: C. Finn et al. Code in github.com:cbfinn/gps.git
"""
import logging
try:
   import cPickle as pickle
except:
   import pickle

import datetime

LOGGER = logging.getLogger(__name__)


class DataLogger(object):
    """
    This class pickles data into files and unpickles data from files.
    TODO: Handle logging text to terminal, GUI text, and/or log file at
        DEBUG, INFO, WARN, ERROR, FATAL levels.
    TODO: Handle logging data to terminal, GUI text/plots, and/or data
          files.
    """
    def __init__(self):
        pass

    @staticmethod
    def pickle(filename, data):
        """ Pickle data into file specified by filename. """
        if filename.endswith('.pkl'):
            filename = filename[:-4]

        filename = filename+datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+'.pkl'
        pickle.dump(data, open(filename, 'wb'))
        return filename

    @staticmethod
    def unpickle(filename):
        """ Unpickle dta from file specified by filename. """
        try:
            if filename.endswith('.pkl'):
                return pickle.load(open(filename, 'rb'))
            else:
                return pickle.load(open(filename+'.pkl', 'rb'))
        except IOError:
            print('Unpickle error. Cannot find file: %s', filename)
            return None
