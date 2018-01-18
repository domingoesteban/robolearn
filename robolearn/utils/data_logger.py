"""
Data logger
Author: C. Finn et al. Code in github.com:cbfinn/gps.git
"""
try:
   import cPickle as pickle
except:
   import pickle

import datetime
import os


class DataLogger(object):
    """
    This class pickles data into files and unpickles data from files.
    TODO: Handle logging text to terminal, GUI text, and/or log file at
        DEBUG, INFO, WARN, ERROR, FATAL levels.
    TODO: Handle logging data to terminal, GUI text/plots, and/or data
          files.
    """
    def __init__(self, directory_path=None):
        if directory_path is None:
            self.dir_path = './LOG_'+str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        else:
            self.dir_path = directory_path

    def pickle(self, filename, data, dir_path=None):
        """ Pickle data into file specified by filename. """
        if dir_path is None:
            dir_path = self.dir_path

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if filename.endswith('.pkl'):
            filename = filename[:-4]

        #filename = filename+datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+'.pkl'
        pickle.dump(data, open(dir_path+'/'+filename+'.pkl', 'wb'))
        return filename

    def unpickle(self, filename):
        """ Unpickle dta from file specified by filename. """
        try:
            if filename.endswith('.pkl'):
                return pickle.load(open(self.dir_path+'/'+filename, 'rb'))
            else:
                return pickle.load(open(self.dir_path+'/'+filename+'.pkl', 'rb'))
        except IOError:
            print('Unpickle error. Cannot find file: %s', filename)
            return None

