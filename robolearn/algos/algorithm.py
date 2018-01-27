import sys
import os
import copy
import traceback
import logging
from robolearn.utils.print_utils import print_skull


class Algorithm(object):
    def __init__(self, default_hyperparams, hyperparams):
        """
        Algorithm base class constructor. It will set _hyperparams attribute
        with default values and replace only the specific object hyperparameters
        :param default_hyperparams: Default algorithm hyperparameters.
        :param hyperparams: Particular object hyperparameters.
        """
        config = copy.deepcopy(default_hyperparams)
        config.update(hyperparams)
        assert isinstance(config, dict)
        self._hyperparams = config

        self.max_iterations = None

    def run(self, itr_load=None):
        """
        Run the algorithm. If itr_load is specified, first loads the algorithm
        state from that iteration and resumes training at the next iteration.

        Args:
            itr_load: Desired iteration to load algorithm from

        Returns: True/False if the algorithm has finished properly

        """
        run_successfully = True

        try:
            if itr_load is None:
                print('Starting %s algorithm from zero!' % type(self).__name__)
                itr_start = 0
            else:
                itr_start = self._restore_algo_state(itr_load)

            # Start/Continue iteration
            for itr in range(itr_start, self.max_iterations):
                self._iteration(itr)

        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            print("#"*30)
            print("#"*30)
            print_skull()
            print("Panic: Error in %s algorithm!!!!" % type(self).__name__)
            print("#"*30)
            print("#"*30)
            run_successfully = False

        finally:
            # self._end()
            return run_successfully

    def _iteration(self, *args, **kwargs):
        raise NotImplementedError

    def _restore_algo_state(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _setup_logger(logger_name, dir_path, log_file, level=logging.INFO,
                      also_screen=False):

        logger = logging.getLogger(logger_name)

        formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                      "%H:%M:%S")

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        fileHandler = logging.FileHandler(dir_path+log_file, mode='w')
        fileHandler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(fileHandler)

        # if also_screen:
        #     streamHandler = logging.StreamHandler()
        #     streamHandler.setFormatter(formatter)
        #     l.addHandler(streamHandler)

        return logger

