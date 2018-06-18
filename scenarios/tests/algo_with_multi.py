import sys
import os
import time
import threading
import numpy as np
from robolearn.old_utils.algo_interface import AlgoInterface
from robolearn.old_envs.manipulator2d.manipulator2d_env import Manipulator2dEnv
from robolearn.old_utils.ros_utils import get_available_port

init_roscore_port = 11312
init_gzserver_port = 11347


class SimpleRLAlgo(object):
    def __init__(self, N, T, ts):
        interface_fcns = [(self.stop, 'stop'), (self.start, 'start'), (self.restart, 'restart'),
                          (self.is_running_fcn, 'is_running'), (self.kill_me, 'kill')]
        self.algo_interface = AlgoInterface(interface_fcns)
        self.n_iterations = N
        self.T = T
        self.Ts = ts
        self.is_running = False
        self.is_training = False
        self.is_finished = False

        self.total_gz_ros = 2

        self.rosgazebos = [None for _ in range(self.total_gz_ros)]
        self.roscore_ports = [None for _ in range(self.total_gz_ros)]
        self.gzserver_ports = [None for _ in range(self.total_gz_ros)]

        for ii in range(self.total_gz_ros):
            if ii == 0:
                last_roscore_port = init_roscore_port
                last_gzserver_port = init_gzserver_port
            else:
                last_roscore_port = self.roscore_ports[ii-1] + 1
                last_gzserver_port = self.gzserver_ports[ii-1] + 1
            self.roscore_ports[ii] = get_available_port('localhost', last_roscore_port)
            self.gzserver_ports[ii] = get_available_port('localhost', last_gzserver_port)
            self.rosgazebos[ii] = Manipulator2dEnv('localhost', roscore_port=self.roscore_ports[ii],
                                                   gzserver_port=self.gzserver_ports[ii])
            self.rosgazebos[ii].start()

        self.running_thread = threading.Thread(target=self.running, args=[])
        self.running_thread.setDaemon(True)
        self.running_thread.start()

    def start(self):
        print("This is starting")
        self.is_running = True
        return True

    def running(self):
        while not self.is_finished:
            if self.is_running:
                self.is_training = True
                for nn in range(self.n_iterations):
                    if self.is_running is False:
                        break
                    # Interaction
                    for ii in range(self.total_gz_ros):
                        self.rosgazebos[ii].reset(time=None, freq=None, cond=0)
                    for t in range(self.T):
                        if self.is_running is False:
                            break

                        for ii in range(self.total_gz_ros):
                            # get obs/state
                            print("State env[%d]: %s" % (ii, self.rosgazebos[ii].get_observation()))
                            # act
                            self.rosgazebos[ii].send_action(np.random.randn(3))

                        print("Iteration %d/%d, time=%d/%d" % (nn+1, self.n_iterations, t+1, self.T))
                        time.sleep(self.Ts)
                    # Evaluation

                    # Update

                self.is_training = False

    def restart(self):
        print("This is restarting")
        self.stop()
        while self.is_training:
            pass

        self.start()
        return True

    def stop(self):
        print("This is stopping")
        self.is_running = False
        return True

    def is_running_fcn(self):
        print("Is this running?: %s " % self.is_running)
        return self.is_running

    def kill_me(self):
        print("This is killing itself!!")
        self.finish()
        for ii in range(self.total_gz_ros):
            self.rosgazebos[ii].stop()
        self.stop()
        del self
        return True

    def finish(self):
        self.is_finished = True


if __name__ == "__main__":

    try:
        simple_algo = SimpleRLAlgo(20, 100, 0.2)

        simple_algo.start()

        while not simple_algo.is_finished:
            pass

    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    print("This algorithm has been finished!!")

