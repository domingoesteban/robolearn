import socket
import psutil
import subprocess
import time
import os
import signal


class MultiRosGazebo(object):
    def __init__(self, n_ros_gz, host='localhost'):
        init_port = 11311
        self.host = host
        self.ros_cores = []
        self.ports = []
        self.total_ros_gz = n_ros_gz

        while len(self.ports) < self.total_ros_gz:
            result = 0
            while result == 0:
                result = self.is_port_open(host, init_port)
                if result == 0:
                    init_port += 1

            print("Running roscore with port %d" % init_port)
            self.ports.append(init_port)
            self.ros_cores.append(self.run_roscore(init_port))


    def stop(self, index=None):
        if index is None:
            index = range(len(self.ros_cores))

        for ii in index:
            print("Killing roscore with port %d" % self.ports[ii])
            os.killpg(os.getpgid(self.ros_cores[ii].pid),
                      signal.SIGTERM)

    @staticmethod
    def is_port_open(host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #result = sock.connect_ex(('127.0.0.1', 11311))
        return sock.connect_ex((host, port))

    @staticmethod
    def run_roscore(port):
        # TODO: Change subprocess.PIPE to a file
        roscore_subprocess = subprocess.Popen(['roscore', '-p', '%d' % port], shell=False, preexec_fn=os.setsid,
                                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #roscore_subprocess.wait()
        time.sleep(1)
        return roscore_subprocess

    def __del__(self):
        multi_ros_gz.stop()


multi_ros_gz = MultiRosGazebo(2)

input('hola')
#multi_ros_gz.stop()

print(multi_ros_gz.ports)