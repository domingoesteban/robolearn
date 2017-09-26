import socket
import psutil
import subprocess
import time
import os
import signal

init_roscore_port = 11312
init_gzserver_port = 11347

class MultiRosGazebo(object):
    def __init__(self, n_ros_gz, host='localhost'):
        self.host = host
        self.total_ros_gz = n_ros_gz

        self.ros_cores = [None for _ in range(self.total_ros_gz)]
        self.gz_servers = [None for _ in range(self.total_ros_gz)]
        self.roscore_ports = [None for _ in range(self.total_ros_gz)]
        self.gzserver_ports = [None for _ in range(self.total_ros_gz)]

        self.start()

    def start(self, index=None):

        if index is None:
            index = range(self.total_ros_gz)

        roscore_port = init_roscore_port
        gzserver_port = init_gzserver_port

        # Run roscore
        for ii in index:
            result = 0
            while result == 0:
                result = self.is_port_open(self.host, roscore_port)
                if result == 0:
                    roscore_port += 1

            print("Running roscore with port %d" % roscore_port)
            self.roscore_ports[ii] = roscore_port
            self.ros_cores[ii] = self.run_roscore(roscore_port)

            time.sleep(1)

        # Run gzserver
        for ii in index:
            result = 0
            while result == 0:
                result = self.is_port_open(self.host, gzserver_port)
                if result == 0:
                    gzserver_port += 1

            print("Running gzserver with port %d" % gzserver_port)
            self.gzserver_ports[ii] = gzserver_port
            roscore_port = self.roscore_ports[len(self.gzserver_ports)-1]
            self.gz_servers[ii] = self.run_gzserver(init_gzserver_port, roscore_port)

    def stop(self, index=None):
        if index is None:
            index = range(self.total_ros_gz)

        for ii in index:
            print("Killing roscore with port %d" % self.roscore_ports[ii])
            os.killpg(os.getpgid(self.ros_cores[ii].pid),
                      signal.SIGTERM)
            print("Killing gzserver with port %d" % self.gzserver_ports[ii])
            os.killpg(os.getpgid(self.gz_servers[ii].pid),
                      signal.SIGTERM)

    @staticmethod
    def is_port_open(host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #result = sock.connect_ex(('127.0.0.1', 11311))
        return sock.connect_ex((host, port))

    def run_roscore(self, port):
        # TODO: Change subprocess.PIPE to a file
        new_env = os.environ.copy()
        new_env["ROS_MASTER_URI"] = 'http://%s:%d' % (str(self.host), port)
        roscore_subprocess = subprocess.Popen(['roscore', '-p', '%d' % port], shell=False, preexec_fn=os.setsid,
                                              env=new_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #roscore_subprocess.wait()
        return roscore_subprocess

    def run_gzserver(self, gz_port, roscore_port):
        new_env = os.environ.copy()
        new_env["ROS_MASTER_URI"] = 'http://%s:%d' % (str(self.host), roscore_port)
        new_env["GAZEBO_MASTER_URI"] = 'http://%s:%d' % (str(self.host), gz_port)
        gzserver_subprocess = subprocess.Popen(['rosrun', 'gazebo_ros', 'gzserver'], shell=False, preexec_fn=os.setsid,
                                               env=new_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #roscore_subprocess.wait()
        time.sleep(1)
        return gzserver_subprocess

    def __del__(self):
        self.stop()


multi_ros_gz = MultiRosGazebo(2)

input('hola')
#multi_ros_gz.stop()
