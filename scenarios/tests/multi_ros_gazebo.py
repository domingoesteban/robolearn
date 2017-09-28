import socket
import psutil
import subprocess
import multiprocessing
import time
import os
import signal

init_roscore_port = 11312
init_gzserver_port = 11347


class RosGazebo(object):
    def __init__(self, host='localhost'):
        self.host = host
        self.roscore = None
        self.gzserver = None
        self.roscore_port = None
        self.gzserver_port = None

        self.env_vars = os.environ.copy()

        self.start()

        while True:
            time.sleep(10)

    def start(self):
        self.start_roscore()
        self.start_gzserver()

    def start_roscore(self):
        # Run roscore
        if self.roscore_port is None:
            roscore_port = init_roscore_port
        else:
            roscore_port = self.roscore_port

        roscore_port = self.get_available_port(roscore_port)

        if self.roscore is None:
            print("Running roscore with port %d" % roscore_port)
            self.roscore_port = roscore_port
            self.roscore = self.run_roscore(roscore_port)

            time.sleep(1)  # Sleeping so the roscore ports can be opened

    def run_roscore(self, port):
        # TODO: Change subprocess.PIPE to a file
        self.env_vars["ROS_MASTER_URI"] = 'http://%s:%d' % (str(self.host), port)
        roscore_subprocess = subprocess.Popen(['roscore', '-p', '%d' % port], shell=False, preexec_fn=os.setsid,
                                              env=self.env_vars, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #roscore_subprocess.wait()
        return roscore_subprocess

    def start_gzserver(self):
        if self.gzserver_port is None:
            gzserver_port = init_gzserver_port
        else:
            gzserver_port = self.gzserver_port

        gzserver_port = self.get_available_port(gzserver_port)
        roscore_port = self.roscore_port

        if self.roscore is None:
            print("Error running gzserver. There is not a roscore running at port %d." % roscore_port)
        else:
            print("Running gzserver with port %d." % gzserver_port)
            self.gzserver_port = gzserver_port
            self.gzserver = self.run_gzserver(gzserver_port, roscore_port)

    def run_gzserver(self, gz_port, roscore_port):
        self.env_vars["GAZEBO_MASTER_URI"] = 'http://%s:%d' % (str(self.host), gz_port)
        gzserver_subprocess = subprocess.Popen(['rosrun', 'gazebo_ros', 'gzserver'], shell=False, preexec_fn=os.setsid,
                                               env=self.env_vars, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # roscore_subprocess.wait()
        return gzserver_subprocess

    def stop(self):
        self.stop_gzserver()
        self.stop_roscore()

    def stop_gzserver(self):
        if self.gzserver_port is None:
            print("There is not a gzserver running.")
        else:
            print("Killing gzserver with port %d" % self.gzserver_port)
            os.killpg(os.getpgid(self.gzserver.pid), signal.SIGTERM)
            self.gzserver = None
            self.gzserver_port = None

    def stop_roscore(self):
        if self.roscore_port is None:
            print("There is not a roscore running.")
        else:
            print("Killing roscore with port %d" % self.roscore_port)
            os.killpg(os.getpgid(self.roscore.pid), signal.SIGTERM)
            self.roscore = None
            self.roscore_port = None

    @staticmethod
    def is_port_open(host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # return sock.connect_ex(('127.0.0.1', 11311))
        return sock.connect_ex((host, port))

    def get_available_port(self, init_port):
        result = 0
        while result == 0:
            result = self.is_port_open(self.host, init_port)
            if result == 0:
                init_port += 1
        return init_port

    def __del__(self):
        self.stop()


class MultiRosGazebo(object):
    def __init__(self, n_ros_gz, host='localhost'):
        self.host = host
        self.total_ros_gz = n_ros_gz

        # self.rosgazebos = [RosGazebo(host=self.host) for _ in range(self.total_ros_gz)]

        self.rosgazebos = [None for _ in range(self.total_ros_gz)]

        for ii in range(self.total_ros_gz):
            self.rosgazebos[ii] = multiprocessing.Process(target=RosGazebo, args=['localhost'])
            self.rosgazebos[ii].start()
            time.sleep(1)

        # self.start()

    def start(self, index=None):
        if index is None:
            index = range(self.total_ros_gz)

        for ii in index:
            print("Start rosgazebo[%d]" % ii)
            self.rosgazebos[ii].start()

    def stop(self, index=None):
        if index is None:
            index = range(self.total_ros_gz)

        for ii in index:
            print("Stop rosgazebo[%d]" % ii)
            self.rosgazebos[ii].stop()

    def __del__(self):
        self.stop()


multi_ros_gz = MultiRosGazebo(2)


#for ii in range(multi_ros_gz.total_ros_gz):
#    print(multi_ros_gz.rosgazebos[ii].roscore_port)
#    print(multi_ros_gz.rosgazebos[ii].gzserver_port)

#import sys
#def background_imports(host, port):
#    import talker as talk
#    modulenames = set(sys.modules)&set(globals())
#    allmodules = [sys.modules[name] for name in modulenames]
#    print("AAAAAAAAAAAAAAAAAAAAAA")
#    for ii in allmodules:
#        print(ii)
#    print("AAAAAAAAAAAAAAAAAAAAAA")
#    os.environ["ROS_MASTER_URI"] = 'http://%s:%d' % (str(host), port)
#    #input('aaa')
#    #prueba1 = talk.talker()

#from talker_thread import background_imports

#import threading
#import multiprocessing
#lock = multiprocessing.Lock()
#for ii in range(multi_ros_gz.total_ros_gz):
#    port = multi_ros_gz.rosgazebos[ii].roscore_port
#    #thread = threading.Thread(target=background_imports, args=['localhost', port])
#    #thread.setDaemon(True)
#    #thread.start()
#    #thread.join()
#    process = multiprocessing.Process(target=background_imports, args=['localhost', port])
#    process.start()

raw_input('')

#for ii in range(multi_ros_gz.total_ros_gz):
#    port = multi_ros_gz.rosgazebos[ii].roscore_port
#    print(port)
#    print("$$$$")

#multi_ros_gz.stop()
