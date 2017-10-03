import socket
import psutil
import subprocess
import multiprocessing
import threading
import time
import os
import signal
import xmlrpc.client

try:
    from xmlrpc.server import SimpleXMLRPCServer  # Python 3
except ImportError:
    from SimpleXMLRPCServer import SimpleXMLRPCServer  # Python 2

try:
    from xmlrpc.server import SimpleXMLRPCRequestHandler
except ImportError:
    from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler

init_roscore_port = 11312
init_gzserver_port = 11347


# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


class RosGazebo(object):
    def __init__(self, host='localhost', port=12666):

        self.host = host
        self.roscore = None
        self.gzserver = None
        self.roslaunch = None
        self.roscore_port = None
        self.gzserver_port = None

        self.env_vars = os.environ.copy()

        #self.close_pipe = multiprocessing.Pipe()
        self.client = xmlrpc.client.ServerProxy('http://localhost:8000')
        self.server = SimpleXMLRPCServer((host, port), requestHandler=RequestHandler)
        
        # Enable: listMethods(), methodHelp(), methodSignature()
        self.server.register_introspection_functions()
        self.server.register_function(self.stop, 'stop')
        self.server.register_function(self.start, 'start')
        self.server.register_function(self.restart, 'restart')

        self.server_thread = threading.Thread(target=self.listen_xml_rpc(), args=[])
        self.server_thread.setDaemon(True)
        self.server_thread.start()

        pid = os.getpid()
        self.client.load(pid, host, port)

        self.running = True

    def listen_xml_rpc(self):
        self.server.serve_forever()

    def start(self):
        print("You call start")
        return True
        self.start_all()

        if self.gzserver_port is None:
            gzserver_port = init_gzserver_port
        else:
            gzserver_port = self.gzserver_port
        self.gzserver_port = self.get_available_port(gzserver_port)

        os.environ["ROS_MASTER_URI"] = 'http://%s:%d' % (str(self.host), self.roscore_port)
        os.environ["GAZEBO_MASTER_URI"] = 'http://%s:%d' % (str(self.host), self.gzserver_port)

        self.roslaunch = self.run_roslaunch()

        # Block function
        close_option = self.close_pipe[1].recv()

        if close_option == 'all':
            self.stop_all()
        else:
            raise ValueError("Wrong close_option %s" % close_option)

    def restart(self):
        print("You call restart")
        return True
        if self.running is False:
            self.running = True
            self.start()
        else:
            print("RosGazebo is already running")

    def stop(self):
        print("You call stop")
        return True
        self.close_pipe[0].send('all')
        self.running = False

    def start_all(self):
        print("You call start_all")
        return True
        self.start_roscore()
        #self.start_gzserver()

    def start_roscore(self):
        print("You call start_roscore")
        return True
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
        print("You call run_roscore")
        return True
        # TODO: Change subprocess.PIPE to a file
        self.env_vars["ROS_MASTER_URI"] = 'http://%s:%d' % (str(self.host), port)
        roscore_subprocess = subprocess.Popen(['roscore', '-p', '%d' % port], shell=False, preexec_fn=os.setsid,
                                              env=self.env_vars, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        #roscore_subprocess.wait()
        return roscore_subprocess

    def start_gzserver(self):
        print("You call start_gzserver")
        return True
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
        print("You call run_gzserver")
        return True
        self.env_vars["GAZEBO_MASTER_URI"] = 'http://%s:%d' % (str(self.host), gz_port)
        gzserver_subprocess = subprocess.Popen(['rosrun', 'gazebo_ros', 'gzserver'], shell=False, preexec_fn=os.setsid,
                                               env=self.env_vars, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # roscore_subprocess.wait()
        return gzserver_subprocess

    def run_roslaunch(self):
        print("You call run_roslaunch")
        return True
        roslaunch_cmd = ['roslaunch',
                         'manipulator2d_gazebo',
                         'manipulator2d_world.launch']
        gzserver_subprocess = subprocess.Popen(roslaunch_cmd, shell=False, preexec_fn=os.setsid,
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # roscore_subprocess.wait()
        return gzserver_subprocess

    def stop_all(self):
        print("You call stop_all")
        return True
        self.stop_roslaunch()
        # self.stop_gzserver()
        self.stop_roscore()

    def stop_gzserver(self):
        print("You call stop_gzserver")
        return True
        if self.gzserver_port is None:
            print("There is not a gzserver running.")
        else:
            print("Killing gzserver with port %d" % self.gzserver_port)
            os.killpg(os.getpgid(self.gzserver.pid), signal.SIGTERM)
            self.gzserver = None
            self.gzserver_port = None

    def stop_roscore(self):
        print("You call stop_roscore")
        return True
        if self.roscore_port is None:
            print("There is not a roscore running.")
        else:
            print("Killing roscore with port %d" % self.roscore_port)
            os.killpg(os.getpgid(self.roscore.pid), signal.SIGTERM)
            self.roscore = None
            self.roscore_port = None

    def stop_roslaunch(self):
        print("You call stop_roslaunch")
        return True
        if self.roslaunch is None:
            print("There is not a roslaunch running.")
        else:
            print("Killing roslaunch (PID %d)" % self.roslaunch.pid)
            os.killpg(os.getpgid(self.roslaunch.pid), signal.SIGTERM)
            self.roslaunch = None

    def get_available_port(self, init_port):
        print("You call get_available_port")
        return True
        result = 0
        while result == 0:
            result = self.is_port_open(self.host, init_port)
            if result == 0:
                init_port += 1
        return init_port

    @staticmethod
    def is_port_open(host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # return sock.connect_ex(('127.0.0.1', 11311))
        return sock.connect_ex((host, port))

    #def __del__(self):
    #    print('termino MultiROS')
    #    self.stop_all()


if __name__ == "__main__":
    ros_gazebo = RosGazebo()

    print("Done")

