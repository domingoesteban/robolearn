import sys
import socket
import subprocess
import multiprocessing
from multiprocessing.managers import BaseManager
import threading
import time
import os
import signal
import traceback
import rostopic

from robolearn.envs.environment import Environment
from robolearn.envs.gazebo_ros_env_interface import GazeboROSEnvInterface
from robolearn.utils.general.network_utils import get_available_port

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
INIT_XML_RPC_PORT = 18305


class GazeboROSEnv(Environment):
    def __init__(self, ros_actions=None, ros_observations=None, state_active=None,
                 host='localhost', roscore_port=None, gzserver_port=None, ros_commands=None):
        super(Environment, self).__init__()

        self._reset_pipe = multiprocessing.Pipe()
        self._close_pipe = multiprocessing.Pipe()
        self._action_pipe = multiprocessing.Pipe()
        self._obs_queue = multiprocessing.Queue()

        self.xml_rpc_port = get_available_port(host, INIT_XML_RPC_PORT)

        self.gz_ros_process = GazeboROSProcess(ros_actions=ros_actions,
                                               ros_observations=ros_observations, state_active=state_active,
                                               ros_commands=ros_commands,
                                               reset_pipe=self._reset_pipe, close_pipe=self._close_pipe,
                                               action_pipe=self._action_pipe, obs_queue=self._obs_queue,
                                               host=host, roscore_port=roscore_port, gzserver_port=gzserver_port,
                                               xml_rpc_port=self.xml_rpc_port)
        self.gz_ros_process.start()

        self.xml_rpc_client = xmlrpc.client.ServerProxy('http://' + host + ':' + str(self.xml_rpc_port),
                                                        allow_none=True)

    def send_action(self, action):
        # TODO: FILTER ACTION TO AVOID BAD BEHAVIOR
        self._action_pipe[0].send(action)

    def get_observation(self):
        while self._obs_queue.empty():
            pass
        return self._obs_queue.get()

    def get_action_dim(self):
        """
        Return the environment's action dimension.
        :return: Action dimension
        :rtype: int
        """
        # return self._action_dim.value
        return self.xml_rpc_client.get_action_dim()

    def get_obs_dim(self):
        """
        Return the environment's observation dimension.
        :return: Observation dimension
        :rtype: int
        """
        return self.xml_rpc_client.get_obs_dim()

    def get_state_dim(self):
        """
        Return the environment's state dimension.
        :return: State dimension
        :rtype: int
        """
        return self.xml_rpc_client.get_state_dim()

    def get_obs_info(self, name=None):
        return self.xml_rpc_client.get_obs_info(name)

    def get_state_info(self, name=None):
        return self.xml_rpc_client.get_state_info(name)

    def __del__(self):
        self.gz_ros_process.stop()


# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


class GazeboROSProcess(multiprocessing.Process):
    def __init__(self, ros_actions=None, ros_observations=None, state_active=None, ros_commands=None,
                 host='localhost', roscore_port=None, gzserver_port=None, xml_rpc_port=INIT_XML_RPC_PORT,
                 reset_pipe=None, close_pipe=None, action_pipe=None, obs_queue=None):

        super(GazeboROSProcess, self).__init__()

        self.host = host
        self.roscore = None
        self.gzserver = None
        self.roscore_port = roscore_port
        self.gzserver_port = gzserver_port
        self.env_interface = None
        self._xml_rpc_server = None
        self.xml_rpc_port = xml_rpc_port

        # Commands to be executed in different processes
        if ros_commands is None:
            raise AttributeError("ros_commands not defined!")
        self.ros_commands = ros_commands
        self.ros_cmd_processes = [None for _ in self.ros_commands]

        # Subclass should create this lists before start()
        self.action_active = ros_actions
        self.observation_active = ros_observations
        self.state_active = state_active
        if self.action_active is None:
            raise AttributeError("self.action_active not defined!")
        if self.observation_active is None:
            raise AttributeError("self.observation_active not defined!")
        if self.state_active is None:
            raise AttributeError("self.state_active not defined!")

        # Shared pipes/queues
        self.reset_pipe = reset_pipe
        self.close_pipe = close_pipe
        self.action_pipe = action_pipe
        self.obs_queue = obs_queue
        if self.reset_pipe is None:
            raise AttributeError("self.reset_pipe not defined!")
        if self.close_pipe is None:
            raise AttributeError("self.close_pipe not defined!")
        if self.action_pipe is None:
            raise AttributeError("self.action_pipe not defined!")
        if self.obs_queue is None:
            raise AttributeError("self.obs_queue not defined!")

        # Threads
        self._wait_reset_thread = None
        self._wait_action_thread = None
        self._update_observation_thread = None
        self._xml_rpc_server_thread = None

        self.env_vars = os.environ.copy()

        self.roscore_running = False
        self.running = False

    def run(self):
        try:
            self.start_roscore()
            self.wait_until_roscore_running()
            print("roscore is running!!")

            if self.gzserver_port is None:
                gzserver_port = init_gzserver_port
            else:
                gzserver_port = self.gzserver_port
            self.gzserver_port = get_available_port(self.host, gzserver_port)

            os.environ["ROS_MASTER_URI"] = 'http://%s:%d' % (str(self.host), self.roscore_port)
            os.environ["GAZEBO_MASTER_URI"] = 'http://%s:%d' % (str(self.host), self.gzserver_port)

            for cc, ros_cmd in enumerate(self.ros_commands):
                self.ros_cmd_processes[cc] = self.run_cmd_subprocess(ros_cmd)

            self.env_interface = GazeboROSEnvInterface(action_active=self.action_active,
                                                       observation_active=self.observation_active,
                                                       state_active=self.state_active)

            self._xml_rpc_server = SimpleXMLRPCServer(("localhost", self.xml_rpc_port), requestHandler=RequestHandler,
                                                      allow_none=True)
            self._xml_rpc_server.register_function(self.env_interface.get_action_dim, 'get_action_dim')
            self._xml_rpc_server.register_function(self.env_interface.get_obs_dim, 'get_obs_dim')
            self._xml_rpc_server.register_function(self.env_interface.get_state_dim, 'get_state_dim')
            self._xml_rpc_server.register_function(self.env_interface.get_obs_info, 'get_obs_info')
            self._xml_rpc_server.register_function(self.env_interface.get_state_info, 'get_state_info')

            # Threads
            self._wait_reset_thread = threading.Thread(target=self.wait_reset, args=[])
            self._wait_reset_thread.start()
            self._wait_action_thread = threading.Thread(target=self.wait_action, args=[])
            self._wait_action_thread.start()
            self._update_observation_thread = threading.Thread(target=self.update_obs, args=[])
            self._update_observation_thread.start()
            self._xml_rpc_server_thread = threading.Thread(target=self._xml_rpc_server.serve_forever, args=[])
            self._xml_rpc_server_thread.start()

            # Block function
            close_option = self.close_pipe[1].recv()

            if close_option == 'all':
                self.stop_all()
            else:
                raise ValueError("Wrong close_option %s" % close_option)

        except Exception as e:
            print("Error in RosGazebo with PID:%d" % self.pid)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=2, file=sys.stdout)
            self.stop_all()

    def is_running(self):
        return self.roscore_running

    def wait_until_roscore_running(self):
        while not self.roscore_running:
            try:
                rostopic.get_topic_class('/rosout')
                self.roscore_running = True
            except rostopic.ROSTopicIOException as e:
                self.roscore_running = False

    def wait_reset(self):
        while True:
            print('Waiting for reset command...')
            reset = self.reset_pipe[1].recv()
            self.env_interface.reset(time=reset[0], freq=reset[1], cond=reset[2])  # (time, freq, cond)

    def wait_action(self):
        while True:
            print('Waiting for external command...')
            action = self.action_pipe[1].recv()
            self.env_interface.send_action(action)

    def update_obs(self):
        while True:
            if self.obs_queue.empty():
                self.obs_queue.put(self.env_interface.get_observation())

    def start(self):
        super(GazeboROSProcess, self).start()

    def restart(self):
        if self.running is False:
            self.running = True
            self.run()
        else:
            print("RosGazebo is already running")

    def reset(self, time=None, freq=None, cond=0):
        self.reset_pipe[0].send((time, freq, cond))

    def stop(self):
        self.close_pipe[0].send('all')
        self.running = False

    def start_roscore(self):
        # Run roscore
        if self.roscore_port is None:
            roscore_port = init_roscore_port
        else:
            roscore_port = self.roscore_port

        roscore_port = get_available_port(self.host, roscore_port)

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

        gzserver_port = get_available_port(self.host, gzserver_port)
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

    def run_cmd_subprocess(self, cmd):
        cmd_subprocess = subprocess.Popen(cmd, shell=False, preexec_fn=os.setsid)#,
                                          # stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # roscore_subprocess.wait()
        return cmd_subprocess

    def stop_all(self):
        self.stop_cmd_subprocess()
        # self.stop_gzserver()
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

    def stop_cmd_subprocess(self):
        for cc, cmd_subprocess in enumerate(self.ros_cmd_processes):
            if cmd_subprocess is None:
                print("There is not a cmd_subprocess running.")
            else:
                print("Killing cmd_subprocess:%s (PID %d)" % (self.ros_commands[cc],
                                                              cmd_subprocess.pid))
                os.killpg(os.getpgid(cmd_subprocess.pid), signal.SIGTERM)
                cmd_subprocess = None

    def __del__(self):
        self.stop()
