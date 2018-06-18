import sys
import socket
import psutil
import subprocess
import multiprocessing
import threading
import time
import os
import signal
import traceback

from robolearn.old_envs.gazebo_ros_env_interface import GazeboROSEnvInterface

init_roscore_port = 11312
init_gzserver_port = 11347


class RosGazebo(multiprocessing.Process):
    def __init__(self, host='localhost'):
        super(RosGazebo, self).__init__()

        self.host = host
        self.roscore = None
        self.gzserver = None
        self.roslaunch = None
        self.roscore_port = None
        self.gzserver_port = None
        self.env_interface = None

        self.env_vars = os.environ.copy()

        self.close_pipe = multiprocessing.Pipe()
        self.reset_pipe = multiprocessing.Pipe()
        self.action_pipe = multiprocessing.Pipe()
        self.obs_queue = multiprocessing.Queue()
        self.state_queue = multiprocessing.Queue()

        self.wait_reset_thread = None
        self.wait_action_thread = None
        self.update_observation_thread = None
        self.update_state_thread = None

        self.running = True

    def run(self):
        try:
            self.start_all()

            if self.gzserver_port is None:
                gzserver_port = init_gzserver_port
            else:
                gzserver_port = self.gzserver_port
            self.gzserver_port = self.get_available_port(gzserver_port)

            os.environ["ROS_MASTER_URI"] = 'http://%s:%d' % (str(self.host), self.roscore_port)
            os.environ["GAZEBO_MASTER_URI"] = 'http://%s:%d' % (str(self.host), self.gzserver_port)

            self.roslaunch = self.run_roslaunch()

            action_types = list()
            action_topic_infos = list()
            observation_active = list()
            state_active = list()
            from std_msgs.msg import Float64
            from sensor_msgs.msg import JointState
            for ii in range(3):
                action_types.append({'name': 'joint_effort',
                                     'dof': 1})
                action_topic_infos.append({'name': '/manipulator2d/joint'+str(ii)+'_position_controller/command',
                                           'type': Float64,
                                           'freq': 100})
            observation_active.append({'name': 'joint_state',
                                       'type': 'joint_state',
                                       'ros_class': JointState,
                                       'fields': ['position', 'velocity'],
                                       'joints': [0, 1, 2],  # Joint IDs
                                       'ros_topic': '/manipulator2d/joint_states',
                                       })
            state_active.append({'type': 'joint_state',
                                 'fields': ['position', 'velocity'],
                                 'joints': [0, 1, 2]})
            self.env_interface = GazeboROSEnvInterface(action_types=action_types, action_topic_infos=action_topic_infos,
                                                       observation_active=observation_active,
                                                       state_active=state_active)

            # Threads
            self.wait_reset_thread = threading.Thread(target=self.wait_reset, args=[])
            self.wait_reset_thread.start()
            self.wait_action_thread = threading.Thread(target=self.wait_action, args=[])
            self.wait_action_thread.start()
            self.update_observation_thread = threading.Thread(target=self.update_obs, args=[])
            self.update_observation_thread.start()
            self.update_state_thread = threading.Thread(target=self.update_state, args=[])
            self.update_state_thread.start()

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

    def wait_reset(self):
        while True:
            print('Waiting for reset request...')
            reset = self.reset_pipe[1].recv()
            self.env_interface.reset(time=reset[0], freq=reset[1], cond=reset[2])

    def wait_action(self):
        while True:
            print('Waiting for external command...')
            action = self.action_pipe[1].recv()
            self.env_interface.send_action(action)

    def update_obs(self):
        while True:
            if self.obs_queue.empty():
                self.obs_queue.put(self.env_interface.get_observation())

    def update_state(self):
        while True:
            if self.state_queue.empty():
                self.state_queue.put(self.env_interface.get_state())

    def reset(self, time=None, freq=None, cond=0):
        # TODO: FILTER ACTION TO AVOID BAD BEHAVIOR
        self.reset_pipe[0].send((time, freq, cond))

    def send_action(self, action):
        # TODO: FILTER ACTION TO AVOID BAD BEHAVIOR
        self.action_pipe[0].send(action)

    def get_observation(self):
        while self.obs_queue.empty():
            pass
        return self.obs_queue.get()

    def get_state(self):
        while self.state_queue.empty():
            pass
        return self.state_queue.get()

    def start(self):
        super(RosGazebo, self).start()

    def restart(self):
        if self.running is False:
            self.running = True
            self.run()
        else:
            print("RosGazebo is already running")

    def stop(self):
        self.close_pipe[0].send('all')
        self.running = False

    def start_all(self):
        self.start_roscore()
        #self.start_gzserver()

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
        roscore_subprocess = subprocess.Popen(['roscore', '-p', '%d' % port], shell=False, preexec_fn=os.setsid)#,
                                              # env=self.env_vars, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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

    def run_roslaunch(self):
        roslaunch_cmd = ['roslaunch',
                         'manipulator2d_gazebo',
                         'manipulator2d_world.launch']
        gzserver_subprocess = subprocess.Popen(roslaunch_cmd, shell=False, preexec_fn=os.setsid,
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # roscore_subprocess.wait()
        return gzserver_subprocess

    def stop_all(self):
        self.stop_roslaunch()
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

    def stop_roslaunch(self):
        if self.roslaunch is None:
            print("There is not a roslaunch running.")
        else:
            print("Killing roslaunch (PID %d)" % self.roslaunch.pid)
            os.killpg(os.getpgid(self.roslaunch.pid), signal.SIGTERM)
            self.roslaunch = None

    def get_available_port(self, init_port):
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


class MultiRosGazebo(object):
    def __init__(self, n_ros_gz, host='localhost'):
        self.host = host
        self.total_ros_gz = n_ros_gz

        # self.rosgazebos = [RosGazebo(host=self.host) for _ in range(self.total_ros_gz)]

        self.rosgazebos = [None for _ in range(self.total_ros_gz)]

        for ii in range(self.total_ros_gz):
            self.rosgazebos[ii] = RosGazebo('localhost')
            self.start([ii])

    def start(self, index=None):
        if index is None:
            index = range(self.total_ros_gz)

        for ii in index:
            self.rosgazebos[ii].start()
            print("rosgazebo[%d] is running with PID:%d" % (ii, self.rosgazebos[ii].pid))
            time.sleep(2)

    def stop(self, index=None):
        if index is None:
            index = range(self.total_ros_gz)

        for ii in index:
            print("Stop rosgazebo[%d]" % ii)
            #self.rosgazebos[ii].terminate()
            self.rosgazebos[ii].stop()

    def restart(self, index=None):
        if index is None:
            index = range(self.total_ros_gz)

        for ii in index:
            self.rosgazebos[ii].restart()
            print("Restarting rosgazebo[%d] with PID:%d" % (ii, self.rosgazebos[ii].pid))
            time.sleep(2)

    def __del__(self):
        self.stop()


#roslaunch_cmd = '/etc/bash -c source activate py27 && roslaunch manipulator2d_gazebo manipulator2d_world.launch'
#roslaunch_cmd = 'ls'
#gzserver_subprocess = subprocess.Popen('/bin/bash -c ls', shell=True, preexec_fn=os.setsid)#,
#raw_input('borra')

multi_ros_gz = MultiRosGazebo(1)


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

#print('sleeping')
#time.sleep(5)

# multi_ros_gz.start()


time.sleep(1)

raw_input('send_action')

import numpy as np

for ii in range(200):
    print('get observation...')
    print(multi_ros_gz.rosgazebos[0].get_observation())
    print('get state...')
    print(multi_ros_gz.rosgazebos[0].get_state())
    print('sending action...')
    multi_ros_gz.rosgazebos[0].send_action(np.random.randn(3))
    #multi_ros_gz.rosgazebos[0].send_action(np.array([-0.2, 0.1, 0.4]))
    time.sleep(0.1)

raw_input('reset')
multi_ros_gz.rosgazebos[0].reset()

raw_input('stop')
#time.sleep(5)

multi_ros_gz.stop()

time.sleep(1)

#raw_input('restart')
#multi_ros_gz.restart()

#raw_input('stop')
#multi_ros_gz.stop()

raw_input('finish_script')
