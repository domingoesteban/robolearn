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


# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


class MultiRosGazeboServer(object):
    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.total_ros_gz = 0

        # self.rosgazebos = [RosGazebo(host=self.host) for _ in range(self.total_ros_gz)]

        self.rosgazebos = list()  # [None for _ in range(self.total_ros_gz)]

        self.xml_rpc_server = SimpleXMLRPCServer(("localhost", 8000), requestHandler=RequestHandler)

        # Enable: listMethods(), methodHelp(), methodSignature()
        self.xml_rpc_server.register_introspection_functions()

        self.xml_rpc_server.register_function(self.stop, 'stop')

        self.xml_rpc_server.register_function(self.start, 'start')

        self.xml_rpc_server.register_function(self.restart, 'restart')
        
        self.xml_rpc_server.register_function(self.load, 'load')

        self.xml_rpc_server.register_function(self.print_rosgazebos, 'print_rosgazebos')

        # for ii in range(self.total_ros_gz):
        #     self.rosgazebos[ii] = RosGazebo('localhost')
        #     self.start([ii])

    def run(self):
        self.xml_rpc_server.serve_forever()

    def print_rosgazebos(self):
        print_text = ''
        if self.rosgazebos:
            print_text += "ID \t PID \t host \t port \n"
            for ii in range(len(self.rosgazebos)):
                print_text += ('%02d' % ii) + '\t' + str(self.rosgazebos[ii]['pid']) + '\t' + \
                              self.rosgazebos[ii]['host'] + '\t' + str(self.rosgazebos[ii]['port']) + '\n'

            print(print_text)
            return True
        else:
            print("There is not any rosgazebos running!")
            return False

    def load(self, pid, host, port):
        xml_rpc_client = xmlrpc.client.ServerProxy('http://localhost:8000')
        print("Registering process with PID %d, and listen on %s:%d" % (pid, host, port))
        self.rosgazebos.append({'pid': pid, 'host': host, 'port': port, 'client': xml_rpc_client})
        return True

    def start(self, index=None):
        if index == 'all':
            index = range(self.total_ros_gz)

        for ii in index:
            self.rosgazebos[ii].start()
            print("rosgazebo[%d] is running with PID:%d" % (ii, self.rosgazebos[ii].pid))
            time.sleep(2)
        return True

    def stop(self, index='all'):
        if index == 'all':
            index = range(self.total_ros_gz)
        elif not issubclass(type(index), list):
            index = [index]

        for ii in index:
            print("Stop rosgazebo[%d]" % ii)
            #self.rosgazebos[ii].terminate()
            #self.rosgazebos[ii].stop()
            self.rosgazebos[ii]['client'].stop()
        print("DONE")
        return True

    def restart(self, index=None):
        if index == 'all':
            index = range(self.total_ros_gz)

        for ii in index:
            self.rosgazebos[ii].restart()
            print("Restarting rosgazebo[%d] with PID:%d" % (ii, self.rosgazebos[ii].pid))
            time.sleep(2)
        return True

    def __del__(self):
        self.stop()


if __name__ == "__main__":
    multi_ros_gz = MultiRosGazeboServer()

    print("Running MultiROSGazebo Server (%s:%d)" % (multi_ros_gz.host, multi_ros_gz.port))

    multi_ros_gz.run()

