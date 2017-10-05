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


class MultiAlgosServer(object):
    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.total_ros_gz = 0

        self.algos = list()  # [None for _ in range(self.total_ros_gz)]

        self.xml_rpc_server = SimpleXMLRPCServer(("localhost", 8000), requestHandler=RequestHandler)

        # Enable: listMethods(), methodHelp(), methodSignature()
        self.xml_rpc_server.register_introspection_functions()

        self.xml_rpc_server.register_function(self.stop, 'stop')

        self.xml_rpc_server.register_function(self.start, 'start')

        self.xml_rpc_server.register_function(self.restart, 'restart')
        
        self.xml_rpc_server.register_function(self.load, 'load')

        self.xml_rpc_server.register_function(self.kill, 'kill')

        self.xml_rpc_server.register_function(self.print_algos, 'print_algos')

        # for ii in range(self.total_ros_gz):
        #     self.algos[ii] = RosGazebo('localhost')
        #     self.start([ii])

    def run(self):
        self.xml_rpc_server.serve_forever()

    def print_algos(self):
        print_text = ''
        if self.algos:
            print_text += "ID \t PID \t host \t port \n"
            for ii in range(len(self.algos)):
                print_text += ('%02d' % ii) + '\t' + str(self.algos[ii]['pid']) + '\t' + \
                              self.algos[ii]['host'] + '\t' + str(self.algos[ii]['port']) + '\n'

        else:
            print_text += "There is not any algos running!"

        return print_text

    def load(self, pid, host, port):
        xml_rpc_client = xmlrpc.client.ServerProxy('http://' + host + ':' + str(port))
        print("Registering process with PID %d, and listen on %s:%d" % (pid, host, port))
        self.algos.append({'pid': pid, 'host': host, 'port': port, 'client': xml_rpc_client})
        return True

    def start(self, index=None):
        if index == 'all':
            index = range(self.total_ros_gz)
        elif not issubclass(type(index), list):
            index = [int(index)]

        for ii in index:
            if ii < len(self.algos):
                self.algos[ii]['client'].start()
                print("algo[%d] is running with PID:%d" % (ii, self.algos[ii]['pid']))
                time.sleep(2)
            else:
                print("algo[%d] does not exist. Was it loaded?" % ii)
        return True

    def stop(self, index='all'):
        if index == 'all':
            index = range(self.total_ros_gz)
        elif not issubclass(type(index), list):
            index = [int(index)]

        for ii in index:
            #self.algos[ii].terminate()
            #self.algos[ii].stop()
            if ii < len(self.algos):
                print("Stop algo[%d]" % ii)
                self.algos[ii]['client'].stop()
            else:
                print("algo[%d] does not exist. Was it loaded?" % ii)
        return True

    def kill(self, index='all'):
        if index == 'all':
            index = range(self.total_ros_gz)
        elif not issubclass(type(index), list):
            index = [int(index)]

        for ii in index:
            if ii < len(self.algos):
                print("Kill algo[%d]" % ii)
                self.algos[ii]['client'].kill()
                self.algos.pop(ii)
            else:
                print("algo[%d] does not exist. Was it loaded?" % ii)
        return True

    def restart(self, index=None):
        if index == 'all':
            index = range(self.total_ros_gz)
        elif not issubclass(type(index), list):
            index = [int(index)]

        for ii in index:
            if ii < len(self.algos):
                self.algos[ii]['client'].restart()
                print("Restarting algo[%d] with PID:%d" % (ii, self.algos[ii]['pid']))
                time.sleep(2)
            else:
                print("algo[%d] does not exist. Was it loaded?" % ii)
        return True

    def __del__(self):
        self.stop()


if __name__ == "__main__":
    multi_algos = MultiAlgosServer()

    print("Running Algorithm Server (%s:%d)" % (multi_algos.host, multi_algos.port))

    multi_algos.run()

