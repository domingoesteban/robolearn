import os
import socket
import threading
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
init_algointerface_port = 12666


# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


class AlgoInterface(object):
    def __init__(self, interface_fcns):
        server_host = 'localhost'
        server_port = str(8000)

        self.host = 'localhost'
        self.port = self.get_available_port(init_algointerface_port)

        #self.close_pipe = multiprocessing.Pipe()
        self.client = xmlrpc.client.ServerProxy('http://' + server_host + ':' + server_port)

        try:
            self.client.system.listMethods()
        except Exception:
            print_text = "Communication error!\n"
            print_text += ("It was not possible to communicate to Algo server (" + server_host + ':' + server_port +
                           '). ' + "Check if it is running.")
            raise AttributeError(print_text)

        self.server = SimpleXMLRPCServer((self.host, self.port), requestHandler=RequestHandler)

        # Enable: listMethods(), methodHelp(), methodSignature()
        self.server.register_introspection_functions()

        for fcn in interface_fcns:
            self.server.register_function(fcn[0], fcn[1])

        self.server_thread = threading.Thread(target=self.listen_xml_rpc, args=[])
        self.server_thread.setDaemon(True)
        self.server_thread.start()

        pid = os.getpid()
        self.client.load(pid, self.host, self.port)

    def listen_xml_rpc(self):
        self.server.serve_forever()

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
