import socket


def is_port_open(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # return sock.connect_ex(('127.0.0.1', 11311))
    return sock.connect_ex((host, port))


def get_available_port(host, init_port):
    result = 0
    while result == 0:
        result = is_port_open(host, init_port)
        if result == 0:
            init_port += 1
    return init_port
