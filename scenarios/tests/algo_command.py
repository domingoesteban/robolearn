#!/usr/bin/env python

import sys
import xmlrpc.client

default_server_host = 'localhost'
default_server_port = '8000'


def check_and_send_command(command, gzros_id):

    server_host = default_server_host
    server_port = default_server_port

    s = xmlrpc.client.ServerProxy('http://' + server_host + ':' + server_port)

    try:
        s.system.listMethods()

    except Exception:
        print("Communication error!")
        print("It was not possible to communicate to server (" + server_host + ':' + server_port + '). ' +
              "Check if it is running.")
        return

    if command.lower() == 'stop':
        s.stop(gzros_id)
    elif command.lower() == 'start':
        s.start(gzros_id)
    elif command.lower() == 'restart':
        s.restart(gzros_id)
    elif command.lower() == 'kill':
        s.kill(gzros_id)
    elif command.lower() == 'list':
        print(s.print_algos())
    elif command.lower() == 'help':
        print("usage: algo_command.py command algo_ID")
        print('Available commands are:')
        methods = [x for x in s.system.listMethods() if not x.startswith('system.')]
        print(methods)
    else:
        print('Wrong command. Available commands are:')
        methods = [x for x in s.system.listMethods() if not x.startswith('system.')]
        print(methods)


if __name__ == "__main__":
    # TODO: Better arguments: https://docs.python.org/2.7/library/argparse.html
    if len(sys.argv) < 2:
        print("usage: algo_commands.py command algos_ID")
    elif len(sys.argv) == 2:
        check_and_send_command(sys.argv[1], None)
    else:
        check_and_send_command(sys.argv[1], sys.argv[2])

