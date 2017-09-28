import sys
import os
from plumbum import local

import rosgraph

def _init_node_params(argv, node_name):
    """
    Uploads private params to the parameter server. Private params are specified
    via command-line remappings.

    @raises: ROSInitException
    """

    # #1027: load in param name mappings
    params = load_command_line_node_params(argv)
    for param_name, param_value in params.items():
        #logdebug("setting param %s to %s"%(param_name, param_value))
        set_param(rosgraph.names.PRIV_NAME + param_name, param_value)



def background_imports(host, port):
    #modulenames = set(sys.modules)&set(globals())
    #allmodules = [sys.modules[name] for name in modulenames]
    #print("AAAAAAAAAAAAAAAAAAAAAA")
    #for ii in allmodules:
    #    print(ii)
    #print("AAAAAAAAAAAAAAAAAAAAAA")
    #os.environ["ROS_MASTER_URI"] = 'http://%s:%d' % (str(host), port)
    #print(os.environ["ROS_MASTER_URI"])

    print("@@@@@@ %s" % str(os.environ['ROS_MASTER_URI']))
    new_env = os.environ#.copy()
    new_env["ROS_MASTER_URI"] = 'http://%s:%d' % (str(host), port)

    #from plumbum import local
    #with local.env(ROS_MASTER_URI='http://%s:%d' % (str(host), port)):
    #    import rospy
    #    rospy.init_node('holaaa', anonymous=True)

    #from rospy.impl.init import start_node
    #import rospy.core as core
    #node = start_node(new_env, "/prueba_%d" % port, master_uri=None, port=0, tcpros_port=0)
    #core.set_node_uri(node.uri)
    #core.add_shutdown_hook(node.shutdown)
    #if core.is_shutdown():
    #    logger.warn("aborting node initialization as shutdown has been triggered")
    #    raise rospy.exceptions.ROSInitException("init_node interrupted before it could complete")

    #print("**********%s" % str(type(node)))
    #print(core.get_node_uri())
    print("&&&&&&&&&& %s" % str(os.getpid()))
    print("^^^^^^^^^^ %s" % str(os.environ['ROS_MASTER_URI']))
    #print(client.get_master().search_param())
    import talker as talk
    add_to_topic = os.getpid()
    prueba1 = talk.talker(add_to_topic, variable)


