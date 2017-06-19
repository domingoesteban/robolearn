import numpy as np
import rospy
import math

from XCM.msg import CommandAdvr


publisher = rospy.Publisher("/xbotcore/bigman/command", CommandAdvr, queue_size=10)
#publisher = rospy.Publisher("/xbotcore/centauro/command", CommandAdvr, queue_size=10)

rospy.init_node('command_example')

des_cmd = CommandAdvr()

#des_cmd.name = ["LShSag"]
#des_cmd.position = [np.deg2rad(-45)]

#des_cmd.name = ['LShSag', 'LShLat', 'LShYaw', 'LElbj', 'LForearmPlate', 'LWrj1', 'LWrj2']
#des_cmd.position = [np.deg2rad(0), np.deg2rad(90), np.deg2rad(0), np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]

des_cmd.name = ['LShSag']
#des_cmd.name = ['j_arm1_1']

pub_rate = rospy.Rate(100)

while not rospy.is_shutdown():
    #des_cmd.position = [math.sin(rospy.Time.now().to_sec())]
    des_cmd.effort = [50*math.sin(rospy.Time.now().to_sec())]
    #des_cmd.effort = [500*np.random.randn()]
    print("Sending cmd...")
    publisher.publish(des_cmd)
    pub_rate.sleep()
