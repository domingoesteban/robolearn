import numpy as np
import rospy

from XCM.msg import CommandAdvr


publisher = rospy.Publisher("/xbotcore/bigman/command", CommandAdvr, queue_size=10)

rospy.init_node('command_example')

des_cmd = CommandAdvr()

#des_cmd.name = ["LShSag"]
#des_cmd.position = [np.deg2rad(-45)]

des_cmd.name = ['LShSag', 'LShLat', 'LShYaw', 'LElbj', 'LForearmPlate', 'LWrj1', 'LWrj2']
des_cmd.position = [np.deg2rad(0), np.deg2rad(90), np.deg2rad(0), np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]


pub_rate = rospy.Rate(100)

while not rospy.is_shutdown():
    print("Sending cmd...")
    publisher.publish(des_cmd)
    pub_rate.sleep()
