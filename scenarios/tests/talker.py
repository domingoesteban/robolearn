#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String

def talker(add_to_topic, variable):
    pub = rospy.Publisher('chatter'+str(add_to_topic), String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    counter = 0
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        #rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

        if counter%20 == 0:
            print('hola'+str(variable))

        counter += 1
