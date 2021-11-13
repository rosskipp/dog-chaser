#!/usr/bin/python3

import rospy, time, math
from i2cpwm_board.msg import Servo, ServoArray
from sensor_msgs.msg import Joy
from depthai_ros_msgs.msg import SpatialDetectionArray
from vision_msgs.msg import BoundingBox2D
from geometry_msgs.msg import Point
from sensor_msgs.msg import Range


class Debugger():

    def __init__(self):

        pass