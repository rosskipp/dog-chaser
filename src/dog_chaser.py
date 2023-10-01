#!/usr/bin/python3

import rospy, time, math, statistics
import cv2
import pyttsx3
import numpy as np
import depthai as dai

from i2cpwm_board.msg import Servo, ServoArray
from sensor_msgs.msg import Joy, Range
from depthai_ros_msgs.msg import SpatialDetectionArray
from vision_msgs.msg import BoundingBox2D
from geometry_msgs.msg import Point
from sensor_msgs.msg import Range, Image
from std_msgs.msg import Float32, Bool

from dog_chase_debugger import Debugger

# Setup the voice system
# engine = pyttsx3.init(driverName='espeak')
# engine.setProperty('rate', 120)
# voices = engine.getProperty('voices')


class ServoConvert:
    """
    Class for controlling the servos = convert an input to a servo value
    """

    def __init__(
        self,
        id=1,
        center_value_throttle=333,
        center_value_steer=300,
        range_throttle=100,
        range_steer=150,
    ):
        self.id = id
        self._center_throttle = center_value_throttle
        self._center_steer = center_value_steer
        self._range = range
        self._half_range_throttle = 0.5 * range_throttle
        self._half_range_steer = 0.5 * range_steer

    def getServoValue(self, value_in, type):
        # value is in [-1, 1]
        if type == "steer":
            self.value_out = int(value_in * self._half_range_steer + self._center_steer)
        else:
            self.value_out = int(
                value_in * self._half_range_throttle + self._center_throttle
            )
        return self.value_out


class DogChaser:
    """
    Class for the dog chaser robot
    """

    def __init__(self):
        rospy.loginfo("Setting up Dog Chaser Node...")
        rospy.init_node("dog_chaser")

        # Setup some global variables
        self.SEND_DEBUG = True
        self.DEBUG_IMAGES = True
        self.SAVE_IMAGES = False
        self.VOICE = False
        self.startTime = time.monotonic()

        # Control Variables
        # Steer is positive left
        self.minThrottle = 0.35  # Nothing seems to happen below this value
        self.maxThrottle = 0.43  # [0.0, 1.0]
        self.noSteerDistance = 5.0  # meters
        self.fullSpeedDistance = 3.0  # meters
        self.deadBandSteer = 0.1  # meters
        self.nThrottleAvg = (
            6  # Average the previous n throttle commands in autonomous mode
        )
        self.nSteerAvg = 6  # Average the previous n steer commands in autonomous mode
        self.nSonarAvg = 5  # average previous n sonar values
        self.sonarAvoid = 1.5  # when do we take action on the sonar data and slow down?
        self.sonarReverse = (
            -0.5
        )  # max reverse speed for when we are at 0 on one of the sonar sensors

        # Image Detection labels for YoloV4
        self.labelMap = [
            "person",
            "bicycle",
            "car",
            "motorbike",
            "aeroplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "sofa",
            "pottedplant",
            "bed",
            "diningtable",
            "toilet",
            "tvmonitor",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

        # if self.VOICE:
        #     engine.say("Initializing robot")
        #     engine.runAndWait()

        self.debugger = Debugger(self.labelMap, self.startTime, self.SAVE_IMAGES)

        self.throttle = 0.0
        self.steer = 0.0
        self.steerValues = []
        self.throttleValues = []

        """
        Create actuator dictionary
        {
            throttle: ServoConvert(id=1)
            steer: ServoConvert(id=2)
        }
        """
        self.actuators = {}
        self.actuators["throttle"] = ServoConvert(id=1)
        self.actuators["steering"] = ServoConvert(id=2)

        # Joystick controller values. These will be between -1.0 and 1.0
        self.joystick = {
            "steerMessage": 0.0,
            "throttleMessage": 0.0,
        }

        # Switch for going into dog finding mode
        self.autonomous_mode = False

        # Image Detection
        self.allDetections = None
        # Is there a dog in this frame?
        self.foundDog = False
        # 2D bounding box surrounding the object.
        self.dog_bbox = BoundingBox2D()
        # Center of the detected object in meters
        # Z is distance in front of camera (+ away)
        # X is lateral distance (+ right)
        # Y is vertical distance of point (+ up)
        self.dog_position = Point()
        self.dogAngle = 0.0

        # Keep track of depth and image data
        self.cameraColorImage = Image()
        self.cameraDepthImage = Image()

        # Create servo array
        # 2 servos - 1 = Throttle | 2 = Steer
        self.servoMessage = ServoArray()
        for i in range(2):
            self.servoMessage.servos.append(Servo())

        # Sonar Data
        # Raw Readings
        self.leftSonarValue = 0.0
        self.centerSonarValue = 0.0
        self.rightSonarValue = 0.0
        # reading arrays
        self.leftSonarValues = []
        self.centerSonarValues = []
        self.rightSonarValues = []
        # avg values
        self.leftSonarAvg = 0.0
        self.centerSonarAvg = 0.0
        self.rightSonarAvg = 0.0

        # ----------- #
        # ROS Pub/Sub #
        # ----------- #
        # Create the servo array publisher
        self.publishServo = rospy.Publisher(
            "/servos_absolute", ServoArray, queue_size=1
        )
        rospy.loginfo("> Publisher correctly initialized")

        # Create the Subscriber to Joystick commands
        rospy.Subscriber("/joy", Joy, self.setJoystickValues)
        rospy.loginfo("> Joystick subscriber correctly initialized")

        # Create the subscriber to the sonar data
        # rospy.Subscriber("/sonar_array", Range, self.processSonarData)
        rospy.Subscriber("/car/sonar/2", Range, self.processCenterSonarData)
        rospy.Subscriber("/car/sonar/1", Range, self.processRightSonarData)
        rospy.Subscriber("/car/sonar/0", Range, self.processLeftSonarData)

        # Create the subscriber to depthai detections
        rospy.Subscriber(
            "/yolov4_publisher/color/yolov4_Spatial_detections",
            SpatialDetectionArray,
            self.processSpatialDetections,
        )

        # Create the subscriber to depthai depth data
        rospy.Subscriber("/yolov4_publisher/stereo/depth", Image, self.processDepthData)

        # Create subscriber to depthai images
        rospy.Subscriber("/yolov4_publisher/color/image", Image, self.processImageData)

        rospy.loginfo("Initialization complete")

        if self.VOICE:
            pass
            # engine.say("robot ready to rumble")
            # engine.runAndWait()

    def sendDebugValues(self):
        self.debugger.sendDebugValues(
            self.steer,
            self.throttle,
            self.foundDog,
            self.dog_position,
            self.dogAngle,
            self.leftSonarAvg,
            self.centerSonarAvg,
            self.rightSonarAvg,
        )

    def processSpatialDetections(self, message):
        self.foundDog = False
        self.allDetections = message.detections

        if len(message.detections) != 0:
            labels_found = []
            for detection in message.detections:
                # if len(detection.results) > 1:
                # rospy.loginfo(detection)
                for result in detection.results:
                    id = result.id
                    label = self.labelMap[id]
                    labels_found.append(self.labelMap[id])
                    if label == "person":  # "dog"
                        self.foundDog = True
                        self.dog_bbox = detection.bbox
                        self.dog_position = detection.position
            # rospy.loginfo('labels found: ' + str(labels_found))

        # else:
        # rospy.loginfo('no detections found')

    def processImageData(self, image):
        self.cameraColorImage = image

    def processDepthData(self, image):
        self.cameraDepthImage = image

    def processLeftSonarData(self, message):
        self.leftSonarValue = message.range
        self.leftSonarValues.append(message.range)
        self.leftSonarValues = self.leftSonarValues[-self.nSonarAvg :]
        self.leftSonarAvg = statistics.mean(self.leftSonarValues)

    def processCenterSonarData(self, message):
        self.centerSonarValue = message.range
        self.centerSonarValues.append(message.range)
        self.centerSonarValues = self.centerSonarValues[-self.nSonarAvg :]
        self.centerSonarAvg = statistics.mean(self.centerSonarValues)

    def processRightSonarData(self, message):
        self.rightSonarValue = message.range
        self.rightSonarValues.append(message.range)
        self.rightSonarValues = self.rightSonarValues[-self.nSonarAvg :]
        self.rightSonarAvg = statistics.mean(self.rightSonarValues)

    def setJoystickValues(self, message):
        """
        Get a Joystick message from joy, set actuators based on message.
        Using Xbox controller - left stick for steer, right stick for throttle

        Joy looks like:
        Reports the state of a joysticks axes and buttons.
        Header header           # timestamp in the header is the time the data is received from the joystick
        float32[] axes          # the axes measurements from a joystick
        int32[] buttons         # the buttons measurements from a joystick

        axes: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
              left stick         right stick
        """

        # Get the data from the message
        axes = message.axes
        buttons = message.buttons
        a_button = buttons[0]
        self.joystick["steerMessage"] = axes[0]
        self.joystick["throttleMessage"] = axes[4]
        # print(self.joystick)
        if a_button == 1:
            self.autonomous_mode = not self.autonomous_mode
            # if self.autonomous_mode:
            #     if self.VOICE:
            #         engine.say("Autonomous mode activated. Time to find some puppies.")
            #         engine.runAndWait()
            # else:
            #     pass
            # engine.say('Manual mode activated')
            rospy.loginfo(
                "Swapping autonomous modes, now: {}".format(self.autonomous_mode)
            )

    def calculateInputs(self):
        """
        Calculate the steer and throttle commands to be sent to the robot.
        """
        throttleMessage = 0.0
        steerMessage = 0.0

        # First figure out if we're going to hit something - sonar data, if we are send a brake/steer command accordingly
        minSonarDistance = min(
            [self.leftSonarAvg, self.centerSonarAvg, self.rightSonarAvg]
        )
        if minSonarDistance > self.sonarAvoid:
            throttleMessage = (
                (self.maxThrottle - self.sonarReverse) / self.sonarAvoid
            ) * minSonarDistance - self.sonarReverse
            # figure out if there's something to the left or right
            if self.leftSonarAvg > self.rightSonarAvg:
                # Steer to the left
                steerMessage = 1.0
            else:
                # steer to the right
                steerMessage = -1.0

            # set the throttle & steer messages & return
            self.setThrottleSteer(throttleMessage, steerMessage)
            return

        # Next check if autonomous mode is disabled, if it is then set throttle and steer based of joystick commands
        if not self.autonomous_mode:
            self.steer = self.joystick["steerMessage"]
            self.throttle = self.joystick["throttleMessage"]

        # if we're not in autonomous mode then do autonomous things
        if self.autonomous_mode:
            # If we found a dog, then drive towards it!
            if self.foundDog:
                # rospy.loginfo('we have a dog')
                # rospy.loginfo('dog position: {}'.format(self.dog_position))
                z = self.dog_position.z
                x = self.dog_position.x
                # Set throttle based on Z position of dog
                if z > self.fullSpeedDistance:
                    throttleMessage = 1.0
                else:
                    throttleMessage = z / self.fullSpeedDistance

                # Set steer based on X & Z position
                # if (abs(z) > noSteerDistance) or (x < deadBandSteer):
                #     steerMessage = 0.0
                # Calculate angle of dog to camera
                if z != 0:
                    theta = math.degrees(math.atan(x / z))
                    self.dogAngle = theta
                    if theta > 45.0:
                        steerMessage = -1.0
                    elif theta < -45.0:
                        steerMessage = 1.0
                    else:
                        steerMessage = -1.0 * (1 / 45.0) * theta

            else:
                # If we don't have any detections, then drive in a circle to try to find detections
                throttleMessage = 0.1
                steerMessage = -1.0

        # set the throttle & steer messages
        self.setThrottleSteer(throttleMessage, steerMessage)

    def setServoValues(self):
        """
        Set servo values based on data set on DogChaser class
        Send the servo message at the end
        """

        # Scale the throttle based on max speed, but only forward
        # Scale using a min and max value
        rangeOld = 1.0
        rangeNew = self.maxThrottle - self.minThrottle
        if self.throttle > 0.0:
            self.throttle = (rangeNew / rangeOld) * (
                self.throttle - 1.0
            ) + self.maxThrottle

        self.actuators["throttle"].getServoValue(self.throttle, "throttle")
        self.actuators["steering"].getServoValue(self.steer, "steer")

        # rospy.loginfo("Got a command Throttle = {} Steer = {}".format(self.throttle, self.steer))

        self.sendServoMessage()

    def setThrottleSteer(self, throttle, steer):
        # Compute and set the throttle
        self.throttleValues.append(throttle)
        self.throttleValues = self.throttleValues[-self.nThrottleAvg :]
        self.throttle = statistics.mean(self.throttleValues)
        # print('current throttle array: {}'.format(self.throttleValues))
        # print('throttle command: {}'.format(self.throttle))

        # Compute and set the steer
        self.steerValues.append(steer)
        self.steerValues = self.steerValues[-self.nSteerAvg :]
        self.steer = statistics.mean(self.steerValues)
        # print('current steer array: {}'.format(self.steerValues))
        # print('steer command: {}'.format(self.steer))

    def sendServoMessage(self):
        for actuator_name, servo_obj in iter(self.actuators.items()):
            self.servoMessage.servos[servo_obj.id - 1].servo = servo_obj.id
            self.servoMessage.servos[servo_obj.id - 1].value = servo_obj.value_out
            # rospy.loginfo("Sending to {} command {}".format(actuator_name, servo_obj.value_out))

        self.publishServo.publish(self.servoMessage)

    def run(self):
        # Set the control rate
        # Run the loop @ 10hz
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            # Sleep until next cycle
            self.calculateInputs()
            self.setServoValues()
            if self.SEND_DEBUG:
                self.sendDebugValues()
            if self.DEBUG_IMAGES:
                self.debugger.sendDebugImage(self.cameraColorImage, self.allDetections)
            rate.sleep()


if __name__ == "__main__":
    chaser = DogChaser()
    chaser.run()
