#!/usr/bin/python3

import rospy, time, math, statistics
import cv2
import pyttsx3
import numpy as np
import depthai as dai

from i2cpwm_board.msg import Servo, ServoArray
from sensor_msgs.msg import Joy, Range
from vision_msgs.msg import BoundingBox2D
from geometry_msgs.msg import Point
from sensor_msgs.msg import Range, Image
from std_msgs.msg import Float32, Bool
from dog_chaser.msg import Collision, SpatialDetectionArray, SpatialDetection

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
        center_value=312,
        range=35,
    ):
        self.id = id
        self.center = center_value
        self.range = range

    def getServoValue(self, value_in):
        # value is in [-1, 1]
        # Value out needs to be a PWM value
        self.value_out = int(value_in * self.range + self.center)
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
        self.DEBUG_IMAGES = False
        self.SAVE_IMAGES = False
        self.VOICE = False
        self.CHECK_COLLISION = False  # turn this off for desk testing
        self.startTime = time.monotonic()
        self.firstStart = True

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
        # self.nSonarAvg = 5  # average previous n sonar values
        self.sonarAvoid = (
            1000  # when do we take action on the sonar data and slow down?
        )
        # self.sonarReverse = (
        #     -0.5
        # )  # max reverse speed for when we are at 0 on one of the sonar sensors

        # Image Detection labels for YoloV4
        self.labelMap = [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",  # 12
            "horse",
            "motorbike",
            "person",  # 15
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
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
        self.actuators["left"] = ServoConvert(id=1)
        self.actuators["right"] = ServoConvert(id=2)

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
        # tracking status of our detection
        self.tracking_status = None
        self.is_tracking = False
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

        # Collision Data
        self.leftCollisionDistance = 10000.0
        self.leftCollisionDetected = False
        self.centerCollisionDistance = 10000.0
        self.centerCollisionDetected = False
        self.rightCollisionDistance = 10000.0
        self.rightCollisionDetected = False

        # ----------- #
        # ROS Pub/Sub #
        # ----------- #
        # Create the servo array publisher
        self.publishServo = rospy.Publisher(
            "/servos_absolute", ServoArray, queue_size=1
        )

        # Create the Subscriber to Joystick commands
        rospy.Subscriber("/joy", Joy, self.setJoystickValues)

        # Create the subscriber to the sonar data
        rospy.Subscriber(
            "/collision_detection/left_distance",
            Collision,
            self.processLeftCollisionData,
        )
        rospy.Subscriber(
            "/collision_detection/center_distance",
            Collision,
            self.processCenterCollisionData,
        )
        rospy.Subscriber(
            "/collision_detection/right_distance",
            Collision,
            self.processRightCollisionData,
        )

        # Create the subscriber to depthai detections
        rospy.Subscriber(
            "/object_tracker/detections",
            SpatialDetectionArray,
            self.processSpatialDetections,
        )

        # # Create the subscriber to depthai depth data
        # rospy.Subscriber("/yolov4_publisher/stereo/depth", Image, self.processDepthData)

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
            self.leftCollisionDistance,
            self.centerCollisionDistance,
            self.rightCollisionDistance,
            self.tracking_status,
            self.is_tracking,
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
                    if label == "person" and detection.is_tracking == True:  # "dog"
                        self.foundDog = True
                        self.dog_bbox = detection.bbox
                        self.dog_position = detection.position
                        self.tracking_status = detection.tracking_status
                        self.is_tracking = detection.is_tracking
            # rospy.loginfo('labels found: ' + str(labels_found))

        # else:
        # rospy.loginfo('no detections found')

    def processImageData(self, image):
        self.cameraColorImage = image

    def processDepthData(self, image):
        self.cameraDepthImage = image

    def processLeftCollisionData(self, message):
        self.leftCollisionDistance = message.distance
        self.leftCollisionDetected = message.detected

    def processCenterCollisionData(self, message):
        self.centerCollisionDistance = message.distance
        self.centerCollisionDetected = message.detected

    def processRightCollisionData(self, message):
        self.rightCollisionDistance = message.distance
        self.rightCollisionDetected = message.detected

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
        # print("autonomous mode: ", self.autonomous_mode)

        # First figure out if we're going to hit something - if we are send a brake/steer command accordingly
        if self.CHECK_COLLISION and (
            self.leftCollisionDetected
            or self.centerCollisionDetected
            or self.rightCollisionDetected
        ):
            throttleMessage = 0
            # figure out if there's something to the left or right
            if self.rightCollisionDetected or self.centerCollisionDetected:
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
            steerMessage = self.joystick["steerMessage"]
            throttleMessage = self.joystick["throttleMessage"]

        # if we're not in autonomous mode then do autonomous things
        if self.autonomous_mode:
            # If we found a dog, then drive towards it!
            if self.foundDog:
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
                throttleMessage = 0.0
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
        # rangeOld = 1.0
        # rangeNew = self.maxThrottle - self.minThrottle
        # if self.throttle > 0.0:
        #     self.throttle = (rangeNew / rangeOld) * (
        #         self.throttle - 1.0
        #     ) + self.maxThrottle

        ### Mixer for tracked vehicle
        leftValue = self.throttle - self.steer
        rightValue = self.throttle + self.steer

        # i wired the motors backwards i think...
        self.actuators["left"].getServoValue(-leftValue)
        self.actuators["right"].getServoValue(-rightValue)

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

    def initializeServos(self):
        # for actuator_name, servo_obj in iter(self.actuators.items()):
        #     self.servoMessage.servos[servo_obj.id - 1].servo = servo_obj.id
        #     self.servoMessage.servos[servo_obj.id - 1].value = servo_obj.value_out
        # self.publishServo.publish(self.servoMessage)
        pass

    def run(self):
        if self.firstStart:
            print("initializing servos")
            # self.firstStart = False
            # time.sleep(10)
            # self.initializeServos()
            # print("servos initialized")

        # Set the control rate
        # Run the loop @ 10hz
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            # Sleep until next cycle
            self.calculateInputs()

            # print(
            #     "steer joystick: {} throttle joystick: {}".format(
            #         self.joystick["steerMessage"], self.joystick["throttleMessage"]
            #     )
            # )
            # print("throttle: {} steer: {}".format(self.throttle, self.steer))

            self.setServoValues()
            if self.SEND_DEBUG:
                self.sendDebugValues()
            # if self.DEBUG_IMAGES:
            #     self.debugger.sendDebugImage(self.cameraColorImage, self.allDetections)
            rate.sleep()


if __name__ == "__main__":
    chaser = DogChaser()
    chaser.run()
