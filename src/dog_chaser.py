#!/usr/bin/python3

import rospy, time, math
from i2cpwm_board.msg import Servo, ServoArray
from sensor_msgs.msg import Joy
from depthai_ros_msgs.msg import SpatialDetectionArray
from vision_msgs.msg import BoundingBox2D
from geometry_msgs.msg import Point
from sensor_msgs.msg import Range
import pyttsx3

# Setup the voice system
engine = pyttsx3.init(driverName='espeak')
engine.setProperty('rate', 125)
voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)


# noSteerDistance         = 5.    # meters
fullSpeedDistance       = 3.    # meters
deadBandSteer           = 0.1   # meters

class ServoConvert():
    """
    Class for controlling the servos = convert an input to a servo value
    """
    def __init__(self, id=1, center_value_throttle=333, center_value_steer=300, range=100):
        self.id = id
        self._center_throttle = center_value_throttle
        self._center_steer = center_value_steer
        self._range = range
        self._half_range = 0.5 * range

    def getServoValue(self, value_in, type):
        # value is in [-1, 1]
        if type == "steer":
            self.value_out = int(value_in * self._half_range + self._center_steer)
        else:
            self.value_out = int(value_in * self._half_range + self._center_throttle)
        return(self.value_out)

class DogChaser():
    """
    Class for the dog chaser robot
    """
    def __init__(self):

        rospy.loginfo("Setting up Dog Chaser Node...")
        rospy.init_node('dog_chaser')

        engine.say("Initializing robot")
        engine.runAndWait()

        """
        Create actuator dictionary
        {
            throttle: ServoConvert(id=1)
            steer: ServoConvert(id=2)
        }
        """
        self.actuators = {}
        self.actuators['throttle'] = ServoConvert(id=1)
        self.actuators['steering'] = ServoConvert(id=2)

        # Joystick controller values. These will be between -1.0 and 1.0
        self.joystick = {
            'steerMessage': 0.0,
            'throttleMessage': 0.0,
        }

        # Switch for going into dog finding mode
        self.autonomous_mode = False

        # Image Detection
        # Is there a dog in this frame?
        self.found_dog = False
        # 2D bounding box surrounding the object.
        self.dog_bbox = BoundingBox2D()
        # Center of the detected object in meters
        # Z is distance in front of camera (+ away)
        # X is lateral distance (+ right)
        # Y is vertical distance of point (+ up)
        self.dog_position = Point()

        # Create servo array
        # 2 servos - 1 = Throttle | 2 = Steer
        self.servoMessage = ServoArray()
        for i in range(2): self.servoMessage.servos.append(Servo())

        # ----------- #
        # ROS Pub/Sub #
        # ----------- #
        # Create the servo array publisher
        self.publishServo = rospy.Publisher("/servos_absolute", ServoArray, queue_size=1)
        rospy.loginfo("> Publisher correctly initialized")

        # Create the Subscriber to Joystick commands
        rospy.Subscriber("/joy", Joy, self.setJoystickValues)
        rospy.loginfo("> Joystick subscriber correctly initialized")

        # Create the subscriber to the sonar data
        # rospy.Subscriber("/sonar")

        # Create the subscriber to depthai detections
        rospy.Subscriber("/yolov4_publisher/color/yolov4_Spatial_detections", SpatialDetectionArray, self.processSpatialDetections)

        # Create the subscriber to depthai depth data

        # Create subscriber to depthai images


        rospy.loginfo("Initialization complete")
        engine.say("robot ready to rumble")
        engine.runAndWait()


    def processSpatialDetections(self, message):
        labelMap = [
            "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
            "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
            "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
            "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
            "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
            "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
            "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
            "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
            "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
            "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
            "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
            "teddy bear",     "hair drier", "toothbrush"
        ]
        # rospy.loginfo("spatial detection: " + str(message.detections))

        self.found_dog = False
        if len(message.detections) != 0:
            labels_found = []
            for detection in message.detections:
                # if len(detection.results) > 1:
                # rospy.loginfo(detection)
                for result in detection.results:
                    id = result.id
                    label = labelMap[id]
                    labels_found.append(labelMap[id])
                    if label == "person": #"dog"
                        self.found_dog = True
                        self.dog_bbox = detection.bbox
                        self.dog_position = detection.position
            # rospy.loginfo('labels found: ' + str(labels_found))

        # else:
            # rospy.loginfo('no detections found')



    def processDepthaiImages(self, message):
        pass

    def processDepthaiDepthData(self, message):
        pass

    def processSonarData(self, message):
        pass


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
        self.joystick['steerMessage'] = axes[0]
        self.joystick['throttleMessage'] = axes[4]
        if a_button == 1:
            self.autonomous_mode = not self.autonomous_mode
            if self.autonomous_mode:
                engine.say('Autonomous mode activated. Time to find some puppies.')
            else:
                engine.say('Manual mode activated')
            rospy.loginfo('Swapping autonomous modes, now: {}'.format(self.autonomous_mode))

    def setServoValues(self):
        """
        Set servo values based on data set on DogChaser class
        Send the servo message at the end
        """

        throttleMessage = 0.
        steerMessage = 0.

        # First figure out if we're going to hit something - sonar data, if we are send a brake/steer command accordingly


        # Next check if autonomous mode is disabled, if it is then set throttle and steer based of joystick commands
        if not self.autonomous_mode:
            throttleMessage = self.joystick['throttleMessage']
            steerMessage = self.joystick['steerMessage']

        # if we're not in autonomous mode then do autonomous things
        if self.autonomous_mode:

            # If we found a dog, then drive towards it!
            if self.found_dog:
                rospy.loginfo('we have a god')
                rospy.loginfo('dog position: {}'.format(self.dog_position))
                z = self.dog_position.z
                x = self.dog_position.x
                # Set throttle based on Z position of dog
                if z > fullSpeedDistance:
                    throttleMessage = 1.0
                else:
                    throttleMessage = z / fullSpeedDistance

                # Set steer based on X & Z position
                # if (abs(z) > noSteerDistance) or (x < deadBandSteer):
                #     steerMessage = 0.0
                # Calculate angle of dog to camera
                theta = math.degrees(math.atan(x/z))
                if theta > 45.0:
                    steerMessage = 1.0
                elif theta < -45.0:
                    steerMessage = 0.1
                else:
                    steerMessage = (1/45.) * theta


            else:
                # If we don't have any detections, then drive in a circle to try to find detections
                throttleMessage = 0.1
                steerMessage = 1.0


        self.actuators['throttle'].getServoValue(throttleMessage, 'throttle')
        self.actuators['steering'].getServoValue(steerMessage, 'steer')

        rospy.loginfo("Got a command Throttle = {} Steer = {}".format(throttleMessage, steerMessage))

        self.sendServoMessage()


    def sendServoMessage(self):

        for actuator_name, servo_obj in iter(self.actuators.items()):
            self.servoMessage.servos[servo_obj.id-1].servo = servo_obj.id
            self.servoMessage.servos[servo_obj.id-1].value = servo_obj.value_out
            rospy.loginfo("Sending to {} command {}".format(actuator_name, servo_obj.value_out))

        self.publishServo.publish(self.servoMessage)

    def run(self):

        # Set the control rate
        # Run the loop @ 10hz
        rate = rospy.Rate(10)
        engine.startLoop()

        while not rospy.is_shutdown():
            # Sleep until next cycle
            self.setServoValues()
            rate.sleep()

if __name__ == "__main__":
    chaser = DogChaser()
    chaser.run()