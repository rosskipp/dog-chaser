#!/usr/bin/python3

import rospy, time, math
import cv2
import pyttsx3
import numpy as np
import depthai as dai

from i2cpwm_board.msg import Servo, ServoArray
from sensor_msgs.msg import Joy
from depthai_ros_msgs.msg import SpatialDetectionArray
from vision_msgs.msg import BoundingBox2D
from geometry_msgs.msg import Point
from sensor_msgs.msg import Range, Image
from std_msgs.msg import Float32, Bool

# Setup the voice system
engine = pyttsx3.init(driverName='espeak')
engine.setProperty('rate', 120)
voices = engine.getProperty('voices')

# Open CV Bridge
from cv_bridge import CvBridge

# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
# def frameNorm(frame, bbox):
#     normVals = np.full(len(bbox), frame.shape[0])
#     normVals[::2] = frame.shape[1]
#     return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

class ServoConvert():
    """
    Class for controlling the servos = convert an input to a servo value
    """
    def __init__(self, id=1, center_value_throttle=333, center_value_steer=300, range_throttle=100, range_steer=150):
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
            self.value_out = int(value_in * self._half_range_throttle + self._center_throttle)
        return(self.value_out)

class DogChaser():
    """
    Class for the dog chaser robot
    """
    def __init__(self):

        rospy.loginfo("Setting up Dog Chaser Node...")
        rospy.init_node('dog_chaser')

        self.bridge = CvBridge()

        # Setup some global variables
        self.SEND_DEBUG = True
        self.DEBUG_IMAGES = True
        self.SAVE_IMAGES = False
        self.VOICE = False
        self.startTime = time.monotonic()
        self.minThrottle            = 0.3 # Nothing seems to happen below this value
        self.maxThrottle            = 0.45 # [0.0, 1.0]
        self.noSteerDistance         = 5.    # meters
        self.fullSpeedDistance       = 3.    # meters
        self.deadBandSteer           = 0.1   # meters

        self.labelMap = [
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


        if self.VOICE:
            engine.say("Initializing robot")
            engine.runAndWait()


        self.throttle = 0.0
        self.steer = 0.0
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
        self.imageCounter = 0
        self.sendImageCounter = 0

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
        rospy.Subscriber('/yolov4_publisher/stereo/depth', Image, self.processDepthData)

        # Create subscriber to depthai images
        rospy.Subscriber('/yolov4_publisher/color/image', Image, self.processImageData)

        # Create debug publishers
        self.publishDebugImage = rospy.Publisher("/dog_chaser/debug_image", Image, queue_size=1)
        self.publishDebugSteer = rospy.Publisher("/dog_chaser/debug_steer", Float32, queue_size=1)
        self.publishDebugThrottle = rospy.Publisher("/dog_chaser/debug_throttle", Float32, queue_size=1)
        self.publishDebugDogBool = rospy.Publisher("/dog_chaser/debug_found_dog", Bool, queue_size=1)
        self.publishDebugDogPosition = rospy.Publisher("/dog_chaser/debug_dog_position", Point, queue_size=1)
        self.publishDebugDogAngle = rospy.Publisher("/dog_chaser/debug_dog_angle", Float32, queue_size=1)

        rospy.loginfo("Initialization complete")

        if self.VOICE:
            engine.say("robot ready to rumble")
            engine.runAndWait()

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
                    if label == "person": #"dog"
                        self.foundDog = True
                        self.dog_bbox = detection.bbox
                        self.dog_position = detection.position
            # rospy.loginfo('labels found: ' + str(labels_found))

        # else:
            # rospy.loginfo('no detections found')

    def processImageData(self, image):
        self.cameraColorImage = image #np.array(image)

    def processDepthData(self, image):
        self.cameraDepthImage = image

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
                if self.VOICE:
                    engine.say('Autonomous mode activated. Time to find some puppies.')
                    engine.runAndWait()
            else:
                pass
                # engine.say('Manual mode activated')
            rospy.loginfo('Swapping autonomous modes, now: {}'.format(self.autonomous_mode))

    def setServoValues(self):
        """
        Set servo values based on data set on DogChaser class
        Send the servo message at the end
        """

        throttleMessage = 0.0
        steerMessage = 0.0

        # First figure out if we're going to hit something - sonar data, if we are send a brake/steer command accordingly
        # SONAR CODE HERE

        # Next check if autonomous mode is disabled, if it is then set throttle and steer based of joystick commands
        if not self.autonomous_mode:
            throttleMessage = self.joystick['throttleMessage']
            steerMessage = self.joystick['steerMessage']

        # if we're not in autonomous mode then do autonomous things
        if self.autonomous_mode:

            # If we found a dog, then drive towards it!
            if self.foundDog:
                rospy.loginfo('we have a dog')
                rospy.loginfo('dog position: {}'.format(self.dog_position))
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
                if z !=0:
                    theta = math.degrees(math.atan(x/z))
                    self.dogAngle = theta
                    if theta > 45.0:
                        steerMessage = -1.0
                    elif theta < -45.0:
                        steerMessage = 1.0
                    else:
                        steerMessage = -1.0 *(1/45.) * theta


            else:
                # If we don't have any detections, then drive in a circle to try to find detections
                throttleMessage = 0.1
                steerMessage = 1.0

        # Scale the throttle based on max speed, but only forward
        # Scale using a min and max value
        rangeOld = 1.0
        rangeNew = self.maxThrottle - self.minThrottle
        if throttleMessage > 0.0:
            throttleMessage = (rangeNew / rangeOld) * (throttleMessage - 1.0) + self.maxThrottle

        self.throttle = throttleMessage
        self.steer = steerMessage
        self.actuators['throttle'].getServoValue(self.throttle, 'throttle')
        self.actuators['steering'].getServoValue(self.steer, 'steer')

        # rospy.loginfo("Got a command Throttle = {} Steer = {}".format(self.throttle, self.steer))

        self.sendServoMessage()

    def setDebugValues(self):

        ###
        self.publishDebugSteer.publish(self.steer)
        self.publishDebugThrottle.publish(self.throttle)
        self.publishDebugDogBool.publish(self.foundDog)
        self.publishDebugDogPosition.publish(self.dog_position)
        self.publishDebugDogAngle.publish(self.dogAngle)

    def sendDebugImage(self):
        counter = self.imageCounter
        counter += 1
        i = self.sendImageCounter
        frame = self.cameraColorImage
        # print(frame)
        if frame.height != 0:
            # convert image to cv2
            frame = self.bridge.imgmsg_to_cv2(frame, "bgr8")
            detections = self.allDetections
            color2 = (255, 255, 255)

            # on every 10th time through, send an image
            if i > 9:
                i = 0
                color = (255, 0, 0)
                for detection in detections:
                    position_x = round(detection.position.x, 3)
                    position_y = round(detection.position.y, 3)
                    position_z = round(detection.position.z, 3)
                    center_x = int(detection.bbox.center.x)
                    center_y = int(detection.bbox.center.y)
                    halfsize_x = int(detection.bbox.size_x / 2)
                    halfsize_y = int(detection.bbox.size_y / 2)
                    # bbox = frameNorm(frame, ((center_x - halfsize_x), (center_y - halfsize_y), (center_x + halfsize_x), (center_y + halfsize_y)))
                    bbox = ((center_x - halfsize_x), (center_y - halfsize_y), (center_x + halfsize_x), (center_y + halfsize_y))
                    # Put label on the image
                    cv2.putText(frame, self.labelMap[detection.results[0].id], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    # Put score on the image
                    cv2.putText(frame, f"{int(detection.results[0].score * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    # Put the depth measurement on the image
                    cv2.putText(frame, "position = x: {}".format(position_x), (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, "y:{}".format(position_y), (bbox[0] + 10, bbox[1] + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, "z:{}".format(position_x), (bbox[0] + 10, bbox[1] + 100), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    # Put the bounding box on the image
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - self.startTime)),
                            (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)
                # Save image if indicated
                if self.SAVE_IMAGES:
                    cv2.imwrite("/home/ubuntu/robot_data/images/{}.jpeg".format(time.time_ns()), frame)
                # convert the image back to ROS image
                frame = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                self.publishDebugImage.publish(frame)

            i += 1
            self.imageCounter = counter
            self.sendImageCounter = i


    def sendServoMessage(self):

        for actuator_name, servo_obj in iter(self.actuators.items()):
            self.servoMessage.servos[servo_obj.id-1].servo = servo_obj.id
            self.servoMessage.servos[servo_obj.id-1].value = servo_obj.value_out
            # rospy.loginfo("Sending to {} command {}".format(actuator_name, servo_obj.value_out))

        self.publishServo.publish(self.servoMessage)

    def run(self):

        # Set the control rate
        # Run the loop @ 10hz
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            # Sleep until next cycle
            self.setServoValues()
            if self.SEND_DEBUG:
                self.setDebugValues()
            if self.DEBUG_IMAGES:
                self.sendDebugImage()
            rate.sleep()

if __name__ == "__main__":
    chaser = DogChaser()
    chaser.run()