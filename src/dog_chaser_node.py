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
from sensor_msgs.msg import Range, Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float32, Bool
from dog_chaser.msg import Collision, SpatialDetectionArray, SpatialDetection
import scipy.signal as signal
from kalman import DogKalmanFilter

from dog_chase_debugger import Debugger

# Setup the voice system
# engine = pyttsx3.init(driverName='espeak')
# engine.setProperty('rate', 120)
# voices = engine.getProperty('voices')


class LidarPoint:
    """
    Class for a point in the lidar data
    x, y are the coordinates in mm
    theta is the angle in degrees
    r is the distance in mm
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.theta = (180 / math.pi) * math.atan2(y, x)
        self.r = math.sqrt(x**2 + y**2)

    def __str__(self):
        return f"x: {self.x}, y: {self.y}, theta: {self.theta}, r: {self.r}"


class ServoConvert:
    """
    Class for controlling the servos = convert an input to a servo value
    """

    def __init__(self, id=1, center_value=312, range=35, range_steer=20):
        self.id = id
        self.center = center_value
        self.range = range

    def get_servo_values(self, value_in):
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
        self.start_time = time.monotonic()
        self.first_start = True

        # Control Variables
        # Steer is positive left
        # self.min_throttle = 0.35  # Nothing seems to happen below this value
        # self.max_throttle = 0.43  # [0.0, 1.0]
        self.steer_multiplier = (
            0.25  # this is to reduce the sensitivity of the steering
        )

        # Collision Variables
        self.collision_throttle_decay = 0.5
        self.collision_steer_multiplier = 1.25

        # go straight if the dog is this far away (meters)
        self.noSteerDistance = 5.0  # meters
        # only go full speed after this distance from the object
        self.full_speed_distance = 3.0
        # if the dog is within this distance from center, don't steer
        self.deadBandSteer = 0.1  # meters
        # Filter previous n throttle commands in autonomous mode
        self.nThrottleAvg = 5
        # Filter the previous n steer commands in autonomous mode
        self.nSteerAvg = 5
        # when do we take action on the depth data? (mm) - this is 1 meter
        self.sonarAvoid = 1000

        ### Variables for controlled motion (Dog Searching)
        # how long to run the command (cycles)
        self.COMMAND_LENGTH = 6
        self.COMMAND_PAUSE = 15
        self.is_controlled_command = False
        self.controlled_command_count = 0
        self.controlled_command_throttle = 0.0
        self.controlled_command_steer = -1.0

        ### Variables for controlled motion (Chase Mode) - Throttle
        # how long to run the command (cycles)
        self.CHASE_COMMAND_THROTTLE_PHASE = 15
        self.CHASE_COMMAND_TURN_PHASE = 6
        self.chase_command_count = 0
        self.chase_command_throttle = 1.0
        self.chaseCommandSteer = -1.0

        # Image Detection labels for mobile net
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

        self.debugger = Debugger(self.labelMap, self.start_time, self.SAVE_IMAGES)

        #####################
        ### Control Variables
        #####################
        self.throttle = 0.0
        self.steer = 0.0
        self.steer_values = []
        self.throttle_values = []

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
            "steer_message": 0.0,
            "throttle_message": 0.0,
        }

        # Switch for going into dog finding mode
        self.autonomous_mode = False
        self.autonomous_chase_mode = False

        #####################
        ### Image Detection
        #####################
        self.detection_string = "person"  # "dog"
        self.all_detections = None
        # Is there a dog in this frame?
        self.found_dog = False

        #####################
        ### Spatial Detection
        #####################
        # 2D bounding box surrounding the object.
        self.dog_bbox = BoundingBox2D()
        # tracking status of our detection
        self.is_tracking = False
        # Center of the detected object in meters
        # Z is distance in front of camera (+ away)
        # X is lateral distance (+ right)
        # Y is vertical distance of point (+ up)
        self.dog_raw_position = Point()
        self.dog_position = Point()
        self.dogAngle = 0.0

        # Kalman filter
        self.kalman = DogKalmanFilter()

        #####################
        ### Depth and Image Data
        #####################
        self.cameraColorImage = Image()
        self.cameraDepthImage = Image()

        #####################
        ### Servo Data
        #####################
        # Create servo array
        # 2 servos - 1 = Throttle | 2 = Steer
        self.servoMessage = ServoArray()
        for i in range(2):
            self.servoMessage.servos.append(Servo())

        #####################
        ### Collision Data
        #####################
        self.leftCollisionDistance = 10000.0
        self.leftCollisionDetected = False
        self.centerCollisionDistance = 10000.0
        self.centerCollisionDetected = False
        self.rightCollisionDistance = 10000.0
        self.rightCollisionDetected = False

        #####################
        ### ROS Pub/Sub #
        #####################
        # Create the servo array publisher
        self.publishServo = rospy.Publisher(
            "/servos_absolute", ServoArray, queue_size=1
        )

        # Create the Subscriber to Joystick commands
        rospy.Subscriber("/joy", Joy, self.setJoystickValues)

        # # Create the subscriber to the sonar data
        # rospy.Subscriber(
        #     "/collision_detection/left_distance",
        #     Collision,
        #     self.processLeftCollisionData,
        # )
        # rospy.Subscriber(
        #     "/collision_detection/center_distance",
        #     Collision,
        #     self.processCenterCollisionData,
        # )
        # rospy.Subscriber(
        #     "/collision_detection/right_distance",
        #     Collision,
        #     self.processRightCollisionData,
        # )

        # Create the subscriber to depthai detections
        rospy.Subscriber(
            "/object_tracker/detections",
            SpatialDetectionArray,
            self.processSpatialDetections,
        )

        rospy.Subscriber(
            "/converted_pc", PointCloud2, self.process_pointcloud_data, queue_size=1
        )

        # # Create the subscriber to depthai depth data
        # rospy.Subscriber("/yolov4_publisher/stereo/depth", Image, self.processDepthData)

        if self.VOICE:
            pass
            # engine.say("robot ready to rumble")
            # engine.runAndWait()

        rospy.loginfo("Initialization complete")

    def process_pointcloud_data(self, message: PointCloud2):

        # convert the message to a generator with the individual points
        point_generator = pc2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        )

        # filter out points that are behind the camera
        points_of_interest = [
            LidarPoint(x=p[0], y=p[1]) for p in point_generator if p[0] > 0
        ]

        # Convert points to numpy arrays for faster calculation
        points_array = np.array([[p.x, p.y] for p in points_of_interest])

        # Calculate distances using numpy (much faster than individual calculations)
        # sqrt(x^2 + y^2) for each point
        distances = np.sqrt(np.sum(points_array**2, axis=1))

        # Find points within threshold distance (e.g. 2 meters)
        DISTANCE_THRESHOLD = 2000 # mm
        nearby_point_indices = np.where(distances <= DISTANCE_THRESHOLD)[0]

        # Get the filtered points
        nearby_points = [points_of_interest[i] for i in nearby_point_indices]

        for point in nearby_points:
            print(point)

    def sendDebugValues(self):
        self.debugger.sendDebugValues(
            self.steer,
            self.throttle,
            self.found_dog,
            self.dog_position,
            self.dogAngle,
            self.leftCollisionDistance,
            self.centerCollisionDistance,
            self.rightCollisionDistance,
            self.tracking_status,
            self.is_tracking,
            self.dog_raw_position,
            self.dog_position,
        )

    def get_kalman_prediction(self):
        return self.kalman.predict()

    def update_dog_position(self):
        position = self.kalman.get_position()
        self.dog_position = Point(x=position[0], y=position[1], z=position[2])

    def processSpatialDetections(self, message):
        found_dog_frame = False
        self.all_detections = message.detections

        if len(message.detections) != 0:
            labels_found = []
            for detection in message.detections:
                for result in detection.results:
                    id = result.id
                    label = self.labelMap[id]
                    labels_found.append(self.labelMap[id])
                    if (
                        label == self.detection_string
                    ):  # and detection.is_tracking == True:
                        found_dog_frame = True
                        self.dog_raw_position = detection.position
                        self.dog_bbox = detection.bbox
                        self.kalman.correct(detection.position)
                        self.update_dog_position()

        if not found_dog_frame:
            self.kalman.correct(None)

        self.is_tracking = self.kalman.get_tracking()

    def processImageData(self, image):
        self.cameraColorImage = image

    def processDepthData(self, image):
        self.cameraDepthImage = image

    # def processLeftCollisionData(self, message):
    #     self.leftCollisionDistance = message.distance
    #     self.leftCollisionDetected = message.detected

    # def processCenterCollisionData(self, message):
    #     self.centerCollisionDistance = message.distance
    #     self.centerCollisionDetected = message.detected

    # def processRightCollisionData(self, message):
    #     self.rightCollisionDistance = message.distance
    #     self.rightCollisionDetected = message.detected

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
        b_button = buttons[1]
        self.joystick["steer_message"] = axes[0]
        self.joystick["throttle_message"] = axes[4]
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
        if b_button == 1:
            self.autonomous_chase_mode = not self.autonomous_chase_mode
            rospy.loginfo(
                "Swapping autonomous chase modes, now: {}".format(
                    self.autonomous_chase_mode
                )
            )

    def calculateInputs(self):
        """
        Calculate the steer and throttle commands to be sent to the robot.
        """
        throttle_message = 0.0
        steer_message = 0.0
        # print("autonomous mode: ", self.autonomous_mode)
        current_throttle = self.throttle
        current_steer = self.steer

        # First figure out if we're going to hit something - if we are send a brake/steer command accordingly
        if self.CHECK_COLLISION and (
            self.leftCollisionDetected
            or self.centerCollisionDetected
            or self.rightCollisionDetected
        ):
            # decay the throttle command
            throttle_message = current_throttle * self.collision_throttle_decay

            # increase the steer command based on our multiplier
            steer_message = abs(current_steer) * self.collision_steer_multiplier
            if steer_message < 1:
                steer_message = 1
            # figure out if there's something to the left or right
            if self.rightCollisionDetected or self.centerCollisionDetected:
                # Steer to the left
                steer_message = steer_message
            else:
                # steer to the right
                steer_message = -1 * steer_message

            # set the throttle & steer messages & return
            self.setThrottleSteer(throttle_message, steer_message)
            return

        # Autonomous Chase Mode
        if self.autonomous_chase_mode:
            # If the counter is full, then we reset the counter
            self.chase_command_count += 1
            if self.chase_command_count >= (
                self.CHASE_COMMAND_THROTTLE_PHASE + self.CHASE_COMMAND_TURN_PHASE
            ):
                self.chase_command_count = 0
            elif self.chase_command_count < self.CHASE_COMMAND_THROTTLE_PHASE:
                throttle_message = self.chase_command_throttle
                steer_message = 0.0
            elif self.chase_command_count >= self.CHASE_COMMAND_THROTTLE_PHASE:
                throttle = 0.0
                steer_message = self.chaseCommandSteer

            print("chase command count:", self.chase_command_count)
            print("chase command throttle:", throttle_message)
            print("chase command steer:", steer_message)

        # Autonomous Mode (Dog Finding)
        elif self.autonomous_mode:
            # If we found a dog, then drive towards it!
            if self.found_dog:
                self.is_controlled_command = False
                self.controlled_command_count = 0
                z = self.dog_z_position
                x = self.dog_x_position
                # Set throttle based on Z position of dog
                if z > self.full_speed_distance:
                    throttle_message = 1.0
                else:
                    throttle_message = z / self.full_speed_distance

                # Set steer based on X & Z position
                # if (abs(z) > noSteerDistance) or (x < deadBandSteer):
                #     steer_message = 0.0
                # Calculate angle of dog to camera
                if z != 0:
                    theta = math.degrees(math.atan(x / z))
                    self.dogAngle = theta
                    if theta > 45.0:
                        steer_message = -1.0
                    elif theta < -45.0:
                        steer_message = 1.0
                    else:
                        steer_message = -1.0 * (1 / 45.0) * theta

            else:
                if self.is_controlled_command:
                    if (
                        self.controlled_command_count
                        > self.COMMAND_LENGTH + self.COMMAND_PAUSE
                    ):
                        self.is_controlled_command = False
                        self.controlled_command_count = 0
                    elif self.controlled_command_count < self.COMMAND_LENGTH:
                        throttle_message = self.controlled_command_throttle
                        steer_message = self.controlled_command_steer
                        self.controlled_command_count += 1
                    else:
                        self.controlled_command_count += 1

                else:
                    # If we don't have any detections, then drive in a (slow) circle to try to find detections
                    self.is_controlled_command = True
            print("controlled command:", self.is_controlled_command)
            print("controlled command count:", self.controlled_command_count)

        else:
            # if neither autonomous mode is on, then use the joystick
            steer_message = self.joystick["steer_message"]
            throttle_message = self.joystick["throttle_message"]

        # set the throttle & steer messages
        self.setThrottleSteer(throttle_message, steer_message)

    def setServoValues(self):
        """
        Set servo values based on data set on DogChaser class
        Send the servo message at the end
        """
        ### Mixer for tracked vehicle
        leftValue = self.throttle - (self.steer_multiplier * self.steer)
        rightValue = self.throttle + (self.steer_multiplier * self.steer)

        # i wired the motors backwards i think...
        self.actuators["left"].get_servo_values(-leftValue)
        self.actuators["right"].get_servo_values(-rightValue)

        # rospy.loginfo("Got a command Throttle = {} Steer = {}".format(self.throttle, self.steer))

        self.sendServoMessage()

    def setThrottleSteer(self, throttle, steer):
        # Compute and set the throttle
        # append this value to our list of throttle values
        self.throttle_values.append(throttle)
        # only keep the last n throttle values
        self.throttle_values = self.throttle_values[-self.nThrottleAvg :]
        # filter the throttle values
        self.throttle = statistics.mean(self.throttle_values)
        # print('current throttle array: {}'.format(self.throttleValFilter))
        # print('throttle command: {}'.format(self.throttle))

        # Compute and set the steer
        self.steer_values.append(steer)
        self.steer_values = self.steer_values[-self.nSteerAvg :]
        self.steer = statistics.mean(self.steer_values)
        # print('current steer array: {}'.format(self.steer_values))
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
        if self.first_start:
            print("initializing servos")
            # self.first_start = False
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
            #         self.joystick["steer_message"], self.joystick["throttle_message"]
            #     )
            # )
            # print("throttle: {} steer: {}".format(self.throttle, self.steer))

            self.setServoValues()
            if self.SEND_DEBUG:
                self.sendDebugValues()
            if self.DEBUG_IMAGES:
                self.debugger.sendDebugImage(self.cameraColorImage, self.all_detections)
            rate.sleep()


if __name__ == "__main__":
    chaser = DogChaser()
    chaser.run()
