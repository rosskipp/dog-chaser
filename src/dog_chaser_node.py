#!/usr/bin/python3

import rospy, time, math, statistics
import pyttsx3
import numpy as np
from datetime import datetime

from i2cpwm_board.msg import Servo, ServoArray
from sensor_msgs.msg import Joy, Range
from vision_msgs.msg import BoundingBox2D
from geometry_msgs.msg import Point
from sensor_msgs.msg import Range, Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float32, Bool
from dog_chaser.msg import Collision, SpatialDetectionArray, SpatialDetection
from kalman import Kalman3DAcceleration

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
            0.35  # this is to reduce the sensitivity of the steering
        )

        # Collision Variables
        self.collision_throttle_decay = 0.5
        self.collision_steer_multiplier = 1.25

        # RUN AWAY MODE VARIABLES
        # go straight if the dog is this far away (meters)
        self.noSteerDistance = 5.0  # meters
        # only go full speed after this distance from the object
        self.full_speed_distance = 3.0
        # if the dog is within this distance from center, don't steer
        self.deadBandSteer = 0.1  # meters

        # FILTER VARIABLES
        # These will filter the commands to smooth out the control
        self.filter_enabled = True
        # Filter previous n throttle commands in autonomous mode
        self.nThrottleAvg = 5
        # Filter the previous n steer commands in autonomous mode
        self.nSteerAvg = 5

        ### Variables for controlled motion (Dog Searching)
        # how long to run the command (seconds)
        self.COMMAND_LENGTH = 3
        self.COMMAND_PAUSE = 10
        self.is_controlled_command = False
        self.controlled_command_start_time = datetime.now()
        self.controlled_command_throttle = 0.0
        self.controlled_command_steer = -1.0

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
        self.tracking_status = False
        # Center of the detected object in meters
        # Z is distance in front of camera (+ away)
        # X is lateral distance (+ right)
        # Y is vertical distance of point (+ up)
        self.dog_raw_position = Point()
        self.dog_position = Point()
        self.dog_angle = 0.0

        # Kalman filter
        self.kalman = Kalman3DAcceleration()
        self.current_kalman_prediction = Point()


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
        self.left_collision_distance = 10000.0
        self.left_collision = False
        self.center_collision_distance = 10000.0
        self.center_collision = False
        self.right_collision_distance = 10000.0
        self.right_collision = False

        #####################
        ### ROS Pub/Sub #
        #####################
        # Create the servo array publisher
        self.publishServo = rospy.Publisher(
            "/servos_absolute", ServoArray, queue_size=1
        )

        # Create the Subscriber to Joystick commands
        rospy.Subscriber("/joy", Joy, self.setJoystickValues)

        # Create the subscriber to depthai detections
        rospy.Subscriber(
            "/object_tracker/detections",
            SpatialDetectionArray,
            self.process_spatial_detections,
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

        # for point in nearby_points:
        #     print(point)

    def sendDebugValues(self):
        self.debugger.sendDebugValues(
            self.steer,
            self.throttle,
            self.found_dog,
            self.dog_position,
            self.dog_angle,
            self.left_collision_distance,
            self.center_collision_distance,
            self.right_collision_distance,
            self.tracking_status,
            self.dog_raw_position,
            self.dog_position,
        )

    def get_kalman_prediction(self):
        return self.kalman.predict()

    def update_dog_position(self):
        """Convert Kalman filter position estimate to ROS Point"""
        position = self.kalman.get_position()
        self.dog_position = Point(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2])
        )

    def process_spatial_detections(self, message):
        # First predict the next state
        self.current_kalman_prediction = self.kalman.predict()

        # Update position with prediction
        self.update_dog_position()

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
                    ):  # and detection.tracking_status == True:
                        found_dog_frame = True
                        self.dog_raw_position = detection.position
                        self.dog_bbox = detection.bbox

                        # Convert detection.position to numpy array if it isn't already
                        measurement = np.array([
                            detection.position.x,
                            detection.position.y,
                            detection.position.z
                        ], dtype=np.float32)

                        # Correct with the measurement
                        self.kalman.correct(measurement)
                        # Update position after correction
                        self.update_dog_position()
                        break  # Found the dog, no need to check other detections

        if not found_dog_frame:
            # No detection found, just correct with None
            self.kalman.correct(None)
            # Position was already updated with prediction

        self.found_dog = self.kalman.get_tracking()
        self.tracking_status = self.kalman.get_tracking()

    def processImageData(self, image):
        self.cameraColorImage = image

    def processDepthData(self, image):
        self.cameraDepthImage = image


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
        # a_button = buttons[0]
        b_button = buttons[1]
        self.joystick["steer_message"] = axes[0]
        self.joystick["throttle_message"] = axes[4]
        if b_button == 1:
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
        throttle_message = 0.0
        steer_message = 0.0
        # print("autonomous mode: ", self.autonomous_mode)
        current_throttle = self.throttle
        current_steer = self.steer

        # First figure out if we're going to hit something - if we are send a brake/steer command accordingly
        # TODO: NEED TO UPDATE THIS TO USE THE LIDAR DATA
        if self.CHECK_COLLISION and (
            self.left_collision
            or self.center_collision
            or self.right_collision
        ):
            # decay the throttle command
            throttle_message = current_throttle * self.collision_throttle_decay

            # increase the steer command based on our multiplier
            steer_message = abs(current_steer) * self.collision_steer_multiplier
            if steer_message < 1:
                steer_message = 1
            # figure out if there's something to the left or right
            if self.right_collision or self.center_collision:
                # Steer to the left
                steer_message = steer_message
            else:
                # steer to the right
                steer_message = -1 * steer_message

            # set the throttle & steer messages & return
            self.set_throttle_steer(throttle_message, steer_message)
            return


        # If we're not going to hit something, then check if we're in autonomous mode
        # Autonomous Mode (Dog Finding)
        if self.autonomous_mode:
            # If we found a dog, then drive away from it
            # we want to try to keep the dog in the center of the frame, while also not hitting anything
            if self.found_dog:
                # TODO: NEED TO UPDATE THIS ALGORITHM TO RUN FROM DOG
                # reset the controlled command counters, in case we just transitioned from finding the dog
                self.is_controlled_command = False

                # Get the Z and X position of the dog
                z = self.dog_position.z
                x = self.dog_position.x
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
                    self.dog_angle = theta
                    if theta > 45.0:
                        steer_message = -1.0
                    elif theta < -45.0:
                        steer_message = 1.0
                    else:
                        steer_message = -1.0 * (1 / 45.0) * theta

            # If we don't have a dog detection, then drive in a (slow) circle to try to find detections
            else:
                if self.is_controlled_command:
                    timer_delta = datetime.now() - self.controlled_command_start_time
                    # If we're in a controlled command, then we need to check if it's time to stop
                    if (
                        timer_delta.total_seconds()
                        > self.COMMAND_LENGTH + self.COMMAND_PAUSE
                    ):
                        # If we've been in the controlled command for the sum of the length and pause, then stop and reset the counter
                        self.is_controlled_command = False
                    elif timer_delta.total_seconds() < self.COMMAND_LENGTH:
                        # If we're in the first part of the controlled command, then drive at the controlled command throttle and steer
                        throttle_message = self.controlled_command_throttle
                        steer_message = self.controlled_command_steer
                    else:
                        # If we're in the second part of the controlled command, then drive at the controlled command throttle and steer
                        throttle_message = 0
                        steer_message = 0
                else:
                    self.is_controlled_command = True
                    self.controlled_command_start_time = datetime.now()

        else:
            # if neither autonomous mode is on, then use the joystick
            steer_message = self.joystick["steer_message"]
            throttle_message = self.joystick["throttle_message"]

        # set the throttle & steer messages
        self.set_throttle_steer(throttle_message, steer_message)

    def setServoValues(self):
        """
        Set servo values based on data set on DogChaser class
        Send the servo message at the end
        """
        ### Mixer for tracked vehicle
        left_value = self.throttle - (self.steer_multiplier * self.steer)
        right_value = self.throttle + (self.steer_multiplier * self.steer)

        # i wired the motors backwards i think...
        self.actuators["left"].get_servo_values(-left_value)
        self.actuators["right"].get_servo_values(-right_value)

        # rospy.loginfo("Got a command Throttle = {} Steer = {}".format(self.throttle, self.steer))

        self.sendServoMessage()

    def set_throttle_steer(self, throttle, steer):
        # Compute and set the throttle
        # append this value to our list of throttle values
        self.throttle_values.append(throttle)
        # only keep the last n throttle values
        if self.filter_enabled:
            self.throttle_values = self.throttle_values[-self.nThrottleAvg :]
            # filter the throttle values
            self.throttle = statistics.mean(self.throttle_values)
        else:
            self.throttle = throttle

        # Compute and set the steer
        self.steer_values.append(steer)
        if self.filter_enabled:
            self.steer_values = self.steer_values[-self.nSteerAvg :]
            self.steer = statistics.mean(self.steer_values)
        else:
            self.steer = steer

    def sendServoMessage(self):
        for _, servo_obj in iter(self.actuators.items()):
            self.servoMessage.servos[servo_obj.id - 1].servo = servo_obj.id
            self.servoMessage.servos[servo_obj.id - 1].value = servo_obj.value_out

        self.publishServo.publish(self.servoMessage)

    def run(self):
        if self.first_start:
            print("initializing servos")
            # self.first_start = False
            # time.sleep(10)

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
