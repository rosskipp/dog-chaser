#!/usr/bin/python3

import rospy, time
from i2cpwm_board.msg import Servo, ServoArray
from sensor_msgs.msg import Joy
# from depthai.msg import

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
        self.joystick = {
            'steer_msg': 0.0,
            'throttleMessage': 0.0,
        }
        self.autonomous_mode = False
        self.found_dog = False
        self.dog_location = [0, 0, 0]

        # Create servo array
        # 2 servos - 1 = Throttle | 2 = Steer
        self._servo_msg = ServoArray()
        for i in range(2): self._servo_msg.servos.append(Servo())

        # Create the servo array publisher
        self.ros_pub_servo_array = rospy.Publisher("/servos_absolute", ServoArray, queue_size=1)
        rospy.loginfo("> Publisher correctly initialized")

        # Create the Subscriber to Joystick commands
        self.ros_sub_twist = rospy.Subscriber("/joy", Joy, self.setJoystickValues)
        rospy.loginfo("> Subscriber correctly initialized")

        # Create the subscriber to depthai detections


        # Create the subscriber to depthai depth data


        # Create subscriber to depthai images


        rospy.loginfo("Initialization complete")


    def processDepthaiDetections(self, message):
        pass

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
        a_button = buttons[1]
        self.joystick['steer_msg'] = axes[0]
        self.joystick['throttleMessage'] = axes[4]
        if a_button == 1:
            rospy.loginfo('Swapping autonomous modes, now: %s'.format(self.autonomous_mode))
            self.autonomous_mode = not self.autonomous_mode

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
            steerMessage = self.joystick['steer_msg']

        # if we're not in autonomous mode
        if self.autonomous_mode:
            pass

            # check for detections

            # If we don't have any detections, then drive in a circle to try to find detections
            throttleMessage = 0.1
            steerMessage = 1.0


        self.actuators['throttle'].getServoValue(self.joystick['throttleMessage'], 'throttle')
        self.actuators['steering'].getServoValue(self.joystick['steer_msg'], 'steer')

        rospy.loginfo("Got a command v = %2.1f  s = %2.1f"%(throttleMessage, steer_msg))

        self.sendServoMessage()


    def sendServoMessage(self):

        for actuator_name, servo_obj in iter(self.actuators.items()):
            print(servo_obj)
            self._servo_msg.servos[servo_obj.id-1].servo = servo_obj.id
            self._servo_msg.servos[servo_obj.id-1].value = servo_obj.value_out
            rospy.loginfo("Sending to %s command %d"%(actuator_name, servo_obj.value_out))

        self.ros_pub_servo_array.publish(self._servo_msg)

    def run(self):

        # Set the control rate
        # Run the loop @ 10hz
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            # Sleep until next cycle
            rate.sleep()

if __name__ == "__main__":
    chaser = DogChaser()
    chaser.run()