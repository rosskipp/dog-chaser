# Dog Chaser

Code for my dog-chasing robot.

## Hardware
- Raspberry Pi running Ubuntu
- Luxonis Oak-D camera
- 3x Sonar sensors
- Some sort of chassis - I am using a modified version of this: https://www.instructables.com/FPV-Rover-V20/

### Dependencies
- Ubuntu
- ROS noetic
- Install the Luxonis ros package: https://github.com/luxonis/depthai-ros
- you need ros-i2cpwmboard package: https://gitlab.com/bradanlane/ros-i2cpwmboard
- Install Foxglove studio bridge (if you want to see the data) `sudo apt install ros-noetic-foxglove-bridge`
- I use a USB joystick - if you don't want to use this then you could change the launch file for a keyboard

### Running
- make sure all the things are plugged in 😄
- run the src/setup.bash file
- export ROS_IP=raspberry_pi_IP
