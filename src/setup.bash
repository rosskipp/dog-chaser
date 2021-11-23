#!/bin/bash

sudo chmod a+rw /dev/input/js0
sudo chmod o+rw /dev/gpio*

# this will fail if we're not running a ROS master
rosparam set joy_node/dev "/dev/input/js0"