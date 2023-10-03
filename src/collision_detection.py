#!/usr/bin/python3

import cv2
import depthai as dai
import math
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32


# CV bridge
bridge = CvBridge()

# Init ROS node
rospy.init_node("collision_detection")

# Create ROS publishers
imagePub = rospy.Publisher("/collision_detection/depth_image", Image, queue_size=1)
leftBoolPub = rospy.Publisher("/collision_detection/left_collision", Bool, queue_size=1)
leftDistancePub = rospy.Publisher(
    "/collision_detection/left_distance", Float32, queue_size=1
)
rightBoolPub = rospy.Publisher(
    "/collision_detection/right_collision", Bool, queue_size=1
)
rightDistancePub = rospy.Publisher(
    "/collision_detection/right_distance", Float32, queue_size=1
)
centerBoolPub = rospy.Publisher(
    "/collision_detection/center_collision", Bool, queue_size=1
)
centerDistancePub = rospy.Publisher(
    "/collision_detection/center_distance", Float32, queue_size=1
)

# User-defined constants
WARNING = 1000  # 1m, orange
CRITICAL = 500  # 50cm, red

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

### Properties

# Mono
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setCamera("right")

# Stereo Depth
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setConfidenceThreshold(50)
stereo.setExtendedDisparity(True)
stereo.setSubpixel(True)
spatialLocationCalculator.inputConfig.setWaitForMessage(False)


# Create rows and columns of ROIs for the SpatialLocationCalculator
for x in range(15):
    for y in range(9):
        config = dai.SpatialLocationCalculatorConfigData()
        config.depthThresholds.lowerThreshold = 200
        config.depthThresholds.upperThreshold = 10000
        config.roi = dai.Rect(
            dai.Point2f((x + 0.5) * 0.0625, (y + 0.5) * 0.1),
            dai.Point2f((x + 1.5) * 0.0625, (y + 1.5) * 0.1),
        )
        spatialLocationCalculator.initialConfig.addROI(config)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    device.setIrLaserDotProjectorBrightness(1000)

    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(
        name="spatialData", maxSize=4, blocking=False
    )
    color = (0, 200, 40)
    fontType = cv2.FONT_HERSHEY_TRIPLEX

    leftDistance = None
    leftDetected = False
    centerDistance = None
    centerDetected = False
    rightDistance = None
    rightDetected = False

    while True:
        inDepth = (
            depthQueue.get()
        )  # Blocking call, will wait until a new data has arrived

        depthFrame = inDepth.getFrame()  # depthFrame values are in millimeters

        depth_downscaled = depthFrame[::4]
        min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
        max_depth = np.percentile(depth_downscaled, 99)
        depthFrameColor = np.interp(
            depthFrame, (min_depth, max_depth), (0, 255)
        ).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_PINK)

        spatialData = spatialCalcQueue.get().getSpatialLocations()

        width = inDepth.getWidth()
        leftStartX = 0
        leftEndX = width / 3
        centerStartX = leftEndX
        centerEndX = leftEndX * 2
        rightStartX = centerEndX
        rightEndX = width

        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(
                width=depthFrameColor.shape[1], height=depthFrameColor.shape[0]
            )

            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            # Track the left/center/right distances
            region = None
            if xmin > leftStartX and xmax < leftEndX:
                region = "left"
            elif xmin > centerStartX and xmax < centerEndX:
                region = "center"
            elif xmin > rightStartX and xmax < rightEndX:
                region = "right"

            coords = depthData.spatialCoordinates
            distance = math.sqrt(coords.x**2 + coords.y**2 + coords.z**2)

            if distance == 0:  # Invalid
                continue

            if distance < CRITICAL:
                if region == "left":
                    if leftDistance == None or distance < leftDistance:
                        leftDetected = True
                        leftDistance = distance
                if region == "center":
                    if centerDistance == None or distance < centerDistance:
                        centerDetected = True
                        centerDistance = distance
                if region == "right":
                    if rightDistance == None or distance < rightDistance:
                        rightDetected = True
                        rightDistance = distance

                color = (0, 0, 255)
                cv2.rectangle(
                    depthFrameColor, (xmin, ymin), (xmax, ymax), color, thickness=4
                )
                cv2.putText(
                    depthFrameColor,
                    "{:.1f}m".format(distance / 1000),
                    (xmin + 10, ymin + 20),
                    fontType,
                    0.5,
                    color,
                )
            elif distance < WARNING:
                color = (0, 140, 255)
                cv2.rectangle(
                    depthFrameColor, (xmin, ymin), (xmax, ymax), color, thickness=2
                )
                cv2.putText(
                    depthFrameColor,
                    "{:.1f}m".format(distance / 1000),
                    (xmin + 10, ymin + 20),
                    fontType,
                    0.5,
                    color,
                )

        # Show the frame
        # cv2.imshow("depth", depthFrameColor)
        # Send the frame
        frame = bridge.cv2_to_imgmsg(depthFrameColor, "bgr8")
        imagePub.publish(frame)

        # send the collision messages
        leftBoolPub.publish(leftDetected)
        leftDistancePub.publish(leftDistance)
        rightBoolPub.publish(rightDetected)
        rightDistancePub.publish(rightDistance)
        centerBoolPub.publish(centerDetected)
        centerDistancePub.publish(centerDistance)

        if cv2.waitKey(1) == ord("q"):
            break
