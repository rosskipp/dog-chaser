#!/usr/bin/python3

import cv2
import depthai as dai
import math
import numpy as np
import blobconverter
import rospy
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32

###
### User Config
###

# User-defined constants for collision detection
WARNING = 1000  # 1m, orange
CRITICAL = 500  # 50cm, red

fullFrameTracking = True

# NN model
# model_path = blobconverter.from_zoo(
#     name="yolov7tiny_coco_640x352", zoo_type="depthai", shaves=6
# )
model_path = blobconverter.from_zoo(name="mobilenet-ssd", shaves=5)

###
### Label Map for MobileNetSSD
###
labelMap = [
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
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

###
### ROS SETUP
###

# CV bridge
bridge = CvBridge()

# Init ROS node
rospy.init_node("depthai_node")

# Create ROS publishers
imageObjectPub = rospy.Publisher("/object_tracker/image", Image, queue_size=1)
imageCollisionPub = rospy.Publisher("/collision_detection/depth_image", Image, queue_size=1)
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

###
### DepthAI Setup
###

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)
objectTracker = pipeline.create(dai.node.ObjectTracker)
spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)
xoutRgb = pipeline.create(dai.node.XLinkOut)
trackerOut = pipeline.create(dai.node.XLinkOut)

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
xoutRgb.setStreamName("preview")
trackerOut.setStreamName("tracklets")

### Properties

# RGB Camera
camRgb.setPreviewSize(300, 300)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

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
# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

spatialDetectionNetwork.setBlobPath(model_path)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

objectTracker.setDetectionLabelsToTrack([15])  # track only person
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

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

camRgb.preview.link(spatialDetectionNetwork.input)
objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
objectTracker.out.link(trackerOut.input)

if fullFrameTracking:
    camRgb.setPreviewKeepAspectRatio(False)
    camRgb.video.link(objectTracker.inputTrackerFrame)
    objectTracker.inputTrackerFrame.setBlocking(False)
    # do not block the pipeline if it's too slow on full frame
    objectTracker.inputTrackerFrame.setQueueSize(2)
else:
    spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
spatialDetectionNetwork.out.link(objectTracker.inputDetections)
stereo.depth.link(spatialDetectionNetwork.inputDepth)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    device.setIrLaserDotProjectorBrightness(1000)

    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(
        name="spatialData", maxSize=4, blocking=False
    )

    preview = device.getOutputQueue("preview", 4, False)
    tracklets = device.getOutputQueue("tracklets", 4, False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (0, 200, 40)
    fontType = cv2.FONT_HERSHEY_TRIPLEX

    while True:

        ###
        ### Collision Detection Logic
        ###
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

        leftDistance = None
        leftDetected = False
        centerDistance = None
        centerDetected = False
        rightDistance = None
        rightDetected = False

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
                        leftDetectedThisFrame = True
                if region == "center":
                    if centerDistance == None or distance < centerDistance:
                        centerDetected = True
                        centerDistance = distance
                        centerDetectedThisFrame = True
                if region == "right":
                    if rightDistance == None or distance < rightDistance:
                        rightDetected = True
                        rightDistance = distance
                        rightDetectedThisFrame = True

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

        ###
        ### End Collision Detection Logic
        ###


        ###
        ### Object Detection Logic
        ###

        imgFrame = preview.get()
        track = tracklets.get()

        counter += 1
        current_time = time.monotonic()
        if (current_time - startTime) > 1:
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        objFrame = imgFrame.getCvFrame()
        trackletsData = track.tracklets

        for t in trackletsData:
            roi = t.roi.denormalize(objFrame.shape[1], objFrame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            # print('x1', x1)
            # print('y1', y1)
            # print('x2', x2)
            # print('y2', y2)

            try:
                label = labelMap[t.label]
            except:
                label = t.label

            cv2.putText(
                objFrame,
                str(label),
                (x1 + 10, y1 + 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.75,
                255,
            )
            cv2.putText(
                objFrame,
                f"ID: {[t.id]}",
                (x1 + 10, y1 + 35),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.75,
                255,
            )
            cv2.putText(
                objFrame,
                t.status.name,
                (x1 + 10, y1 + 50),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.75,
                255,
            )
            cv2.rectangle(objFrame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

            cv2.putText(
                objFrame,
                f"X: {int(t.spatialCoordinates.x)} mm",
                (x1 + 10, y1 + 65),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.75,
                255,
            )
            cv2.putText(
                objFrame,
                f"Y: {int(t.spatialCoordinates.y)} mm",
                (x1 + 10, y1 + 80),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.putText(
                objFrame,
                f"Z: {int(t.spatialCoordinates.z)} mm",
                (x1 + 10, y1 + 95),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )

        cv2.putText(
            objFrame,
            "NN fps: {:.2f}".format(fps),
            (2, objFrame.shape[0] - 4),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.4,
            color,
        )

        ###
        ### End Object Detection Logic
        ###


        # Send the collison frame
        frame = bridge.cv2_to_imgmsg(depthFrameColor, "bgr8")
        imageCollisionPub.publish(frame)

        # send the object detetion frame
        objSendFrame = bridge.cv2_to_imgmsg(objFrame, "bgr8")
        imageObjectPub.publish(objSendFrame)

        # send the collision messages
        leftBoolPub.publish(leftDetected)
        leftDistancePub.publish(leftDistance)
        rightBoolPub.publish(rightDetected)
        rightDistancePub.publish(rightDistance)
        centerBoolPub.publish(centerDetected)
        centerDistancePub.publish(centerDistance)

        if cv2.waitKey(1) == ord("q"):
            break
