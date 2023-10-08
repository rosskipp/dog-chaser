#!/usr/bin/python3

import rospy, time, math
import cv2
from cv_bridge import CvBridge

from geometry_msgs.msg import Point
from sensor_msgs.msg import Range, Image
from std_msgs.msg import Float32, Bool, String


class Debugger:
    def __init__(self, labelMap, startTime, saveImages):
        self.SAVE_IMAGES = saveImages
        self.sendImageCount = 2  # send every __ images
        self.imageCounter = 0
        self.sendImageCounter = 0
        self.bridge = CvBridge()
        self.labelMap = labelMap
        self.startTime = startTime

        # Create debug publishers
        self.publishDebugImage = rospy.Publisher(
            "/dog_chaser/debug_image", Image, queue_size=1
        )
        self.publishDebugSteer = rospy.Publisher(
            "/dog_chaser/debug_steer", Float32, queue_size=1
        )
        self.publishDebugThrottle = rospy.Publisher(
            "/dog_chaser/debug_throttle", Float32, queue_size=1
        )
        self.publishDebugDogBool = rospy.Publisher(
            "/dog_chaser/debug_found_dog", Bool, queue_size=1
        )
        self.publishDebugDogPosition = rospy.Publisher(
            "/dog_chaser/debug_dog_position", Point, queue_size=1
        )
        self.publishDebugDogAngle = rospy.Publisher(
            "/dog_chaser/debug_dog_angle", Float32, queue_size=1
        )
        self.publishDebugLeftSonar = rospy.Publisher(
            "/dog_chaser/debug_left_sonar", Float32, queue_size=1
        )
        self.publishDebugCenterSonar = rospy.Publisher(
            "/dog_chaser/debug_center_sonar", Float32, queue_size=1
        )
        self.publishDebugRightSonar = rospy.Publisher(
            "/dog_chaser/debug_right_sonar", Float32, queue_size=1
        )
        self.publishIsTracking = rospy.Publisher(
            "/dog_chaser/is_tracking", String, queue_size=1
        )
        self.publishTrackingStatus = rospy.Publisher(
            "/dog_chaser/tracking_status", String, queue_size=1
        )

    def sendDebugValues(
        self,
        steer,
        throttle,
        foundDog,
        dogPosition,
        dogAngle,
        leftSonar,
        centerSonar,
        rightSonar,
        isTracking,
        trackingStatus,
    ):
        self.publishDebugSteer.publish(steer)
        self.publishDebugThrottle.publish(throttle)
        self.publishDebugDogBool.publish(foundDog)
        self.publishDebugDogPosition.publish(dogPosition)
        self.publishDebugDogAngle.publish(dogAngle)
        self.publishDebugLeftSonar.publish(leftSonar)
        self.publishDebugCenterSonar.publish(centerSonar)
        self.publishDebugRightSonar.publish(rightSonar)
        self.publishIsTracking.publish(str(isTracking))
        self.publishTrackingStatus.publish(str(trackingStatus))

    def sendDebugImage(self, frame, detections):
        counter = self.imageCounter
        counter += 1
        i = self.sendImageCounter
        if frame.height != 0:
            # convert image to cv2
            frame = self.bridge.imgmsg_to_cv2(frame, "bgr8")
            color2 = (255, 255, 255)

            # on every 10th time through, send an image
            if i > self.sendImageCount:
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
                    bbox = (
                        (center_x - halfsize_x),
                        (center_y - halfsize_y),
                        (center_x + halfsize_x),
                        (center_y + halfsize_y),
                    )
                    # Put label on the image
                    cv2.putText(
                        frame,
                        self.labelMap[detection.results[0].id],
                        (bbox[0] + 10, bbox[1] + 20),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        255,
                    )
                    # Put score on the image
                    cv2.putText(
                        frame,
                        f"{int(detection.results[0].score * 100)}%",
                        (bbox[0] + 10, bbox[1] + 40),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        255,
                    )
                    # Put the depth measurement on the image
                    cv2.putText(
                        frame,
                        "position = x: {}".format(position_x),
                        (bbox[0] + 10, bbox[1] + 60),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        255,
                    )
                    cv2.putText(
                        frame,
                        "y:{}".format(position_y),
                        (bbox[0] + 10, bbox[1] + 80),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        255,
                    )
                    cv2.putText(
                        frame,
                        "z:{}".format(position_x),
                        (bbox[0] + 10, bbox[1] + 100),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        255,
                    )
                    # Put the bounding box on the image
                    cv2.rectangle(
                        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2
                    )

                cv2.putText(
                    frame,
                    "NN fps: {:.2f}".format(
                        counter / (time.monotonic() - self.startTime)
                    ),
                    (2, frame.shape[0] - 4),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.4,
                    color2,
                )
                # Save image if indicated
                if self.SAVE_IMAGES:
                    cv2.imwrite(
                        "/home/ubuntu/robot_data/images/{}.jpeg".format(time.time_ns()),
                        frame,
                    )
                # convert the image back to ROS image
                frame = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                self.publishDebugImage.publish(frame)

            i += 1
            self.imageCounter = counter
            self.sendImageCounter = i
