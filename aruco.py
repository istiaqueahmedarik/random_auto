#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import numpy as np
import json
import math

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')

        self.declare_parameter('image_topic', '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/rgb/color/rect/camera_info')
        self.declare_parameter('marker_size', 0.25)
        self.declare_parameter('allowed_marker_ids', [1, 2])
        
        image_topic = self.get_parameter('image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.marker_size = self.get_parameter('marker_size').value
        self.allowed_marker_ids = self.get_parameter('allowed_marker_ids').value

        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Image, image_topic, self.image_callback, 10)

        # Publish relative coordinates
        self.rel_pub = self.create_publisher(String, '/aruco_relative_positions', 10)
        self.image_pub = self.create_publisher(Image, '/aruco_image_result', 10)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        else:
            self.detector = None

        self.get_logger().info("Basic ArUco Detector Initialized (Relative XYZ)")

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Camera matrix initialized.")

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        results = {}

        if ids is not None and len(ids) > 0:
            filtered_corners = []
            filtered_ids = []
            for i in range(len(ids)):
                if int(ids[i][0]) in self.allowed_marker_ids:
                    filtered_corners.append(corners[i])
                    filtered_ids.append(ids[i])

            if len(filtered_ids) > 0:
                for i in range(len(filtered_ids)):
                    marker_id = int(filtered_ids[i][0])
                    c = filtered_corners[i]
                    
                    # Estimate pose
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        c, self.marker_size, self.camera_matrix, self.dist_coeffs
                    )
                    
                    # tvec is in optical frame: x-right, y-down, z-forward
                    # distance to marker is primarily z
                    x_opt, y_opt, z_opt = tvec[0][0]
                    
                    # Calculate Euclidean distance on ground plane
                    distance = float(math.hypot(x_opt, z_opt))
                    # Left/Right offset (optical x is right, so -x is left)
                    offset = float(-x_opt)

                    results[str(marker_id)] = {
                        "distance": distance,
                        "offset": offset
                    }
                
                # Draw markers and axes
                filtered_ids_np = np.array(filtered_ids)
                cv2.aruco.drawDetectedMarkers(cv_image, filtered_corners, filtered_ids_np)
                for i in range(len(filtered_ids)):
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        filtered_corners[i], self.marker_size, self.camera_matrix, self.dist_coeffs
                    )
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_size / 2)

        # Publish the relative positions if any markers were processed
        if len(results) > 0:
            msg_str = String()
            msg_str.data = json.dumps(results)
            self.rel_pub.publish(msg_str)

        # Publish the image with drawn markers
        try:
            image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.image_pub.publish(image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
