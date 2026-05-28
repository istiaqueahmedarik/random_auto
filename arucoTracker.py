#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float64
from cv_bridge import CvBridge

import cv2
import numpy as np
import json
import math

class ArucoOdomTracker(Node):
    def __init__(self):
        super().__init__('aruco_odom_tracker')

        # Declare parameters
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/rgb/color/rect/camera_info')
        self.declare_parameter('odom_topic', '/Odometry')
        self.declare_parameter('heading_topic', '/mavros/global_position/compass_hdg')
        self.declare_parameter('marker_size', 0.25) # Marker size in meters
        
        # List of allowed IDs. If empty[], it will accept ALL markers.
        self.declare_parameter('allowed_marker_ids',[0,1,2]) 

        image_topic = self.get_parameter('image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        heading_topic = self.get_parameter('heading_topic').value
        self.marker_size = self.get_parameter('marker_size').value
        self.allowed_marker_ids = self.get_parameter('allowed_marker_ids').value
        self.sensor_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # Subscribers
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.create_subscription(Float64, heading_topic, self.heading_callback, 10)
        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Image, image_topic, self.image_callback, 10)

        # Publishers
        self.dict_pub = self.create_publisher(String, '/aruco_global_positions', 10)
        self.image_pub = self.create_publisher(Image, '/aruco_image_result', 10)

        # State Variables
        self.bridge = CvBridge()
        self.latest_odom = None
        self.latest_heading = None
        self.initial_heading = None
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Memory of all tags seen: { marker_id: {"x": .., "y": .., "z": ..} }
        self.tags_seen = {}

        # Set up ArUco DICT_4X4_250
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # OpenCV 4.7+ vs older versions compatibility
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        else:
            self.detector = None

        self.get_logger().info(f"ArUco Tracker Initialized. Filtering for IDs: {self.allowed_marker_ids if self.allowed_marker_ids else 'ALL'}")

    @staticmethod
    def optical_to_ros_flu(tvec_optical):
        """Convert OpenCV/ROS-optical frame vector [x, y, z] to ROS FLU [x, y, z].

        Optical frame: x-right, y-down, z-forward
        ROS FLU frame: x-forward, y-left, z-up
        """
        x_opt, y_opt, z_opt = tvec_optical
        return np.array([z_opt, -x_opt, -y_opt], dtype=np.float64)

    def odom_callback(self, msg):
        self.latest_odom = msg.pose.pose

    def heading_callback(self, msg):
        if self.initial_heading is None:
            self.initial_heading = msg.data
            self.get_logger().info(f"Captured initial heading: {self.initial_heading:.1f} degrees")
        
        # Calculate heading relative to startup, so it acts like it starts at 0
        self.latest_heading = msg.data - self.initial_heading

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Camera info received.")

    def quat_to_matrix(self, q):
        """Convert a geometry_msgs Quaternion to a 3x3 Rotation Matrix"""
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],[2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],[2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])

    def heading_to_matrix(self, heading_degrees):
        """Convert a heading in degrees to a 3x3 Z-axis Rotation Matrix"""
        yaw = math.radians(heading_degrees)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

    def image_callback(self, msg):
        if self.camera_matrix is None or self.latest_odom is None or self.latest_heading is None:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        drawn_image = cv_image.copy() # Make a copy to draw on

        if self.detector:
            corners, ids, rejected = self.detector.detectMarkers(cv_image)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            filtered_corners = []
            filtered_ids =[]

            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                if len(self.allowed_marker_ids) == 0 or marker_id in self.allowed_marker_ids:
                    filtered_corners.append(corners[i])
                    filtered_ids.append([marker_id])
            
            if len(filtered_ids) > 0:
                filtered_corners = tuple(filtered_corners)
                filtered_ids = np.array(filtered_ids, dtype=np.int32)

                cv2.aruco.drawDetectedMarkers(drawn_image, filtered_corners, filtered_ids)

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    filtered_corners,
                    self.marker_size,
                    self.camera_matrix,
                    self.dist_coeffs,
                )

                odom_pos = np.array([
                    float(self.latest_odom.position.x),
                    float(self.latest_odom.position.y),
                    float(self.latest_odom.position.z)
                ])
                odom_rot_matrix = self.heading_to_matrix(self.latest_heading)

                self.get_logger().info(
                    f"Applying Offset: Pos(X={odom_pos[0]:.2f}, Y={odom_pos[1]:.2f}) Heading={self.latest_heading:.1f}deg", 
                    throttle_duration_sec=2.0
                )

                for i in range(len(filtered_ids)):
                    marker_id = int(filtered_ids[i][0])

                    rvec = rvecs[i][0]
                    tvec_optical = tvecs[i][0]
                    cv2.drawFrameAxes(
                        drawn_image,
                        self.camera_matrix,
                        self.dist_coeffs,
                        rvec,
                        tvec_optical,
                        self.marker_size / 2,
                    )

                    t_cam_ros = self.optical_to_ros_flu(tvec_optical)
                    t_global_odom = np.dot(odom_rot_matrix, t_cam_ros) + odom_pos

                    self.tags_seen[marker_id] = {
                        "x": round(float(t_global_odom[0]), 3),
                        "y": round(float(t_global_odom[1]), 3),
                        "z": round(float(t_global_odom[2]), 3)
                    }

                    self.get_logger().info(
                        f"ID {marker_id} local(optical): "
                        f"x={tvec_optical[0]:.3f}, y={tvec_optical[1]:.3f}, z={tvec_optical[2]:.3f} m",
                        throttle_duration_sec=1.0,
                    )

        dict_msg = String()
        dict_msg.data = json.dumps(self.tags_seen)
        self.dict_pub.publish(dict_msg)

        out_img_msg = self.bridge.cv2_to_imgmsg(drawn_image, encoding='bgr8')
        out_img_msg.header = msg.header
        self.image_pub.publish(out_img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoOdomTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
