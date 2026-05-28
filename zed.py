# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import socket
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy

# class ZedImageUdpSender(Node):
#     def __init__(self):
#         super().__init__('zed_image_udp_sender')
#         qos_profile = QoSProfile(
#             reliability=QoSReliabilityPolicy.BEST_EFFORT,
#             depth=10
#         )
#         self.subscription = self.create_subscription(
#             Image,
#             '/aruco_image_result',  
#             self.listener_callback,
#             qos_profile)
#             self.subscription = self.create_subscription(
#             Image,
#             '/object_image_result',  
#             self.listener_callback,
#             qos_profile)
#         self.bridge = CvBridge()
#         self.udp_ip = '192.168.2.100' #change it to Dashboards IP
#         self.udp_port = 5007
#         self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         self.frame_count = 0  # Add frame counter

#     def listener_callback(self, msg):
#         self.frame_count += 1
#         # if self.frame_count % 2!= 0:
#         #     return
#         self.get_logger().info("Received image frame")
#         cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#         # Downscale the image to reduce UDP packet size
#         cv_image = cv2.resize(cv_image, (720, 520))  # Try (160, 120) if still too large
#         # Compress with lower quality for smaller size
#         _, jpeg = cv2.imencode('.jpg', cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
#         data = jpeg.tobytes()
#         # Warn if still too large
#         if len(data) > 60000:
#             self.get_logger().warn(f"Image too large for UDP: {len(data)} bytes, skipping.")
#             return
#         self.sock.sendto(data, (self.udp_ip, self.udp_port))
# def main(args=None):
#     rclpy.init(args=args)
#     node = ZedImageUdpSender()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import socket
from rclpy.qos import QoSProfile, QoSReliabilityPolicy


class ZedImageUdpSender(Node):

    def __init__(self):
        super().__init__('zed_image_udp_sender')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # Subscribe to ArUco image topic
        self.aruco_subscription = self.create_subscription(
            Image,
            '/aruco_image_result',
            self.aruco_callback,
            qos_profile
        )

        # Subscribe to Object image topic
        self.object_subscription = self.create_subscription(
            Image,
            '/yolo/annotated_image',
            self.object_callback,
            qos_profile
        )

        self.bridge = CvBridge()

        self.udp_ip = '192.168.2.100'
        self.udp_port = 5007

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.frame_count = 0

    # ─────────────────────────────────────────────
    # ArUco callback
    # ─────────────────────────────────────────────
    def aruco_callback(self, msg):
        self.send_image(msg, "ARUCO")

    # ─────────────────────────────────────────────
    # Object callback
    # ─────────────────────────────────────────────
    def object_callback(self, msg):
        self.send_image(msg, "OBJECT")

    # ─────────────────────────────────────────────
    # Shared image sender
    # ─────────────────────────────────────────────
    def send_image(self, msg, source_name):

        self.frame_count += 1

        self.get_logger().info(f"Received {source_name} image frame")

        cv_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        # Resize image
        cv_image = cv2.resize(cv_image, (720, 520))

        # JPEG compress
        _, jpeg = cv2.imencode(
            '.jpg',
            cv_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 40]
        )

        data = jpeg.tobytes()

        # UDP size protection
        if len(data) > 60000:
            self.get_logger().warn(
                f"{source_name} image too large: {len(data)} bytes"
            )
            return
        payload = source_name.encode("utf-8") + b"|" + data
        self.sock.sendto(
            payload,
            (self.udp_ip, self.udp_port)
        )


def main(args=None):

    rclpy.init(args=args)

    node = ZedImageUdpSender()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()