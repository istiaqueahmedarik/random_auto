#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from rclpy.qos import qos_profile_sensor_data

class GpsConverterNode(Node):
    def __init__(self):
        super().__init__('gps_converter_node')

        # 0 = Auto origin from GPS, 1 = Manual origin (0,0)
        usa = 0

        # Declare parameters with default topics
        self.declare_parameter('odom_topic', '/Odometry')
        self.declare_parameter('heading_topic', '/mavros/global_position/compass_hdg')
        self.declare_parameter('gps_source_topic', '/mavros/global_position/global')
        self.declare_parameter('publish_topic', '/converted_gps')

        # State Variables
        self.origin_received = False
        self.heading_received = False
        self.initial_heading = None
        self.r_earth = 6378137.0

        # Apply Flag Logic for Origin
        if usa == 1:
            self.start_lat = float(input("Enter starting latitude: "))
            self.start_lon = float(input("Enter starting longitude: "))
            self.start_alt = 0.0
            self.origin_received = True
            self.get_logger().info(f"Using MANUAL origin: {self.start_lat}, {self.start_lon}")
        else:
            self.start_lat = None
            self.start_lon = None
            self.start_alt = None
            self.get_logger().info("Waiting for GPS topic to provide origin...")
            
            # Subscribing using standard history depth (10) to match MAVROS QoS
            self.gps_source_sub = self.create_subscription(
                NavSatFix,
                self.get_parameter('gps_source_topic').value,
                self.gps_origin_callback,
                10
            )

        # Heading Subscriber (Changed from Best Effort to 10 for compatibility)
        self.heading_sub = self.create_subscription(
            Float64,
            self.get_parameter('heading_topic').value,
            self.heading_callback,
            10
        )

        # Odometry Subscriber
        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self.odom_callback,
            10
        )

        # Publisher
        self.gps_pub = self.create_publisher(
            NavSatFix,
            self.get_parameter('publish_topic').value,
            10
        )

    def gps_origin_callback(self, msg):
        if not self.origin_received:
            self.start_lat = msg.latitude
            self.start_lon = msg.longitude
            self.start_alt = msg.altitude
            self.origin_received = True
            self.get_logger().info(f"Origin Locked via Topic: Lat={self.start_lat}, Lon={self.start_lon}")

    def heading_callback(self, msg):
        if not self.heading_received:
            self.initial_heading = msg.data
            self.heading_received = True
            self.get_logger().info(f"Initial heading locked: {self.initial_heading} degrees")

    def odom_callback(self, msg):
        # Debug logger to see what is blocking the execution
        print("work")
        if not self.origin_received or not self.heading_received:
            self.get_logger().info(
                f"Waiting for prerequisites... Origin Rx: {self.origin_received} | Heading Rx: {self.heading_received}", 
                throttle_duration_sec=2.0
            )
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        # Convert heading to radians and compute local offsets
        psi_rad = math.radians(self.initial_heading)
        dn = x * math.cos(psi_rad) + y * math.sin(psi_rad)
        de = x * math.sin(psi_rad) - y * math.cos(psi_rad)

        # Calculate changes in Latitude and Longitude
        dLat = (dn / self.r_earth) * (180.0 / math.pi)
        dLon = (de / (self.r_earth * math.cos(math.radians(self.start_lat)))) * (180.0 / math.pi)

        # Construct and publish NavSatFix message
        gps_msg = NavSatFix()
        gps_msg.header = msg.header
        gps_msg.latitude = self.start_lat + dLat
        gps_msg.longitude = self.start_lon + dLon
        gps_msg.altitude = self.start_alt + z
        gps_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

        self.gps_pub.publish(gps_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GpsConverterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()