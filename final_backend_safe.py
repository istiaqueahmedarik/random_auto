#!/usr/bin/env python3
"""
Unified Navigation Backend — test_backend.py

Handles both GPS-only and GPS+ArUco mission types.

GPS mode:
- Converts GPS goal to odom-frame using gps_to_odometry
- Generates sequential intermediate waypoints from current ZED odom pose
- Navigates with retry + skip-on-failure
- Publishes result to /mission/reached_waypoint

ArUco mode:
- Same GPS approach phase as above
- After GPS waypoints complete → spiral search around GPS goal
- Subscribes to /aruco_relative_positions for ArUco marker detections
- Interrupts spiral to dock at detected ArUco markers
- Publishes result to /mission/reached_waypoint

Control:
- /mission/waypoint  (JSON with starting/ending GPS, type)
- /mission/control   (start / stop / pause)
- /zed/zed_node/odom (robot position)
"""


import json
import math
import random
import socket
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import String, Float64, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix
from tf2_ros import Buffer, TransformException, TransformListener

from mavros_msgs.msg import Waypoint, RCOut, WaypointReached, State
from mavros_msgs.srv import WaypointClear, WaypointPush, SetMode, CommandBool

from gps_calc import gps_to_odometry


# ── Constants ───────────────────────────────────────────────────────────────
DEFAULT_WAYPOINT_SPACING = 8.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_NAV_FRAME = "map"
DEFAULT_ODOM_TOPIC = "/Odometry"
DEFAULT_ARUCO_TOPIC = "/aruco_relative_positions"
DEFAULT_SPIRAL_RADIUS = 10.0
DEFAULT_TARGET_TIMEOUT_SEC = 2.0
RANDOM_WALK_DISTANCE_M = 0.0
RANDOM_WALK_STEP_M = 0.25
OBJECT_SEARCH_ROTATE_TOTAL_SEC = 60.0
OBJECT_SEARCH_SPIRAL_TOTAL_SEC = 120.0
OBJECT_CONFIRM_SEC = 3.0

STATUS_NAMES = {
    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
    GoalStatus.STATUS_CANCELED: "CANCELED",
    GoalStatus.STATUS_ABORTED: "ABORTED",
}

UDP_DASHBOARD_IP = "192.168.2.15"
UDP_DASHBOARD_PORT = 5005


# ── ArUco target dataclass ──────────────────────────────────────────────────
@dataclass
class ArucoTarget:
    marker_id: int
    x: float
    y: float
    z: float
    confidence: float
    last_seen_ns: int


# ── Spiral search point generator ──────────────────────────────────────────
def spiral_search_points(
    initial_x: float = 0,
    initial_y: float = 0,
    spiral_radius: float = 10,
) -> List[Tuple[float, float]]:
    """Generate points along a spiral path around (initial_x, initial_y).
    Prepends a random-walk segment totaling 1.5m.
    """
    angles = [i * 45 for i in range(8)]
    a = 0.2
    spiral_arm_gap = 2 * math.pi * a
    full_rotations = int((spiral_radius + (spiral_arm_gap - 1)) / spiral_arm_gap)

    points: List[Tuple[float, float]] = []
    # Random-walk prefix (total distance = 1.5m)
    if RANDOM_WALK_DISTANCE_M > 0 and RANDOM_WALK_STEP_M > 0:
        steps = max(1, int(round(RANDOM_WALK_DISTANCE_M / RANDOM_WALK_STEP_M)))
        x, y = initial_x, initial_y
        for _ in range(steps):
            theta = random.uniform(0.0, 2.0 * math.pi)
            x += RANDOM_WALK_STEP_M * math.cos(theta)
            y += RANDOM_WALK_STEP_M * math.sin(theta)
            points.append((x, y))

    for i in range(full_rotations):
        for angle in angles:
            theta = math.radians(angle) + (i * 2 * math.pi)
            r = a * theta
            x = initial_x + r * math.cos(theta)
            y = initial_y + r * math.sin(theta)
            points.append((x, y))
    return points


# ── Helper ──────────────────────────────────────────────────────────────────
def yaw_to_quaternion(yaw: float):
    """Convert yaw (radians) → (x, y, z, w) quaternion."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Convert quaternion to yaw (radians)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


# ═══════════════════════════════════════════════════════════════════════════
class TestBackendNode(Node):
    """Unified GPS + GPS+ArUco navigation backend."""

    OBJECT_MISSION_TYPES = ("object_1", "object_2", "object_3")

    def __init__(self):
        super().__init__("test_backend")

        # ── UDP dashboard telemetry ──────────────────────────────────────
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # ── Sensor QoS (BEST_EFFORT for MAVROS / ZED topics) ────────────
        self.sensor_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Config ───────────────────────────────────────────────────────
        self.nav_frame = DEFAULT_NAV_FRAME
        self.odom_topic = DEFAULT_ODOM_TOPIC
        self.waypoint_spacing = DEFAULT_WAYPOINT_SPACING
        self.max_retries = DEFAULT_MAX_RETRIES
        self.spiral_radius = DEFAULT_SPIRAL_RADIUS
        self.target_timeout_sec = DEFAULT_TARGET_TIMEOUT_SEC

        # ── Nav2 action client ───────────────────────────────────────────
        self._action_client = ActionClient(
            self, NavigateToPose, "navigate_to_pose"
        )
        self._nav2_ready = False
        self._nav2_check_count = 0

        # ── TF (for map↔odom transforms) ────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── MAVROS Clients & State ──────────────────────────────────────
        self.wp_clear_client = self.create_client(WaypointClear, "/mavros/mission/clear")
        self.wp_push_client = self.create_client(WaypointPush, "/mavros/mission/push")
        self.set_mode_client = self.create_client(SetMode, "/mavros/set_mode")
        self.arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.home_lat_var = None
        self.home_lon_var = None
        self.mavros_mode = "MANUAL"

        # ── Mission state ────────────────────────────────────────────────
        self.current_state = "IDLE"  # IDLE | GPS_NAV | SPIRAL_SEARCH | ARUCO_NAV | SPIRAL_COMPLETE_WAIT | MAVROS_GPS_NAV
        self.mission_type = "gps"   # "gps" or "aruco"
        self.current_goal_handle = None
        self._nav_goal_pending = False
        self._active_goal_phase: Optional[str] = None  # gps_segment | spiral_segment | aruco_dock
        self._retry_count = 0

        # GPS waypoint sequence
        self.waypoints: List[Tuple[float, float]] = []
        self.current_wp_index = 0
        self.results: List[Tuple[int, float, float, str]] = []

        # Spiral search
        self.spiral_points: List[Tuple[float, float]] = []
        self._spiral_center: Optional[Tuple[float, float]] = None

        # ArUco state
        self.targets: Dict[int, ArucoTarget] = {}
        self.visited_markers: List[int] = []
        self._pending_aruco_target: Optional[ArucoTarget] = None

        # Pause / Stop
        self._is_paused = False
        self._is_stopped = False
        self._paused_goal_data: Optional[dict] = None

        # Object detection search
        self.object_detected = False
        self._object_search_start_ns: Optional[int] = None
        self._object_phase_start_ns: Optional[int] = None
        self._object_spiral_points: List[Tuple[float, float]] = []
        self._object_spiral_index = 0
        self._object_confirm_deadline_ns: Optional[int] = None
        self._object_resume_state: Optional[str] = None

        # ── ZED odom ─────────────────────────────────────────────────────
        self._odom_lock = threading.Lock()
        self._latest_odom: Optional[Odometry] = None
        self._have_odom = False

        # ── GPS + Heading (for GPS-to-odom conversion) ──────────────────
        self.initial_lat: Optional[float] = None
        self.initial_lon: Optional[float] = None
        self.initial_alt: Optional[float] = None
        self.initial_heading_deg: Optional[float] = None

        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0
        self.gps_received = False
        self.current_heading_deg = 0.0
        self.heading_received = False

        # ── Subscriptions ────────────────────────────────────────────────
        self.create_subscription(
            Odometry, self.odom_topic, self._odom_cb, 10
        )
        self.create_subscription(
            NavSatFix, "/mavros/global_position/global", self._gps_cb, self.sensor_qos
        )
        self.create_subscription(
            Float64,
            "/mavros/global_position/compass_hdg",
            self._heading_cb,
            self.sensor_qos,
        )
        self.create_subscription(
            String, "/mission/waypoint", self._waypoint_cb, 10
        )
        self.create_subscription(
            String, "/mission/control", self._control_cb, 10
        )
        self.create_subscription(
            String, DEFAULT_ARUCO_TOPIC, self._aruco_cb, 10
        )
        self.create_subscription(
            Bool, "/prediction", self._object_cb, 10
        )
        self.create_subscription(
            RCOut, "/mavros/rc/out", self._rcout_cb, self.sensor_qos
        )
        self.create_subscription(
            WaypointReached, "/mavros/mission/reached", self._wp_reached_cb, self.sensor_qos
        )
        self.create_subscription(
            State, "/mavros/state", self._mavros_state_cb, self.sensor_qos
        )

        # ── Publishers ───────────────────────────────────────────────────
        self.status_pub = self.create_publisher(
            String, "/mission/reached_waypoint", 10
        )
        self.color_pub = self.create_publisher(
            String, "/color_topic", 10
        )
        self.prediction_pub = self.create_publisher(
            Bool, "/prediction", 10
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist, "/cmd_vel", 10
        )

        # ── Timers ───────────────────────────────────────────────────────
        self.create_timer(2.0, self._check_nav2_ready)
        self.create_timer(0.5, self._tick)
        self.create_timer(1.0, self._send_gps_udp)

        self.get_logger().info("═" * 50)
        self.get_logger().info("  TEST BACKEND — Unified Navigator Online")
        self.get_logger().info(f"  Odom: {self.odom_topic} | Frame: {self.nav_frame}")
        self.get_logger().info(f"  ArUco: {DEFAULT_ARUCO_TOPIC}")
        self.get_logger().info("═" * 50)

    # ─────────────────────────────────────────────────────────────────────
    #  Sensor callbacks
    # ─────────────────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        with self._odom_lock:
            self._latest_odom = msg
            self._have_odom = True

    def _gps_cb(self, msg: NavSatFix):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude
        self.current_alt = msg.altitude
        if not self.gps_received:
            self.gps_received = True
            if self.initial_lat is None:
                self.initial_lat = self.current_lat
                self.initial_lon = self.current_lon
                self.initial_alt = self.current_alt
            self.get_logger().info(
                f"[GPS] Connected: lat={self.current_lat:.6f}, lon={self.current_lon:.6f}"
            )

    def _heading_cb(self, msg: Float64):
        self.current_heading_deg = msg.data % 360.0
        if not self.heading_received:
            self.heading_received = True
            if self.initial_heading_deg is None:
                self.initial_heading_deg = self.current_heading_deg
            self.get_logger().info(f"[HEADING] Connected: {self.current_heading_deg:.1f}°")

    def _object_cb(self, msg: Bool):
        self.object_detected = bool(msg.data)

    # ─────────────────────────────────────────────────────────────────────
    #  Robot pose from ZED odom
    # ─────────────────────────────────────────────────────────────────────
    def _get_robot_pose(self) -> Optional[Tuple[float, float]]:
        with self._odom_lock:
            if not self._have_odom or self._latest_odom is None:
                return None
            p = self._latest_odom.pose.pose.position
            return (p.x, p.y)

    def _odom_to_nav_frame(self, x_odom: float, y_odom: float) -> Optional[Tuple[float, float]]:
        """Convert an odom-frame (x, y) point into current nav_frame coordinates."""
        if self.nav_frame == "odom":
            return (x_odom, y_odom)

        try:
            tf = self.tf_buffer.lookup_transform(
                self.nav_frame,
                "odom",
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
            c = math.cos(yaw)
            s = math.sin(yaw)

            x_nav = t.x + (c * x_odom - s * y_odom)
            y_nav = t.y + (s * x_odom + c * y_odom)
            return (x_nav, y_nav)
        except TransformException as e:
            self.get_logger().warn(
                f"TF lookup failed (odom→{self.nav_frame}): {e}",
                throttle_duration_sec=2.0,
            )
            return None

    def _get_robot_pose_in_nav_frame(self) -> Optional[Tuple[float, float]]:
        pose_odom = self._get_robot_pose()
        if pose_odom is None:
            return None
        return self._odom_to_nav_frame(pose_odom[0], pose_odom[1])

    # ─────────────────────────────────────────────────────────────────────
    #  Nav2 readiness
    # ─────────────────────────────────────────────────────────────────────
    def _check_nav2_ready(self):
        if not self._nav2_ready and self._action_client.server_is_ready():
            self._nav2_ready = True
            self.get_logger().info("Nav2 action server READY")
        self._nav2_check_count += 1

    # ─────────────────────────────────────────────────────────────────────
    #  Waypoint generation
    # ─────────────────────────────────────────────────────────────────────
    def _generate_waypoints(
        self, start_x: float, start_y: float, goal_x: float, goal_y: float
    ) -> List[Tuple[float, float]]:
        dx = goal_x - start_x
        dy = goal_y - start_y
        total_dist = math.hypot(dx, dy)

        if total_dist < self.waypoint_spacing:
            return [(goal_x, goal_y)]

        num_segments = int(math.ceil(total_dist / self.waypoint_spacing))
        points: List[Tuple[float, float]] = []
        for i in range(1, num_segments):
            frac = (i * self.waypoint_spacing) / total_dist
            points.append((start_x + frac * dx, start_y + frac * dy))
        points.append((goal_x, goal_y))
        return points

    # ─────────────────────────────────────────────────────────────────────
    #  Mission waypoint callback (from main.py)
    # ─────────────────────────────────────────────────────────────────────
    def _waypoint_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid waypoint JSON: {e}")
            return

        # Sensors check
        if not self.gps_received or not self.heading_received:
            self.get_logger().warn("GPS/Heading not ready, skipping waypoint")
            return

        if not self._have_odom:
            self.get_logger().warn("Local odom not received yet, cannot process waypoint")
            return

        if self.current_goal_handle is not None:
            self.get_logger().warn("Already navigating, ignoring new waypoint")
            return

        # Parse GPS goal
        try:
            ending = data.get("ending", {})
            goal_lat = float(ending.get("latitude", 0.0))
            goal_lon = float(ending.get("longitude", 0.0))
            goal_alt = float(ending.get("altitude", 0.0))
        except (ValueError, TypeError) as e:
            self.get_logger().error(f"Invalid GPS goal coordinates: {e}")
            return

        # Determine mission type from the message
        self.mission_type = str(data.get("type", "gps")).lower().strip()
        if self.mission_type in ("object_1", "object"):
            self.mission_type = "object_1"
        if self.mission_type not in ("gps", "aruco", *self.OBJECT_MISSION_TYPES):
            self.mission_type = "gps"

        if self.mission_type not in ("gps", "aruco") and not self._nav2_ready:
            self.get_logger().warn(f"Nav2 not ready, cannot process {self.mission_type} waypoint")
            return

        if self.mission_type not in ("gps", "aruco") and not self._have_odom:
            self.get_logger().warn("Local odom not received yet, cannot process waypoint")
            return

        if self.current_goal_handle is not None or self.current_state.startswith("MAVROS_GPS_NAV"):
            self.get_logger().warn("Already navigating, ignoring new waypoint")
            return

        self.get_logger().info(
            f"[WAYPOINT] Received mission type={self.mission_type}, "
            f"goal=({goal_lat:.6f}, {goal_lon:.6f})"
        )

        # Save for pause/resume
        self._paused_goal_data = data
        self._is_stopped = False
        self._is_paused = False

        # Always convert GPS to odom-frame to get the spiral center
        init_lat = self.initial_lat if self.initial_lat is not None else self.current_lat
        init_lon = self.initial_lon if self.initial_lon is not None else self.current_lon
        init_alt = self.initial_alt if self.initial_alt is not None else self.current_alt
        init_heading = self.initial_heading_deg if self.initial_heading_deg is not None else self.current_heading_deg
        
        theta_rad = math.radians(init_heading)
        rel_x, rel_y, _ = gps_to_odometry(
            init_lat, init_lon, init_alt,
            goal_lat, goal_lon, goal_alt,
            theta_rad,
        )
        
        self._spiral_center = (rel_x, rel_y)

        # Clear aruco state for new mission
        self.spiral_points = []
        self.visited_markers.clear()
        self.targets.clear()
        self._pending_aruco_target = None

        if self.mission_type in ("gps", "aruco"):
            self.get_logger().info(f"[NAV] Direct MAVROS GPS navigation to ({goal_lat:.6f}, {goal_lon:.6f})")
            self._start_mavros_gps_navigation(goal_lat, goal_lon, goal_alt)
        else:
            # Current position in nav frame (map by default)
            pose = self._get_robot_pose_in_nav_frame()
            if pose is None:
                self.get_logger().error(
                    f"Cannot get robot pose in nav frame '{self.nav_frame}'"
                )
                return
            curr_x, curr_y = pose

            # Absolute goal in nav frame (using direct offset from origin)
            abs_goal_x = rel_x
            abs_goal_y = rel_y

            self.get_logger().info(
                f"[NAV] Current=({curr_x:.2f}, {curr_y:.2f}), "
                f"GPS offset=({rel_x:.2f}, {rel_y:.2f}), "
                f"Goal=({abs_goal_x:.2f}, {abs_goal_y:.2f})"
            )

            # Start GPS approach via Nav2
            self._start_gps_navigation(curr_x, curr_y, abs_goal_x, abs_goal_y)

    # ─────────────────────────────────────────────────────────────────────
    #  MAVROS GPS navigation
    # ─────────────────────────────────────────────────────────────────────
    def _mavros_state_cb(self, msg: State):
        self.mavros_mode = msg.mode

    def _start_mavros_gps_navigation(self, goal_lat, goal_lon, goal_alt):
        self.current_state = "MAVROS_GPS_NAV_PREP"
        self._publish_color("#r#")
        self._publish_color("#r#")
        self._publish_color("#r#")
        
        self.home_lat_var = self.current_lat
        self.home_lon_var = self.current_lon

        # First clear previous mission
        req_clear = WaypointClear.Request()
        future_clear = self.wp_clear_client.call_async(req_clear)
        # Using lambda to bind variables
        future_clear.add_done_callback(lambda f: self._push_mavros_mission(goal_lat, goal_lon, goal_alt))

    def _push_mavros_mission(self, goal_lat, goal_lon, goal_alt):
        req_push = WaypointPush.Request()
        req_push.start_index = 0
        
        # WP 0 is often interpreted as HOME in MAVLink based FCUs
        wp_home = Waypoint()
        wp_home.frame = Waypoint.FRAME_GLOBAL_REL_ALT
        wp_home.command = 16 # NAV_WAYPOINT
        wp_home.is_current = True
        wp_home.autocontinue = True
        wp_home.x_lat = float(self.home_lat_var)
        wp_home.y_long = float(self.home_lon_var)
        wp_home.z_alt = float(max(self.current_alt, 0.0))

        # WP 1 is the actual goal
        wp_goal = Waypoint()
        wp_goal.frame = Waypoint.FRAME_GLOBAL_REL_ALT
        wp_goal.command = 16
        wp_goal.is_current = False
        wp_goal.autocontinue = True
        wp_goal.x_lat = float(goal_lat)
        wp_goal.y_long = float(goal_lon)
        wp_goal.z_alt = float(max(goal_alt, 0.0))

        req_push.waypoints = [wp_home, wp_goal]
        print(f"Pushing MAVROS mission: HOME({wp_home.x_lat:.6f}, {wp_home.y_long:.6f}, {wp_home.z_alt:.1f}) → ")
        
        future_push = self.wp_push_client.call_async(req_push)
        future_push.add_done_callback(self._on_mission_pushed)

    def _on_mission_pushed(self, future):
        try:
            res = future.result()
            if res.success:
                self.get_logger().info("Mission pushed successfully. Setting to GUIDED mode...")
                mode_req = SetMode.Request()
                mode_req.custom_mode = "GUIDED"
                fut = self.set_mode_client.call_async(mode_req)
                fut.add_done_callback(self._on_guided_set)
            else:
                self.get_logger().error("Failed to push mission")
        except Exception as e:
            self.get_logger().error(f"Push mission exception: {e}")

    def _on_guided_set(self, future):
        self.get_logger().info("Arming vehicle...")
        arm_req = CommandBool.Request()
        arm_req.value = True
        fut = self.arming_client.call_async(arm_req)
        fut.add_done_callback(self._on_armed)

    def _on_armed(self, future):
        self.get_logger().info("Setting AUTO mode...")
        self.current_state = "MAVROS_GPS_NAV"
        mode_req = SetMode.Request()
        mode_req.custom_mode = "AUTO"
        self.set_mode_client.call_async(mode_req)

    def _rcout_cb(self, msg: RCOut):
        # 7. make sure when mavros is in normal mode (no mission) rc out should not interfere with cmd_vel
        if self.current_state != "MAVROS_GPS_NAV":
            return
        
        # Also ensure we are in AUTO mode
        if self.mavros_mode != "AUTO":
            return


        # in rcout [0] is left pwm and [1] is right pwm
        if len(msg.channels) >= 2:
            left_pwm = msg.channels[0]
            right_pwm = msg.channels[1]

            #clamp pwm to 1300 to 1700
            left_pwm = max(1400, min(1600, left_pwm))
            right_pwm = max(1400, min(1600, right_pwm))
            self._publish_color("#r#")
            
            left_norm = (left_pwm - 1500) / 500.0
            right_norm = (right_pwm - 1500) / 500.0
            
            twist = Twist()
            twist.linear.x = (left_norm + right_norm) / 2.0
            twist.angular.z = (right_norm - left_norm) / 2.0
            
            self.cmd_vel_pub.publish(twist)

    def _wp_reached_cb(self, msg: WaypointReached):
        if self.current_state == "MAVROS_GPS_NAV":
            # Reached WP 1 means sequence is done (WP 0 is home)
            if msg.wp_seq >= 1:
                self.get_logger().info("🎉 GPS mavros mission COMPLETE")
                
                # Stop rover by switching back to MANUAL
                mode_req = SetMode.Request()
                mode_req.custom_mode = "MANUAL"
                self.set_mode_client.call_async(mode_req)
                
                # Make sure the vehicle stops moving
                self._stop_cmd_vel()

                if self.mission_type == "aruco":
                    self.get_logger().info("GPS approach via MAVROS complete → starting spiral search")
                    self._start_spiral_search()
                elif self.mission_type in self.OBJECT_MISSION_TYPES:
                    self.get_logger().info("GPS approach via MAVROS complete → starting object search")
                    self._start_object_search()
                else:
                    self.current_state = "IDLE"
                    self._publish_success("gps_mission_complete")
                    self._publish_color("#g#")

    # ─────────────────────────────────────────────────────────────────────
    #  GPS navigation start
    # ─────────────────────────────────────────────────────────────────────
    def _start_gps_navigation(
        self, curr_x: float, curr_y: float, goal_x: float, goal_y: float
    ):
        self.waypoints = self._generate_waypoints(curr_x, curr_y, goal_x, goal_y)
        self.current_wp_index = 0
        self._retry_count = 0
        self.results = []
        self.current_state = "GPS_NAV"
        self._spiral_center = (goal_x, goal_y)

        # Clear aruco state for new mission
        self.spiral_points = []
        self.visited_markers.clear()
        self.targets.clear()
        self._pending_aruco_target = None

        # Publish RED to indicate mission started
        self._publish_color("#r#")
        self._publish_color("#r#")

        self.get_logger().info(
            f"GPS mission: {len(self.waypoints)} segments, "
            f"spacing≈{self.waypoint_spacing:.1f}m"
        )
        for i, (wx, wy) in enumerate(self.waypoints):
            self.get_logger().info(f"  WP {i}: ({wx:.2f}, {wy:.2f})")

        self._navigate_to_current_waypoint()

    # ─────────────────────────────────────────────────────────────────────
    #  Navigate to current GPS waypoint
    # ─────────────────────────────────────────────────────────────────────
    def _navigate_to_current_waypoint(self):
        if self._is_stopped:
            self.get_logger().info("Mission stopped, not navigating.")
            return

        if self._is_paused:
            self.get_logger().info("Mission paused at GPS segment, waiting for resume.")
            return

        if self.current_wp_index >= len(self.waypoints):
            # GPS sequence complete
            if self.mission_type == "aruco":
                self.get_logger().info(
                    "GPS approach complete → starting spiral search"
                )
                self._start_spiral_search()
            elif self.mission_type in self.OBJECT_MISSION_TYPES:
                self.get_logger().info(
                    "GPS approach complete → starting object search"
                )
                self._start_object_search()
            else:
                # GPS-only mission complete
                self.get_logger().info("🎉 GPS mission COMPLETE")
                self.current_state = "IDLE"
                self._print_mission_report()
                self._publish_success("gps_mission_complete")
                self._publish_color("#g#")
            return

        x, y = self.waypoints[self.current_wp_index]
        is_last = self.current_wp_index == len(self.waypoints) - 1
        label = "FINAL GPS GOAL" if is_last else f"WP {self.current_wp_index}"

        self.get_logger().info(
            f"━━━ Navigating to {label} ({x:.2f}, {y:.2f}) "
            f"[{self.current_wp_index + 1}/{len(self.waypoints)}] ━━━"
        )
        self._send_nav_goal(x, y, phase="gps_segment")

    # ─────────────────────────────────────────────────────────────────────
    #  Spiral search (ArUco mode only)
    # ─────────────────────────────────────────────────────────────────────
    def _start_spiral_search(self):
        self.current_state = "SPIRAL_SEARCH"
        self._spiral_start_ns = self.get_clock().now().nanoseconds
        self.get_logger().info("Spiral search: starting basic cmd_vel spiral for 2 minutes...")

    # ─────────────────────────────────────────────────────────────────────
    #  Object search (object_1 / object_2 / object_3)
    # ─────────────────────────────────────────────────────────────────────
    def _start_object_search(self):
        self.current_state = "OBJECT_SEARCH_ROTATE"
        now_ns = self.get_clock().now().nanoseconds
        self._object_search_start_ns = now_ns
        self._object_phase_start_ns = now_ns
        self.object_detected = False
        self._object_spiral_points = []
        self._object_spiral_index = 0
        self._stop_cmd_vel()
        self.get_logger().info(
            f"Object search: rotating in place for "
            f"{OBJECT_SEARCH_ROTATE_TOTAL_SEC:.0f}s before spiral."
        )

    def _start_object_spiral(self):
        if self._spiral_center is None:
            self.get_logger().warn("No spiral center, cannot start object spiral")
            self._publish_prediction(False)
            self.current_state = "IDLE"
            return
        cx, cy = self._spiral_center
        self._object_spiral_points = spiral_search_points(cx, cy, self.spiral_radius)
        self._object_spiral_index = 0
        self._object_phase_start_ns = self.get_clock().now().nanoseconds
        self.current_state = "OBJECT_SEARCH_SPIRAL"
        self.get_logger().info(
            f"Object spiral search: {len(self._object_spiral_points)} points "
            f"around ({cx:.2f}, {cy:.2f})"
        )
        self._navigate_to_current_object_spiral_point()

    def _publish_prediction(self, value: bool):
        msg = Bool()
        msg.data = bool(value)
        self.prediction_pub.publish(msg)

    def _start_object_confirm(self, resume_state: str):
        now_ns = self.get_clock().now().nanoseconds
        self._object_confirm_deadline_ns = now_ns + int(OBJECT_CONFIRM_SEC * 1e9)
        self._object_resume_state = resume_state
        self.current_state = "OBJECT_CONFIRM"
        self._stop_cmd_vel()

    def _stop_cmd_vel(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)

    def _finish_object_search_not_found(self):
        self._stop_cmd_vel()
        self._publish_prediction(False)
        self._publish_color("#r#")
        self.current_state = "IDLE"
        self.get_logger().warn("Object search complete: object not found.")

    def _navigate_to_current_object_spiral_point(self):
        if self._is_stopped or self._is_paused:
            return

        now_ns = self.get_clock().now().nanoseconds
        if self._object_phase_start_ns is not None:
            elapsed = (now_ns - self._object_phase_start_ns) / 1e9
            if elapsed >= OBJECT_SEARCH_SPIRAL_TOTAL_SEC:
                self.get_logger().warn(
                    f"Object spiral timed out after {elapsed:.1f}s."
                )
                self._finish_object_search_not_found()
                return

        if self._object_spiral_index >= len(self._object_spiral_points):
            self._finish_object_search_not_found()
            return

        x, y = self._object_spiral_points[self._object_spiral_index]
        self.get_logger().info(
            f"Object spiral {self._object_spiral_index + 1}/"
            f"{len(self._object_spiral_points)} → ({x:.2f}, {y:.2f})"
        )
        self._send_nav_goal(x, y, phase="object_spiral_segment")

    # ─────────────────────────────────────────────────────────────────────
    #  ArUco detection + navigation
    # ─────────────────────────────────────────────────────────────────────
    def _aruco_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            now_ns = self.get_clock().now().nanoseconds
            updated = 0

            # Format A: {"markers": [{"id": 4, "x":..., "y":..., "z":...}, ...]}
            if isinstance(data, dict) and "markers" in data:
                for m in data["markers"]:
                    mid = int(m["id"])
                    self.targets[mid] = ArucoTarget(
                        marker_id=mid,
                        x=float(m["x"]),
                        y=float(m["y"]),
                        z=float(m.get("z", 0.0)),
                        confidence=float(m.get("confidence", 1.0)),
                        last_seen_ns=now_ns,
                    )
                    updated += 1

            # Format B: {"4": {"x": 7.2, "y": -8.8, "z": -0.02}, ...}
            elif isinstance(data, dict):
                for key, val in data.items():
                    if not isinstance(val, dict):
                        continue
                    if "x" not in val or "y" not in val:
                        continue
                    mid = int(key)
                    self.targets[mid] = ArucoTarget(
                        marker_id=mid,
                        x=float(val["x"]),
                        y=float(val["y"]),
                        z=float(val.get("z", 0.0)),
                        confidence=float(val.get("confidence", 1.0)),
                        last_seen_ns=now_ns,
                    )
                    updated += 1

            if updated > 0:
                ids_sorted = sorted(self.targets.keys())
                self.get_logger().info(
                    f"[ARUCO] Updated {updated} marker(s). "
                    f"Total={len(self.targets)} IDs={ids_sorted}"
                )
                # Try to interrupt current navigation for ArUco
                if self.current_state in (
                    "SPIRAL_SEARCH", "SPIRAL_COMPLETE_WAIT"
                ):
                    target = self._select_unvisited_target()
                    if target is not None:
                        print(f"new test aruco {target.x}, {target.y}")
                        self._interrupt_to_aruco(target)
                elif self.current_state == "ARUCO_NAV":
                    if self._pending_aruco_target is not None:
                        # Keep latest target estimate, but do NOT cancel/reissue nav goals.
                        mid = self._pending_aruco_target.marker_id
                        if mid in self.targets:
                            self._pending_aruco_target = self.targets[mid]

        except Exception as e:
            self.get_logger().warn(f"Invalid aruco payload: {e}")

    def _target_age_seconds(self, target: ArucoTarget) -> float:
        now_ns = self.get_clock().now().nanoseconds
        return (now_ns - target.last_seen_ns) / 1e9

    def _select_unvisited_target(self) -> Optional[ArucoTarget]:
        if not self.targets:
            return None

        fresh = [
            t
            for t in self.targets.values()
            if t.marker_id not in self.visited_markers
            and self._target_age_seconds(t) <= self.target_timeout_sec
        ]
        if not fresh:
            return None
        # Since t.x and t.y are relative, distance from camera is simply math.hypot(t.x, t.y)
        return min(fresh, key=lambda t: math.hypot(t.x, t.y))

    def _interrupt_to_aruco(self, target: ArucoTarget):
        if target.marker_id in self.visited_markers:
            return

        self._pending_aruco_target = target
        self.get_logger().info(
            f"ArUco {target.marker_id} found — starting basic PID docking"
        )
        self._start_aruco_navigation(target)

    def _start_aruco_navigation(self, target: ArucoTarget):
        self._pending_aruco_target = target
        self.current_state = "ARUCO_NAV"
        self.get_logger().info(f"Steering towards ArUco {target.marker_id} via basic PID...")

    # ─────────────────────────────────────────────────────────────────────
    #  Tick — periodic ArUco check
    # ─────────────────────────────────────────────────────────────────────
    def _tick(self):
        if self.current_goal_handle is not None or self._nav_goal_pending:
            return
            
        if self.current_state == "SPIRAL_SEARCH":
            target = self._select_unvisited_target()
            if target is not None:
                self._interrupt_to_aruco(target)
                return

            now_ns = self.get_clock().now().nanoseconds
            elapsed_sec = (now_ns - getattr(self, '_spiral_start_ns', now_ns)) / 1e9
            if elapsed_sec > 120.0:
                self.get_logger().info("Spiral search timeout (120s).")
                self._stop_cmd_vel()
                self.current_state = "IDLE"
                self._publish_failure("aruco_not_found")
                return

            twist = Twist()
            twist.linear.x = 0.2
            # Decrease angular velocity gradually so the spiral grows outwards
            twist.angular.z = max(0.1, 0.6 - (0.5 * (elapsed_sec / 120.0)))
            self.cmd_vel_pub.publish(twist)
            return

        if self.current_state == "ARUCO_NAV":
            target = self._pending_aruco_target
            if target is None:
                self._start_spiral_search()
                return

            # Check if target is stale (we lost sight of it)
            if self._target_age_seconds(target) > 5.0:
                self.get_logger().warn(f"Lost sight of ArUco {target.marker_id} for > 5s. Resuming search.")
                self._pending_aruco_target = None
                self._start_spiral_search()
                return

            # Target x and y are purely relative (FLU frame: x-forward, y-left)
            dx = target.x
            dy = target.y
            dist = math.hypot(dx, dy)
            
            if dist < 1.0: # Stop at 1 meter distance
                self.get_logger().info(f"✅ ArUco {target.marker_id} reached (dist={dist:.2f}m)!")
                self._stop_cmd_vel()
                self._pending_aruco_target = None
                self.current_state = "IDLE"
                if target.marker_id not in self.visited_markers:
                    self.visited_markers.append(target.marker_id)
                self._publish_color("#g#")
                self._publish_success("aruco_reached")
                return

            twist = Twist()
            
            # Simple threshold-based steering using y offset (meters left/right)
            # dy > 0 means ArUco is to the LEFT, so rotate LEFT (+)
            # dy < 0 means ArUco is to the RIGHT, so rotate RIGHT (-)
            
            if abs(dy) > 0.4:
                # Too far left/right -> rotate in place
                # 0.4 rad/s is roughly equivalent to 1700/1300 PWM (assuming 1.0 = 2000/1000)
                twist.linear.x = 0.0
                twist.angular.z = 0.4 if dy > 0 else -0.4
            elif abs(dy) > 0.1:
                # Slightly off -> move forward with a soft curve
                twist.linear.x = 0.2
                twist.angular.z = 0.2 if dy > 0 else -0.2
            else:
                # Deadband -> just move straight forward
                twist.linear.x = 0.25
                twist.angular.z = 0.0

            self.cmd_vel_pub.publish(twist)
            return

        # Object search: rotate in place first, then spiral if not detected.
        if self.current_state == "OBJECT_SEARCH_ROTATE":
            now_ns = self.get_clock().now().nanoseconds
            if self.object_detected:
                self._start_object_confirm("OBJECT_SEARCH_ROTATE")
                return

            if self._object_search_start_ns is not None:
                elapsed = (now_ns - self._object_search_start_ns) / 1e9
                if elapsed >= OBJECT_SEARCH_ROTATE_TOTAL_SEC:
                    self._stop_cmd_vel()
                    self._start_object_spiral()
                    return

            twist = Twist()
            twist.angular.z = 0.6
            self.cmd_vel_pub.publish(twist)
            return

        if self.current_state == "OBJECT_SEARCH_SPIRAL":
            now_ns = self.get_clock().now().nanoseconds
            if self.object_detected:
                self._start_object_confirm("OBJECT_SEARCH_SPIRAL")
                return

            if self._object_phase_start_ns is not None:
                elapsed = (now_ns - self._object_phase_start_ns) / 1e9
                if elapsed >= OBJECT_SEARCH_SPIRAL_TOTAL_SEC:
                    self.get_logger().warn(
                        f"Object spiral timed out after {elapsed:.1f}s."
                    )
                    self._finish_object_search_not_found()
                    return

            if self.current_goal_handle is None:
                self._navigate_to_current_object_spiral_point()
                return
            return

        if self.current_state == "OBJECT_CONFIRM":
            now_ns = self.get_clock().now().nanoseconds
            if self._object_confirm_deadline_ns is None:
                self.current_state = "IDLE"
                return

            if now_ns < self._object_confirm_deadline_ns:
                return

            if self.object_detected:
                self._publish_prediction(True)
                self._publish_color("#g#")
                self.current_state = "IDLE"
                return

            # False after confirm window → resume previous phase
            resume_state = self._object_resume_state or "OBJECT_SEARCH_ROTATE"
            self._object_confirm_deadline_ns = None
            self._object_resume_state = None
            self.current_state = resume_state
            if resume_state == "OBJECT_SEARCH_ROTATE":
                self._object_phase_start_ns = now_ns
            return

    # ─────────────────────────────────────────────────────────────────────
    #  Nav2 goal sending
    # ─────────────────────────────────────────────────────────────────────
    def _send_nav_goal(self, x: float, y: float, phase: str):
        if not self._nav2_ready:
            self.get_logger().warn("Nav2 not ready, cannot send goal")
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.nav_frame
        # Stamp=0 asks TF for latest available transform, reducing extrapolation failures.
        goal.pose.header.stamp.sec = 0
        goal.pose.header.stamp.nanosec = 0
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.w = 1.0

        self._active_goal_phase = phase
        self._nav_goal_pending = True
        future = self._action_client.send_goal_async(
            goal, feedback_callback=self._feedback_cb
        )
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        self._nav_goal_pending = False
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error(
                f"Goal REJECTED during phase={self._active_goal_phase}"
            )
            wp_x, wp_y = 0.0, 0.0
            if self._active_goal_phase == "gps_segment" and self.current_wp_index < len(self.waypoints):
                wp_x, wp_y = self.waypoints[self.current_wp_index]
                self.results.append((self.current_wp_index, wp_x, wp_y, "REJECTED"))
                self._advance_gps_waypoint()
            elif self._active_goal_phase == "spiral_segment":
                self.current_wp_index += 1
                self._navigate_to_current_spiral_point()
            elif self._active_goal_phase == "object_spiral_segment":
                self._object_spiral_index += 1
                self._navigate_to_current_object_spiral_point()
            else:
                self._publish_failure("goal_rejected")
                self.current_state = "IDLE"
            self._active_goal_phase = None
            self.current_goal_handle = None
            return

        self.current_goal_handle = handle
        self.get_logger().info(f"Goal accepted (phase={self._active_goal_phase})")
        handle.get_result_async().add_done_callback(self._goal_result_cb)

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        dist = fb.distance_remaining
        self.get_logger().info(
            f"  distance remaining: {dist:.2f} m",
            throttle_duration_sec=5.0,
        )

    def _goal_result_cb(self, future):
        status = future.result().status
        phase = self._active_goal_phase
        status_str = STATUS_NAMES.get(status, f"UNKNOWN({status})")

        self.current_goal_handle = None
        self._active_goal_phase = None

        # Handle stop/pause
        if self._is_stopped:
            self.get_logger().info("Navigation stopped, ignoring result.")
            self.current_state = "IDLE"
            return

        if self._is_paused:
            self.get_logger().info(
                f"Goal finished while paused (status={status_str}). "
                "Waiting for resume."
            )
            return

        # ── GPS segment result ───────────────────────────────────────────
        if phase == "gps_segment":
            wp_x, wp_y = self.waypoints[self.current_wp_index] if self.current_wp_index < len(self.waypoints) else (0.0, 0.0)

            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(
                    f"✅ GPS WP {self.current_wp_index} SUCCEEDED"
                )
                self.results.append(
                    (self.current_wp_index, wp_x, wp_y, "SUCCEEDED")
                )
                self._retry_count = 0
                self._advance_gps_waypoint()

            elif status == GoalStatus.STATUS_ABORTED:
                if self._retry_count < self.max_retries:
                    self._retry_count += 1
                    self.get_logger().warn(
                        f"⚠️ GPS WP {self.current_wp_index} ABORTED — "
                        f"retry {self._retry_count}/{self.max_retries}"
                    )
                    self._navigate_to_current_waypoint()
                else:
                    self.get_logger().error(
                        f"❌ GPS WP {self.current_wp_index} FAILED — SKIPPING"
                    )
                    self.results.append(
                        (self.current_wp_index, wp_x, wp_y, "SKIPPED (ABORTED)")
                    )
                    self._retry_count = 0
                    self._advance_gps_waypoint()

            elif status == GoalStatus.STATUS_CANCELED:
                self.get_logger().warn(
                    f"⛔ GPS WP {self.current_wp_index} CANCELED"
                )
                self.results.append(
                    (self.current_wp_index, wp_x, wp_y, "CANCELED")
                )

            else:
                self.get_logger().warn(
                    f"⚠️ GPS WP {self.current_wp_index} status={status_str} — SKIPPING"
                )
                self.results.append(
                    (self.current_wp_index, wp_x, wp_y, f"SKIPPED ({status_str})")
                )
                self._retry_count = 0
                self._advance_gps_waypoint()

        # ── Spiral segment result ────────────────────────────────────────
        elif phase == "spiral_segment":
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(
                    f"Spiral point {self.current_wp_index + 1} reached."
                )
                target = self._select_unvisited_target()
                if target is not None:
                    self.get_logger().info(
                        f"ArUco {target.marker_id} selected after spiral point completion."
                    )
                    self._start_aruco_navigation(target)
                else:
                    self.current_wp_index += 1
                    self._navigate_to_current_spiral_point()
            elif status == GoalStatus.STATUS_CANCELED:
                self.get_logger().warn("Spiral point canceled")
            else:
                self.get_logger().warn(
                    f"Spiral point failed (status={status_str}), skipping"
                )
                target = self._select_unvisited_target()
                if target is not None:
                    self.get_logger().info(
                        f"ArUco {target.marker_id} selected after spiral point failure."
                    )
                    self._start_aruco_navigation(target)
                else:
                    self.current_wp_index += 1
                    self._navigate_to_current_spiral_point()

        # ── Object spiral result ────────────────────────────────────────
        elif phase == "object_spiral_segment":
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(
                    f"Object spiral point {self._object_spiral_index + 1} reached."
                )
            else:
                self.get_logger().warn(
                    f"Object spiral point failed (status={status_str}), skipping"
                )

            self._object_spiral_index += 1
            if self.object_detected:
                self._start_object_confirm("OBJECT_SEARCH_SPIRAL")
            else:
                self._navigate_to_current_object_spiral_point()

    def _advance_gps_waypoint(self):
        self.current_wp_index += 1
        self._navigate_to_current_waypoint()

    # ─────────────────────────────────────────────────────────────────────
    #  Control callback (start / stop / pause)
    # ─────────────────────────────────────────────────────────────────────
    def _control_cb(self, msg: String):
        command = msg.data.strip().lower()
        self.get_logger().info(f"[CONTROL] Command: '{command}'")

        if command == "stop":
            self._handle_stop()
        elif command == "pause":
            self._handle_pause()
        elif command == "start":
            self._handle_resume()
        else:
            self.get_logger().warn(f"[CONTROL] Unknown: '{command}'")

    def _handle_stop(self):
        self.get_logger().info("[STOP] Stopping navigation...")
        self._is_stopped = True
        self._is_paused = False
        self._paused_goal_data = None

        # Clear mission state
        self.waypoints = []
        self.spiral_points = []
        self.current_wp_index = 0
        self._retry_count = 0
        self._nav_goal_pending = False
        self._pending_aruco_target = None
        self.object_detected = False
        self._object_search_start_ns = None
        self._object_phase_start_ns = None
        self._object_spiral_points = []
        self._object_spiral_index = 0
        self._object_confirm_deadline_ns = None
        self._object_resume_state = None
        self._stop_cmd_vel()

        if self.current_goal_handle is not None:
            self.current_goal_handle.cancel_goal_async()
            self.current_goal_handle = None

        if self.current_state.startswith("MAVROS_GPS_NAV"):
            # clear mavros state
            self.get_logger().info("Stopping MAVROS mission")
            mode_req = SetMode.Request()
            mode_req.custom_mode = "MANUAL"
            self.set_mode_client.call_async(mode_req)
            req_clear = WaypointClear.Request()
            self.wp_clear_client.call_async(req_clear)

        self.current_state = "IDLE"
        self.get_logger().info("[STOP] All navigation state cleared.")

    def _handle_pause(self):
        if self._is_paused:
            self.get_logger().warn("[PAUSE] Already paused")
            return

        if self.current_goal_handle is None and self.current_state == "IDLE":
            self.get_logger().warn("[PAUSE] No active mission to pause")
            return

        self.get_logger().info("[PAUSE] Pausing navigation...")
        self._is_paused = True
        self._is_stopped = False

        if self.current_goal_handle is not None:
            self.current_goal_handle.cancel_goal_async()
            self.current_goal_handle = None
        self._nav_goal_pending = False

    def _handle_resume(self):
        self._is_stopped = False

        if not self._is_paused:
            self.get_logger().info("[START] Ready for new waypoint.")
            return

        self._is_paused = False
        self.get_logger().info("[RESUME] Resuming navigation...")

        # Resume based on current state
        if self.current_state == "GPS_NAV" and self.current_wp_index < len(self.waypoints):
            self._navigate_to_current_waypoint()
        elif self.current_state == "SPIRAL_SEARCH" and self.current_wp_index < len(self.spiral_points):
            self._navigate_to_current_spiral_point()
        elif self.current_state == "OBJECT_SEARCH_ROTATE":
            self.get_logger().info("[RESUME] Object rotation search resumed.")
        elif self.current_state == "OBJECT_SEARCH_SPIRAL":
            self._navigate_to_current_object_spiral_point()
        elif self.current_state in ("SPIRAL_COMPLETE_WAIT",):
            self.get_logger().info("[RESUME] Spiral complete, checking for ArUco targets...")
        else:
            self.get_logger().info("[RESUME] No active navigation to resume, waiting for new waypoint.")

    # ─────────────────────────────────────────────────────────────────────
    #  Mission report
    # ─────────────────────────────────────────────────────────────────────
    def _print_mission_report(self):
        self.get_logger().info("")
        self.get_logger().info("╔══════════════════════════════════════╗")
        self.get_logger().info("║         MISSION REPORT               ║")
        self.get_logger().info("╠══════════════════════════════════════╣")

        succeeded = 0
        failed = 0

        for idx, x, y, status_str in self.results:
            if status_str == "SUCCEEDED":
                icon = "✅"
                succeeded += 1
            else:
                icon = "❌"
                failed += 1
            self.get_logger().info(
                f"║ {icon} WP {idx:2d} ({x:7.2f}, {y:7.2f}) {status_str}"
            )

        self.get_logger().info("╠══════════════════════════════════════╣")
        self.get_logger().info(
            f"║  ✅ {succeeded} succeeded  ❌ {failed} failed"
        )
        self.get_logger().info("╚══════════════════════════════════════╝")

        if failed == 0:
            self.get_logger().info("🎉 All waypoints reached!")
        else:
            self.get_logger().warn(
                f"⚠️ {failed} waypoint(s) were skipped/failed."
            )

    # ─────────────────────────────────────────────────────────────────────
    #  Status publishers
    # ─────────────────────────────────────────────────────────────────────
    def _publish_success(self, message: str):
        msg = String()
        msg.data = json.dumps({
            "status": "success",
            "message": message,
            "state": self.current_state,
            "mission_type": self.mission_type,
            "timestamp": datetime.now().isoformat(),
        })
        self.status_pub.publish(msg)

    def _publish_failure(self, reason: str):
        msg = String()
        msg.data = json.dumps({
            "status": "failed",
            "reason": reason,
            "state": self.current_state,
            "mission_type": self.mission_type,
            "timestamp": datetime.now().isoformat(),
        })
        self.status_pub.publish(msg)
        self._publish_color("#r#")
        self.get_logger().error(f"FAILURE: {reason}")

    def _publish_color(self, color: str):
        """Publish mission phase color status.
        Colors:
                    - #r#: Active mission, intermediate waypoints
                    - #g#: Final destination or successful completion
        """
        color_map = {
            "red": "#r#",
            "r": "#r#",
            "green": "#g#",
            "g": "#g#",
                        "#r#": "#r#",
                        "#g#": "#g#",
        }
        color_code = color_map.get(color, color)
        msg = String()
        msg.data = color_code
        self.color_pub.publish(msg)
        self.get_logger().info(f"[COLOR] {color_code.upper()}")

    # ─────────────────────────────────────────────────────────────────────
    #  UDP telemetry
    # ─────────────────────────────────────────────────────────────────────
    def _send_gps_udp(self):
        if not self.gps_received:
            return
        try:
            payload = json.dumps({
                "latitude": self.current_lat,
                "longitude": self.current_lon,
                "altitude": self.current_alt,
                "state": self.current_state,
                "mission_type": self.mission_type,
                "timestamp": datetime.now().isoformat(),
            })
            self.udp_socket.sendto(
                payload.encode(), (UDP_DASHBOARD_IP, UDP_DASHBOARD_PORT)
            )
        except Exception as e:
            self.get_logger().error(f"UDP send error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = TestBackendNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt — shutting down.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()