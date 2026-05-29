#!/usr/bin/env python3
"""
Unified Navigation Backend — test_backend.py

Handles GPS-only, GPS+ArUco, and Object mission types.
All GPS navigation is handled via MAVROS mission upload.

GPS mode:
- Uploads GPS waypoint to MAVROS mission planner
- Rover navigates autonomously in AUTO mode
- Publishes result to /mission/reached_waypoint

ArUco mode:
- MAVROS GPS approach to target area
- After GPS waypoint reached → Phase 1: Rotate-Stop-Rotate Search
- If not found → Phase 2: Spiral Search
- Threshold-based docking to detected ArUco markers
- Publishes result to /mission/reached_waypoint

Control:
- /mission/waypoint  (JSON with starting/ending GPS, type)
- /mission/control   (start / stop / pause)
"""

import json
import math
import socket
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix

from mavros_msgs.msg import Waypoint, RCOut, WaypointReached, State
from mavros_msgs.srv import WaypointClear, WaypointPush, SetMode, CommandBool


# ── Constants ───────────────────────────────────────────────────────────────
DEFAULT_ARUCO_TOPIC = "/aruco_relative_positions"

# ArUco specific constants
ARUCO_STOP_DISTANCE_M = 2.8        # stop when this close
ARUCO_MAX_DISTANCE_M = 8.0         # ignore targets further than this
DEFAULT_TARGET_TIMEOUT_SEC = 5.0   # lose-sight timeout

ARUCO_ROTATE_TIMEOUT_SEC = 30.0    # initial rotate-in-place total time
ARUCO_ROTATE_MOVE_SEC = 0.3        # interval: how long to actively rotate
ARUCO_ROTATE_STOP_SEC = 0.3        # interval: how long to stop and scan clearly

ARUCO_SPIRAL_TOTAL_SEC = 420.0     # total spiral search budget (7 mins)

# Object specific constants
OBJECT_SEARCH_ROTATE_TOTAL_SEC = 60.0
OBJECT_SEARCH_SPIRAL_TOTAL_SEC = 120.0
OBJECT_CONFIRM_SEC = 3.0

OBJECT_ROTATE_MOVE_SEC = 0.3        # interval: how long to actively rotate for object
OBJECT_ROTATE_STOP_SEC = 0.3        # interval: how long to stop and scan clearly for object


UDP_DASHBOARD_IP = "192.168.2.15"
UDP_DASHBOARD_PORT = 5005


# ── ArUco target dataclass ──────────────────────────────────────────────────
@dataclass
class ArucoTarget:
    marker_id: int
    distance: float
    offset: float
    confidence: float
    last_seen_ns: int


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
        self.target_timeout_sec = DEFAULT_TARGET_TIMEOUT_SEC

        # ── MAVROS Clients & State ──────────────────────────────────────
        self.wp_clear_client = self.create_client(WaypointClear, "/mavros/mission/clear")
        self.wp_push_client = self.create_client(WaypointPush, "/mavros/mission/push")
        self.set_mode_client = self.create_client(SetMode, "/mavros/set_mode")
        self.arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.home_lat_var = None
        self.home_lon_var = None
        self.mavros_mode = "MANUAL"

        # ── Mission state ────────────────────────────────────────────────
        self.current_state = "IDLE"  
        self.mission_type = "gps"   
        
        # ArUco state
        self.targets: Dict[int, ArucoTarget] = {}
        self.visited_markers: List[int] = []
        self._pending_aruco_target: Optional[ArucoTarget] = None
        self._rotate_start_ns: Optional[int] = None
        self._spiral_start_ns: Optional[int] = None

        # Pause / Stop
        self._is_paused = False
        self._is_stopped = False
        self._paused_goal_data: Optional[dict] = None

        # Object detection search
        self.object_detected = False
        self._object_search_start_ns: Optional[int] = None
        self._object_phase_start_ns: Optional[int] = None
        self._object_confirm_deadline_ns: Optional[int] = None
        self._object_resume_state: Optional[str] = None

        # ── GPS + Heading ────────────────────────────────────────────────
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0
        self.gps_received = False
        self.current_heading_deg = 0.0
        self.heading_received = False

        # ── Subscriptions ────────────────────────────────────────────────
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
        self.create_timer(0.1, self._tick)  # Run tick at 10Hz for smooth control
        self.create_timer(1.0, self._send_gps_udp)

        self.get_logger().info("═" * 50)
        self.get_logger().info("  TEST BACKEND — MAVROS Navigator Online")
        self.get_logger().info(f"  ArUco: {DEFAULT_ARUCO_TOPIC}")
        self.get_logger().info("═" * 50)

    # ─────────────────────────────────────────────────────────────────────
    #  Sensor callbacks
    # ─────────────────────────────────────────────────────────────────────

    def _gps_cb(self, msg: NavSatFix):
        self.current_lat = msg.latitude
        self.current_lon = msg.longitude
        self.current_alt = msg.altitude
        if not self.gps_received:
            self.gps_received = True
            self.get_logger().info(
                f"[GPS] Connected: lat={self.current_lat:.6f}, lon={self.current_lon:.6f}"
            )

    def _heading_cb(self, msg: Float64):
        self.current_heading_deg = msg.data % 360.0
        if not self.heading_received:
            self.heading_received = True
            self.get_logger().info(f"[HEADING] Connected: {self.current_heading_deg:.1f}°")

    def _object_cb(self, msg: Bool):
        self.object_detected = bool(msg.data)

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

        if self.current_state.startswith("MAVROS_GPS_NAV"):
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

        # Clear aruco state for new mission
        self.visited_markers.clear()
        self.targets.clear()
        self._pending_aruco_target = None
        self._rotate_start_ns = None
        self._spiral_start_ns = None

        self.get_logger().info(f"[NAV] Direct MAVROS GPS navigation to ({goal_lat:.6f}, {goal_lon:.6f})")
        self._start_mavros_gps_navigation(goal_lat, goal_lon, goal_alt)

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
        if self.current_state != "MAVROS_GPS_NAV":
            return
        
        if self.mavros_mode != "AUTO":
            return

        if len(msg.channels) >= 2:
            left_pwm = msg.channels[0]
            right_pwm = msg.channels[1]

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
                    self.get_logger().info("GPS approach via MAVROS complete → starting ArUco search")
                    self._start_aruco_rotate_search()
                elif self.mission_type in self.OBJECT_MISSION_TYPES:
                    self.get_logger().info("GPS approach via MAVROS complete → starting object search")
                    self._start_object_search()
                else:
                    self.current_state = "IDLE"
                    self._publish_success("gps_mission_complete")
                    self._publish_color("#g#")

    # ─────────────────────────────────────────────────────────────────────
    #  ArUco Search Initiators
    # ─────────────────────────────────────────────────────────────────────
    def _start_aruco_rotate_search(self):
        self.current_state = "ARUCO_ROTATE_SEARCH"
        self._rotate_start_ns = self.get_clock().now().nanoseconds
        self.get_logger().info(f"Phase 1: Rotate-Stop-Rotate search for {ARUCO_ROTATE_TIMEOUT_SEC}s...")

    def _start_aruco_spiral_search(self):
        self.current_state = "ARUCO_SPIRAL_SEARCH"
        self._spiral_start_ns = self.get_clock().now().nanoseconds
        self.get_logger().info(f"Phase 2: Spiral searching for {ARUCO_SPIRAL_TOTAL_SEC/60:.0f} minutes...")

    # ─────────────────────────────────────────────────────────────────────
    #  Object search (object_1 / object_2 / object_3)
    # ─────────────────────────────────────────────────────────────────────
    def _start_object_search(self):
        self.current_state = "OBJECT_SEARCH_ROTATE"
        now_ns = self.get_clock().now().nanoseconds
        self._object_search_start_ns = now_ns
        self._object_phase_start_ns = now_ns
        self.object_detected = False
        self._stop_cmd_vel()
        self.get_logger().info(
            f"Object search: Rotate-Stop-Rotate search for "
            f"{OBJECT_SEARCH_ROTATE_TOTAL_SEC:.0f}s before spiral."
        )

    def _start_object_spiral(self):
        self._object_phase_start_ns = self.get_clock().now().nanoseconds
        self.current_state = "OBJECT_SEARCH_SPIRAL"
        self.get_logger().info("Object spiral search: starting cmd_vel spiral")

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

    # ─────────────────────────────────────────────────────────────────────
    #  ArUco detection + navigation
    # ─────────────────────────────────────────────────────────────────────
    def _aruco_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            now_ns = self.get_clock().now().nanoseconds
            updated = 0

            # Format A: {"markers": [{"id": 4, "distance":..., "offset":...}, ...]}
            if isinstance(data, dict) and "markers" in data:
                for m in data["markers"]:
                    mid = int(m["id"])
                    self.targets[mid] = ArucoTarget(
                        marker_id=mid,
                        distance=float(m["distance"]),
                        offset=float(m["offset"]),
                        confidence=float(m.get("confidence", 1.0)),
                        last_seen_ns=now_ns,
                    )
                    updated += 1

            # Format B: {"4": {"distance": 5.2, "offset": -0.8}, ...}
            elif isinstance(data, dict):
                for key, val in data.items():
                    if not isinstance(val, dict):
                        continue
                    if "distance" not in val or "offset" not in val:
                        continue
                    mid = int(key)
                    self.targets[mid] = ArucoTarget(
                        marker_id=mid,
                        distance=float(val["distance"]),
                        offset=float(val["offset"]),
                        confidence=float(val.get("confidence", 1.0)),
                        last_seen_ns=now_ns,
                    )
                    updated += 1

            if updated > 0:
                ids_sorted = sorted(self.targets.keys())
                self.get_logger().debug(
                    f"[ARUCO] Updated {updated} marker(s). "
                    f"Total={len(self.targets)} IDs={ids_sorted}"
                )
                
                # Try to interrupt current navigation for ArUco
                if self.current_state in ("ARUCO_ROTATE_SEARCH", "ARUCO_SPIRAL_SEARCH"):
                    target = self._select_unvisited_target()
                    if target is not None:
                        self._interrupt_to_aruco(target)
                elif self.current_state == "ARUCO_NAV":
                    if self._pending_aruco_target is not None:
                        # Keep latest target estimate
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

        # Filter: Not visited + Seen recently + Distance <= 8.0m
        fresh = [
            t
            for t in self.targets.values()
            if t.marker_id not in self.visited_markers
            and self._target_age_seconds(t) <= self.target_timeout_sec
            and t.distance <= ARUCO_MAX_DISTANCE_M
        ]
        if not fresh:
            return None
        return min(fresh, key=lambda t: t.distance)

    def _interrupt_to_aruco(self, target: ArucoTarget):
        if target.marker_id in self.visited_markers:
            return

        self._pending_aruco_target = target
        self.get_logger().info(
            f"🎯 ArUco {target.marker_id} detected at {target.distance:.2f}m! Switching to FOLLOW mode."
        )
        self.current_state = "ARUCO_NAV"

    # ─────────────────────────────────────────────────────────────────────
    #  Tick — periodic ArUco / Search check
    # ─────────────────────────────────────────────────────────────────────
    def _tick(self):
        
        # ── ArUco Rotate Search ──────────────────────────────────────────
        if self.current_state == "ARUCO_ROTATE_SEARCH":
            target = self._select_unvisited_target()
            if target is not None:
                self._interrupt_to_aruco(target)
                return

            now_ns = self.get_clock().now().nanoseconds
            elapsed = (now_ns - getattr(self, '_rotate_start_ns', now_ns)) / 1e9
            if elapsed > ARUCO_ROTATE_TIMEOUT_SEC:
                self.get_logger().info(f"Rotation search finished ({ARUCO_ROTATE_TIMEOUT_SEC}s). Transitioning to SPIRAL SEARCH.")
                self._start_aruco_spiral_search()
                return

            # Rotate-Stop-Rotate cycle logic
            cycle_duration = ARUCO_ROTATE_MOVE_SEC + ARUCO_ROTATE_STOP_SEC
            time_in_cycle = elapsed % cycle_duration

            twist = Twist()
            if time_in_cycle < ARUCO_ROTATE_MOVE_SEC:
                # Active rotation phase
                twist.linear.x = 0.0
                twist.angular.z = 0.5  # Positive value for leftward rotation
            else:
                # Stopped scanning phase to prevent camera blur
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            self.cmd_vel_pub.publish(twist)
            return

        # ── ArUco Spiral Search ──────────────────────────────────────────
        if self.current_state == "ARUCO_SPIRAL_SEARCH":
            target = self._select_unvisited_target()
            if target is not None:
                self._interrupt_to_aruco(target)
                return

            now_ns = self.get_clock().now().nanoseconds
            elapsed_sec = (now_ns - getattr(self, '_spiral_start_ns', now_ns)) / 1e9
            if elapsed_sec > ARUCO_SPIRAL_TOTAL_SEC:
                self.get_logger().warn(f"Spiral search timed out after {ARUCO_SPIRAL_TOTAL_SEC}s. No ArUco found.")
                self._stop_cmd_vel()
                self.current_state = "IDLE"
                self._publish_failure("aruco_not_found")
                return

            # Spiral cmd_vel: explicit radius expansion logic
            twist = Twist()
            twist.linear.x = 0.12  # Slower, calmer forward speed
            
            # The radius of the spiral starts at 0.7m and grows smoothly over time
            current_radius = 0.7 + (3.8 * (elapsed_sec / ARUCO_SPIRAL_TOTAL_SEC))
            
            # Kinematic relation: angular_velocity = linear_velocity / radius
            twist.angular.z = twist.linear.x / current_radius
            
            self.cmd_vel_pub.publish(twist)
            return

        # ── ArUco Follow Navigation ──────────────────────────────────────
        if self.current_state == "ARUCO_NAV":
            target = self._pending_aruco_target
            if target is None:
                self._start_aruco_spiral_search()
                return

            # Check if target is stale (we lost sight of it)
            if self._target_age_seconds(target) > self.target_timeout_sec:
                self.get_logger().warn(f"Lost sight of ArUco {target.marker_id} for > {self.target_timeout_sec}s. Resuming spiral.")
                self._pending_aruco_target = None
                self._start_aruco_spiral_search()
                return

            dist = target.distance
            dy = target.offset

            # 1. Safety Limit: > 8m
            if dist > ARUCO_MAX_DISTANCE_M:
                self.get_logger().info(
                    f"⚠️ ArUco {target.marker_id} is {dist:.2f}m away (>{ARUCO_MAX_DISTANCE_M}m). Ignoring and resuming spiral...",
                    throttle_duration_sec=2.0
                )
                self._pending_aruco_target = None
                self._start_aruco_spiral_search()
                return

            # 2. Stop within 2.8m
            if dist < ARUCO_STOP_DISTANCE_M:
                self.get_logger().info(f"✅ ArUco {target.marker_id} reached (dist={dist:.2f}m)!")
                self._stop_cmd_vel()
                self._pending_aruco_target = None
                self.current_state = "IDLE"
                if target.marker_id not in self.visited_markers:
                    self.visited_markers.append(target.marker_id)
                self._publish_color("#g#")
                self._publish_success("aruco_reached")
                return

            # 3. Offset-based steering logic
            twist = Twist()
            if dy > 2.0:
                # Marker is far left -> Hard turn left
                twist.linear.x = 0.0
                twist.angular.z = 0.3     
            elif dy < -2.0:
                # Marker is far right -> Hard turn right
                twist.linear.x = 0.0
                twist.angular.z = -0.3    
            elif -0.8 <= dy <= 0.8:
                # Offset in middle: just move forward
                twist.linear.x = 0.2
                twist.angular.z = 0.0
            else:
                # Otherwise: map and turn (moderate steering while moving forward)
                twist.linear.x = 0.15
                twist.angular.z = 0.15 * dy

            self.cmd_vel_pub.publish(twist)
            return

        # ── Object Search Rotation ───────────────────────────────────────
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

            # Rotate-Stop-Rotate cycle logic matching ArUco parameters
            cycle_duration = OBJECT_ROTATE_MOVE_SEC + OBJECT_ROTATE_STOP_SEC
            time_in_cycle = elapsed % cycle_duration

            twist = Twist()
            if time_in_cycle < OBJECT_ROTATE_MOVE_SEC:
                # Active rotation phase
                twist.linear.x = 0.0
                twist.angular.z = 0.5  # Positive value for leftward rotation
            else:
                # Stopped scanning phase to prevent camera blur
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            self.cmd_vel_pub.publish(twist)
            return

        # ── Object Search Spiral ─────────────────────────────────────────
        if self.current_state == "OBJECT_SEARCH_SPIRAL":
            now_ns = self.get_clock().now().nanoseconds
            if self.object_detected:
                self._start_object_confirm("OBJECT_SEARCH_SPIRAL")
                return

            if self._object_phase_start_ns is not None:
                elapsed = (now_ns - self._object_phase_start_ns) / 1e9
                if elapsed >= OBJECT_SEARCH_SPIRAL_TOTAL_SEC:
                    self.get_logger().warn(f"Object spiral timed out after {elapsed:.1f}s.")
                    self._finish_object_search_not_found()
                    return

            # Explicit radius expansion logic matching ArUco parameters
            twist = Twist()
            twist.linear.x = 0.12  # Slower, calmer forward speed
            
            # Match ArUco spiral radius scaling over the object search budget (120s)
            current_radius = 0.7 + (3.8 * (elapsed / OBJECT_SEARCH_SPIRAL_TOTAL_SEC))
            twist.angular.z = twist.linear.x / current_radius
            
            self.cmd_vel_pub.publish(twist)
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
        self._pending_aruco_target = None
        self._rotate_start_ns = None
        self._spiral_start_ns = None
        self.object_detected = False
        self._object_search_start_ns = None
        self._object_phase_start_ns = None
        self._object_confirm_deadline_ns = None
        self._object_resume_state = None
        self._stop_cmd_vel()

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

        if self.current_state == "IDLE":
            self.get_logger().warn("[PAUSE] No active mission to pause")
            return

        self.get_logger().info("[PAUSE] Pausing navigation...")
        self._is_paused = True
        self._is_stopped = False

    def _handle_resume(self):
        self._is_stopped = False

        if not self._is_paused:
            self.get_logger().info("[START] Ready for new waypoint.")
            return

        self._is_paused = False
        self.get_logger().info("[RESUME] Resuming navigation...")

        # Resume based on current state
        if self.current_state == "OBJECT_SEARCH_ROTATE":
            self.get_logger().info("[RESUME] Object rotation search resumed.")
        elif self.current_state == "OBJECT_SEARCH_SPIRAL":
            self.get_logger().info("[RESUME] Object spiral search resumed.")
        elif self.current_state == "ARUCO_ROTATE_SEARCH":
            self.get_logger().info("[RESUME] ArUco rotation search resumed.")
        elif self.current_state == "ARUCO_SPIRAL_SEARCH":
            self.get_logger().info("[RESUME] ArUco spiral search resumed.")
        else:
            self.get_logger().info("[RESUME] Resuming, current state: " + self.current_state)

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
        """Publish mission phase color status."""
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