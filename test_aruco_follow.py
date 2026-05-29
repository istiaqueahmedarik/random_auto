#!/usr/bin/env python3
"""
test_aruco_follow.py — Standalone ArUco search + follow + stop test.

Usage:
    ros2 run <your_pkg> test_aruco_follow
    # or simply:
    python3 test_aruco_follow.py

Flow:
    1. Starts a spiral search via cmd_vel
    2. Listens for ArUco detections on /aruco_relative_positions
    3. When a marker is found → switches to follow based on new logic
    4. Stops when distance < 2 m
    5. Ctrl+C to quit at any time
"""

import json
import math
from typing import Dict, Optional
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


# ── Tuning constants ────────────────────────────────────────────────────────
STOP_DISTANCE_M = 2.0           # stop when this close
TARGET_TIMEOUT_SEC = 5.0        # lose-sight timeout
SPIRAL_TIMEOUT_SEC = 120.0      # total spiral search budget

@dataclass
class ArucoTarget:
    marker_id: int
    distance: float
    offset: float
    last_seen_ns: int


class ArucoFollowTest(Node):
    def __init__(self):
        super().__init__("aruco_follow_test")

        # State: SPIRAL_SEARCH | ARUCO_FOLLOW | DONE
        self.state = "SPIRAL_SEARCH"
        self.targets: Dict[int, ArucoTarget] = {}
        self._pending_target: Optional[ArucoTarget] = None
        self._spiral_start_ns = self.get_clock().now().nanoseconds

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Subscribers
        self.create_subscription(
            String, "/aruco_relative_positions", self._aruco_cb, 10
        )

        # Tick at 10 Hz
        self.create_timer(0.1, self._tick)

        self.get_logger().info("=" * 50)
        self.get_logger().info("  ArUco Follow Test — STARTED")
        self.get_logger().info("  Spiral searching... waiting for ArUco marker")
        self.get_logger().info("=" * 50)

    # ── ArUco detection callback ────────────────────────────────────────
    def _aruco_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            now_ns = self.get_clock().now().nanoseconds

            # Format A: {"markers": [{"id": 4, "x":..., "y":..., "z":...}, ...]}
            if isinstance(data, dict) and "markers" in data:
                for m in data["markers"]:
                    mid = int(m["id"])
                    self.targets[mid] = ArucoTarget(
                        marker_id=mid,
                        distance=float(m["distance"]),
                        offset=float(m["offset"]),
                        last_seen_ns=now_ns,
                    )

            # Format B: {"4": {"distance": 5.2, "offset": -0.8}, ...}
            elif isinstance(data, dict):
                for key, val in data.items():
                    if not isinstance(val, dict) or "distance" not in val or "offset" not in val:
                        continue
                    mid = int(key)
                    self.targets[mid] = ArucoTarget(
                        marker_id=mid,
                        distance=float(val["distance"]),
                        offset=float(val["offset"]),
                        last_seen_ns=now_ns,
                    )

        except Exception as e:
            self.get_logger().warn(f"Bad aruco payload: {e}")

    # ── Helpers ─────────────────────────────────────────────────────────
    def _target_age_sec(self, t: ArucoTarget) -> float:
        return (self.get_clock().now().nanoseconds - t.last_seen_ns) / 1e9

    def _closest_fresh_target(self) -> Optional[ArucoTarget]:
        fresh = [
            t for t in self.targets.values()
            if self._target_age_sec(t) <= TARGET_TIMEOUT_SEC
        ]
        if not fresh:
            return None
        return min(fresh, key=lambda t: t.distance)

    def _stop(self):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)

    # ── Main loop ───────────────────────────────────────────────────────
    def _tick(self):
        if self.state == "DONE":
            return

        # ── SPIRAL SEARCH ───────────────────────────────────────────────
        if self.state == "SPIRAL_SEARCH":
            # Check for a target first
            target = self._closest_fresh_target()
            if target is not None:
                self._pending_target = target
                self.state = "ARUCO_FOLLOW"
                self.get_logger().info(
                    f"🎯 ArUco {target.marker_id} detected! Switching to FOLLOW mode."
                )
                return

            # Check timeout
            now_ns = self.get_clock().now().nanoseconds
            elapsed = (now_ns - self._spiral_start_ns) / 1e9
            if elapsed > SPIRAL_TIMEOUT_SEC:
                self.get_logger().warn("Spiral search timed out. No ArUco found.")
                self._stop()
                self.state = "DONE"
                return

            # Spiral cmd_vel: explicit radius expansion logic
            twist = Twist()
            twist.linear.x = 0.12  # Slower, calmer forward speed
            
            # The radius of the spiral starts at 0.2m and grows smoothly to 2.0m over time
            current_radius = 0.2 + (1.8 * (elapsed / SPIRAL_TIMEOUT_SEC))
            
            # Kinematic relation: angular_velocity = linear_velocity / radius
            twist.angular.z = twist.linear.x / current_radius
            
            self.cmd_vel_pub.publish(twist)
            return

        # ── ARUCO FOLLOW ────────────────────────────────────────────────
        if self.state == "ARUCO_FOLLOW":
            target = self._pending_target
            if target is None:
                self.get_logger().warn("No pending target, resuming spiral.")
                self.state = "SPIRAL_SEARCH"
                self._spiral_start_ns = self.get_clock().now().nanoseconds
                return

            # Update with latest observation of same marker
            mid = target.marker_id
            if mid in self.targets:
                self._pending_target = self.targets[mid]
                target = self._pending_target

            # Lost sight?
            if self._target_age_sec(target) > TARGET_TIMEOUT_SEC:
                self.get_logger().warn(
                    f"Lost ArUco {mid} for >{TARGET_TIMEOUT_SEC}s. Resuming spiral."
                )
                self._pending_target = None
                self._stop()
                self.state = "SPIRAL_SEARCH"
                self._spiral_start_ns = self.get_clock().now().nanoseconds
                return

            dist = target.distance
            dy = target.offset

            self.get_logger().info(
                f"[FOLLOW] id={mid}  offset={dy:.2f}  dist={dist:.2f}m",
                throttle_duration_sec=0.5,
            )

            # ── 1. STOP if distance < 2m ────────────────────────────────
            if dist < STOP_DISTANCE_M:
                self._stop()
                self.get_logger().info(
                    f"✅ ArUco {mid} reached! dist={dist:.2f}m  — STOPPING."
                )
                self.state = "DONE"
                return

            # ── 2. Offset-based steering logic ──────────────────────────
            twist = Twist()

            # Fix: Positive offset means marker is on the LEFT. Negative means RIGHT.
            if dy > 2.0:
                # Marker is far left -> Hard turn left
                twist.linear.x = 0.0
                twist.angular.z = 0.3     # Positive value for left turn
            elif dy < -2.0:
                # Marker is far right -> Hard turn right
                twist.linear.x = 0.0
                twist.angular.z = -0.3    # Negative value for right turn
            elif -0.8 <= dy <= 0.8:
                # Offset in middle: just move forward
                twist.linear.x = 0.2
                twist.angular.z = 0.0
            else:
                # Otherwise: map and turn (moderate steering while moving forward)
                # Now using +0.15 so positive offset (left) maps to positive angular Z (left turn)
                twist.linear.x = 0.15
                twist.angular.z = 0.15 * dy

            self.cmd_vel_pub.publish(twist)
            return


def main(args=None):
    rclpy.init(args=args)
    node = ArucoFollowTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C — shutting down.")
        # Make sure rover stops
        node._stop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()