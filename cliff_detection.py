#!/usr/bin/env python3

from math import ceil, isfinite
import socket
import threading

import rclpy
from rclpy.qos import ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Header, String



def get_parameter(node, name, default_value):
    node.declare_parameter(name, default_value)
    return node.get_parameter(name).value


def read_parameters(node):
    params = {
        "input_topic": get_parameter(node, "input_topic", "/ground"),
        "danger_topic": get_parameter(node, "danger_topic", "/cliff_danger"),
        "status_topic": get_parameter(node, "status_topic", "/cliff_status"),
        "cliff_cloud_topic": get_parameter(node, "cliff_cloud_topic", "/cliff_cloud"),
        "cliff_cloud_frame_id": get_parameter(node, "cliff_cloud_frame_id", ""),
        "start_distance_m": float(get_parameter(node, "start_distance_m", 0.8)),
        "front_distance_m": float(get_parameter(node, "front_distance_m", 6.0)),
        "front_width_m": float(get_parameter(node, "front_width_m", 1.5)),
        "bin_size_m": float(get_parameter(node, "bin_size_m", 0.25)),
        "max_sample_error_m": float(get_parameter(node, "max_sample_error_m", 0.0)),
        "lidar_upside_down": bool(get_parameter(node, "lidar_upside_down", False)),
        "allowed_drop_m": float(get_parameter(node, "allowed_drop_m", 0.30)),
        "cliff_ratio_threshold": float(
            get_parameter(node, "cliff_ratio_threshold", 2.0)
        ),
        "min_cliff_distance_m": float(
            get_parameter(node, "min_cliff_distance_m", 0.25)
        ),
        "min_ground_z_m": float(get_parameter(node, "min_ground_z_m", -2.5)),
        "max_ground_z_m": float(get_parameter(node, "max_ground_z_m", -0.5)),
        "min_valid_bin_ratio": float(get_parameter(node, "min_valid_bin_ratio", 0.60)),
        "unknown_is_dangerous": bool(get_parameter(node, "unknown_is_dangerous", False)),
        "log_each_cloud": bool(get_parameter(node, "log_each_cloud", False)),
        "debug_print_samples": bool(get_parameter(node, "debug_print_samples", False)),
        "fake_obstacle_depth_m": float(get_parameter(node, "fake_obstacle_depth_m", 0.20)),
        "fake_obstacle_height_m": float(
            get_parameter(node, "fake_obstacle_height_m", 0.60)
        ),
        "fake_obstacle_z_min_m": float(
            get_parameter(node, "fake_obstacle_z_min_m", -1.0)
        ),
        "fake_obstacle_point_step_m": float(
            get_parameter(node, "fake_obstacle_point_step_m", 0.10)
        ),
        "fake_obstacle_margin_m": float(
            get_parameter(node, "fake_obstacle_margin_m", 0.0)
        ),
    }

    params["front_distance_m"] = max(params["front_distance_m"], 0.1)
    params["front_width_m"] = max(params["front_width_m"], 0.1)
    params["bin_size_m"] = max(params["bin_size_m"], 0.05)
    if params["max_sample_error_m"] <= 0.0:
        params["max_sample_error_m"] = params["bin_size_m"] * 0.5
    params["allowed_drop_m"] = max(params["allowed_drop_m"], 0.0)
    params["cliff_ratio_threshold"] = max(params["cliff_ratio_threshold"], 1.0)
    params["min_cliff_distance_m"] = max(
        params["min_cliff_distance_m"],
        params["bin_size_m"],
    )
    params["min_ground_z_m"] = min(params["min_ground_z_m"], params["max_ground_z_m"])
    params["min_valid_bin_ratio"] = min(max(params["min_valid_bin_ratio"], 0.0), 1.0)
    params["fake_obstacle_depth_m"] = max(params["fake_obstacle_depth_m"], 0.05)
    params["fake_obstacle_point_step_m"] = max(
        params["fake_obstacle_point_step_m"],
        0.02,
    )
    params["fake_obstacle_height_m"] = max(
        params["fake_obstacle_height_m"],
        params["fake_obstacle_z_min_m"],
    )
    params["fake_obstacle_margin_m"] = max(params["fake_obstacle_margin_m"], 0.0)

    return params


def point_to_xyz(point):
    try:
        return float(point[0]), float(point[1]), float(point[2])
    except (TypeError, ValueError, IndexError):
        return float(point["x"]), float(point["y"]), float(point["z"])


def level_point(x, y, z, upside_down):
    if upside_down:
        return x, -y, -z
    return x, y, z


def build_sampled_z(cloud_msg, params):
    num_samples = int(ceil(params["front_distance_m"] / params["bin_size_m"]))
    half_width = params["front_width_m"] * 0.5
    sampled_z = [None] * num_samples
    best_x_error = [float("inf")] * num_samples
    best_y_error = [float("inf")] * num_samples

    points = point_cloud2.read_points(
        cloud_msg,
        field_names=("x", "y", "z"),
        skip_nans=True,
    )

    for point in points:
        x, y, z = point_to_xyz(point)

        if not (isfinite(x) and isfinite(y) and isfinite(z)):
            continue

        x, y, z = level_point(
            x,
            y,
            z,
            params["lidar_upside_down"],
        )

        if x <= params["start_distance_m"]:
            continue
        if x > params["start_distance_m"] + params["front_distance_m"]:
            continue
        if abs(y) > half_width:
            continue
        if z < params["min_ground_z_m"] or z > params["max_ground_z_m"]:
            continue

        relative_x = x - params["start_distance_m"]
        sample_number = int((relative_x / params["bin_size_m"]) + 0.5)
        if sample_number < 1 or sample_number > num_samples:
            continue

        target_x = params["start_distance_m"] + sample_number * params["bin_size_m"]
        x_error = abs(x - target_x)
        if x_error > params["max_sample_error_m"]:
            continue

        sample_index = sample_number - 1
        y_error = abs(y)
        current_z = sampled_z[sample_index]

        is_lower_z = current_z is None or z < current_z
        is_same_z_but_better_x = (
            current_z is not None
            and z == current_z
            and x_error < best_x_error[sample_index]
        )
        is_same_z_x_but_centered = (
            current_z is not None
            and z == current_z
            and x_error == best_x_error[sample_index]
            and y_error < best_y_error[sample_index]
        )

        if is_lower_z or is_same_z_but_better_x or is_same_z_x_but_centered:
            sampled_z[sample_index] = z
            best_x_error[sample_index] = x_error
            best_y_error[sample_index] = y_error

    return sampled_z


def sample_distance_m(sample_index, params):
    return params["start_distance_m"] + (sample_index + 1) * params["bin_size_m"]


def format_ratio(ratio):
    if ratio == float("inf"):
        return "inf"
    return f"{ratio:.2f}"


def distance_ratio_check(sampled_z, params):
    num_samples = len(sampled_z)
    valid_samples =[
        (index, z_value)
        for index, z_value in enumerate(sampled_z)
        if z_value is not None
    ]
    valid_count = len(valid_samples)
    min_valid_bins = int(ceil(num_samples * params["min_valid_bin_ratio"]))
    min_valid_bins = min(max(min_valid_bins, 1), num_samples)
    min_cliff_samples = int(
        ceil(params["min_cliff_distance_m"] / params["bin_size_m"])
    )
    min_cliff_samples = max(min_cliff_samples, 1)

    if valid_count == 0:
        return {
            "state": "UNKNOWN",
            "danger": params["unknown_is_dangerous"],
            "text": "UNKNOWN: no usable points in the front area",
        }

    if valid_count < min_valid_bins:
        return {
            "state": "UNKNOWN",
            "danger": params["unknown_is_dangerous"],
            "text": (
                "UNKNOWN: not enough sampled data in the front area "
                f"({valid_count}/{num_samples} samples valid)"
            ),
        }

    for base_position, (base_index, base_z) in enumerate(valid_samples):
        current_distance = -base_z
        max_safe_distance = current_distance + params["allowed_drop_m"]
        deeper_count = 0
        within_count = 0
        first_deeper_index = None
        last_deeper_index = None

        for compare_index, compare_z in valid_samples[base_position + 1 :]:
            if compare_z > base_z:
                continue
            
            compare_distance = -compare_z
            if compare_distance > max_safe_distance:
                deeper_count += 1
                if first_deeper_index is None:
                    first_deeper_index = compare_index
                last_deeper_index = compare_index
            else:
                within_count += 1
                if within_count >= min_cliff_samples * 2:
                    break

        if deeper_count < min_cliff_samples:
            continue

        if within_count == 0:
            deeper_ratio = float("inf")
        else:
            deeper_ratio = deeper_count / within_count

        if deeper_ratio < params["cliff_ratio_threshold"]:
            continue

        # Changed from first_deeper_index to base_index.
        # This makes the obstacle spawn at the very edge of the safe ground, 
        # before the drop happens, pulling the obstacle physically closer to the robot.
        cliff_start_m = sample_distance_m(base_index, params)
        cliff_end_m = min(
            sample_distance_m(last_deeper_index, params),
            params["start_distance_m"] + params["front_distance_m"],
        )
        cliff_end_m = min(
            max(cliff_end_m, cliff_start_m + params["bin_size_m"]),
            params["start_distance_m"] + params["front_distance_m"],
        )

        return {
            "state": "DANGEROUS",
            "danger": True,
            "window_start_m": cliff_start_m,
            "window_end_m": cliff_end_m,
            "text": (
                "DANGEROUS: cliff distance ratio "
                f"{format_ratio(deeper_ratio)} "
                f"({deeper_count} more, {within_count} less) "
                f"after {sample_distance_m(base_index, params):.2f} m "
                f"(threshold {params['allowed_drop_m']:.2f} m)"
            ),
        }

    return {
        "state": "SAFE",
        "danger": False,
        "text": (
            "SAFE: no dangerous cliff found within "
            f"{params['front_distance_m']:.2f} m"
        ),
    }


def publish_result(result, danger_pub, status_pub):
    danger_msg = Bool()
    danger_msg.data = bool(result["danger"])
    danger_pub.publish(danger_msg)

    status_msg = String()
    status_msg.data = result["text"]
    status_pub.publish(status_msg)


def make_cliff_cloud_header(cloud_msg, params):
    header = Header()
    header.stamp = cloud_msg.header.stamp
    header.frame_id = params["cliff_cloud_frame_id"] or cloud_msg.header.frame_id
    return header


def build_fake_obstacle_points(result, params):
    if not result["danger"] or "window_start_m" not in result:
        return[]

    # Shift obstacle even earlier using the new fake_obstacle_margin_m param
    end_x = result["window_start_m"] - params["fake_obstacle_margin_m"]
    start_x = max(params["start_distance_m"], end_x - params["fake_obstacle_depth_m"])
    
    half_width = params["front_width_m"] * 0.5
    step = params["fake_obstacle_point_step_m"]
    points =[]

    x = start_x
    while x <= end_x + 1e-6:
        y = -half_width
        while y <= half_width + 1e-6:
            z = params["fake_obstacle_z_min_m"]
            while z <= params["fake_obstacle_height_m"] + 1e-6:
                if params.get("lidar_upside_down", False):
                    points.append((x, -y, -z))
                else:
                    points.append((x, y, z))
                z += step
            y += step
        x += step

    return points


def publish_cliff_cloud(cloud_msg, params, result, cliff_cloud_pub):
    header = make_cliff_cloud_header(cloud_msg, params)
    points = build_fake_obstacle_points(result, params)
    cliff_cloud_msg = point_cloud2.create_cloud_xyz32(header, points)
    cliff_cloud_pub.publish(cliff_cloud_msg)


def cloud_callback(cloud_msg, node, params, danger_pub, status_pub, cliff_cloud_pub, state):
    if not state.get("cliff_detection_enabled", False):
        return
    sampled_z = build_sampled_z(cloud_msg, params)
    if params["debug_print_samples"]:
        print(sampled_z)
    result = distance_ratio_check(sampled_z, params)
    publish_result(result, danger_pub, status_pub)
    publish_cliff_cloud(cloud_msg, params, result, cliff_cloud_pub)

    should_log = params["log_each_cloud"] or result["state"] != state["last_state"]
    if should_log:
        if result["danger"]:
            node.get_logger().warn(result["text"])
        else:
            node.get_logger().info(result["text"])
        state["last_state"] = result["state"]


def udp_listener(state, node):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", 5555))
    except Exception as e:
        node.get_logger().error(f"Failed to bind UDP socket on port 5555: {e}")
        return

    node.get_logger().info("UDP listener started on port 5555")
    sock.settimeout(1.0)
    while rclpy.ok():
        try:
            data, addr = sock.recvfrom(1024)
            msg = data.decode("utf-8").strip().lower()
            if "true" in msg:
                if not state["cliff_detection_enabled"]:
                    node.get_logger().warn("Cliff detection ENABLED via UDP")
                state["cliff_detection_enabled"] = True
            elif "false" in msg:
                if state["cliff_detection_enabled"]:
                    node.get_logger().warn("Cliff detection DISABLED via UDP")
                state["cliff_detection_enabled"] = False
        except socket.timeout:
            continue
        except Exception as e:
            node.get_logger().error(f"Error in UDP listener: {e}")
            break
    sock.close()


def main():
    rclpy.init()
    node = rclpy.create_node("simple_cliff_detector")
    params = read_parameters(node)

    danger_pub = node.create_publisher(Bool, params["danger_topic"], 10)
    status_pub = node.create_publisher(String, params["status_topic"], 10)
   
    cliff_qos = rclpy.qos.QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
    )
    cliff_cloud_pub = node.create_publisher(
        PointCloud2,
        params["cliff_cloud_topic"],
        cliff_qos,
    )
    state = {
        "last_state": None,
        "cliff_detection_enabled": False,
    }

    # Start the UDP listener daemon thread
    thread = threading.Thread(target=udp_listener, args=(state, node), daemon=True)
    thread.start()

    node.create_subscription(
        PointCloud2,
        params["input_topic"],
        lambda msg: cloud_callback(
            msg,
            node,
            params,
            danger_pub,
            status_pub,
            cliff_cloud_pub,
            state,
        ),
        qos_profile_sensor_data,
    )

    node.get_logger().info(
        "Cliff detector started: "
        f"topic={params['input_topic']}, "
        f"front={params['front_distance_m']:.2f} m, "
        f"sample_step={params['bin_size_m']:.2f} m, "
        f"sample_error={params['max_sample_error_m']:.2f} m, "
        f"ground_z={params['min_ground_z_m']:.2f}..{params['max_ground_z_m']:.2f} m, "
        f"allowed_drop={params['allowed_drop_m']:.2f} m, "
        f"ratio_threshold={params['cliff_ratio_threshold']:.2f}, "
        f"min_cliff_distance={params['min_cliff_distance_m']:.2f} m, "
        f"cliff_cloud_topic={params['cliff_cloud_topic']}"
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
