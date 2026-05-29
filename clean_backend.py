import re

with open('final_backend_safe.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if line.startswith('    def _odom_cb'):
        skip = True
    elif line.startswith('    def _odom_to_nav_frame'):
        skip = True
    elif line.startswith('    def _get_robot_pose'):
        skip = True
    elif line.startswith('    def _check_nav2_ready'):
        skip = True
    elif line.startswith('    def _generate_waypoints'):
        skip = True
    elif line.startswith('    def _start_gps_navigation'):
        skip = True
    elif line.startswith('    def _navigate_to_current_waypoint'):
        skip = True
    elif line.startswith('    def _navigate_to_current_object_spiral_point'):
        skip = True
    elif line.startswith('    def _send_nav_goal'):
        skip = True
    elif line.startswith('    def _goal_response_cb'):
        skip = True
    elif line.startswith('    def _feedback_cb'):
        skip = True
    elif line.startswith('    def _goal_result_cb'):
        skip = True
    elif line.startswith('    def _advance_gps_waypoint'):
        skip = True
    elif line.startswith('    def _print_mission_report'):
        skip = True

    # When to stop skipping: when we hit a top-level method we want to keep
    # Wait, better check if line starts with '    def ' and is NOT one of the above
    if skip and line.startswith('    def '):
        if not any(line.startswith(f'    def {fn}') for fn in [
            '_odom_cb', '_odom_to_nav_frame', '_get_robot_pose',
            '_check_nav2_ready', '_generate_waypoints', '_start_gps_navigation',
            '_navigate_to_current_waypoint', '_navigate_to_current_object_spiral_point',
            '_send_nav_goal', '_goal_response_cb', '_feedback_cb', '_goal_result_cb',
            '_advance_gps_waypoint', '_print_mission_report'
        ]):
            skip = False

    # Also skip the specific comment blocks
    if line.strip() == '# ── Robot pose from ZED odom':
        pass # we might need to skip these section headers too, but let's just keep them or delete manually
    
    if not skip:
        new_lines.append(line)

with open('final_backend_safe.py', 'w') as f:
    f.writelines(new_lines)
