#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import cv2.aruco
import time
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from collections import deque


class ArucoHoming(Node):
    def __init__(self):
        super().__init__('aruco_homing')

        # Camera setup
        self.bridge = CvBridge()
        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=10)
        )

        # Movement control
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.move_timer = self.create_timer(0.1, self.movement_control)

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.target_ids = [1, 2, 3, 4]
        self.current_target = 1
        self.detected_ids = set()

        # State control
        self.state = "SEARCH"  # "SEARCH", "PAUSE", "RETURN_HOME", "VISUAL_HOMING", or "STOP"
        self.pause_start_time = 0
        self.pause_duration = 2.0

        # Movement recording
        self.movement_history = deque(maxlen=1000)
        self.current_movement_start = time.time()
        self.current_linear = 0.0
        self.current_angular = 0.0

        # Homing control
        self.return_movements = []
        self.return_index = 0
        self.homing_start_time = 0
        self.marker_positions = {}  # Stores last seen positions of markers

        # Camera parameters (adjust based on your camera)
        self.camera_center_x = 320  # Assuming 640x480 resolution
        self.marker_size = 0.1  # Physical size of markers in meters
        self.focal_length = 500  # Approximate focal length in pixels

        self.get_logger().info("Aruco Homing Ready - Will use visual homing after path retrace")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect markers
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                current_frame_ids = set(ids.flatten())
                self.detected_ids.update(current_frame_ids)

                # Store marker positions for visual homing
                for i, marker_id in enumerate(ids.flatten()):
                    center_x = np.mean(corners[i][0][:, 0])
                    self.marker_positions[marker_id] = center_x

                # Check if all target IDs are detected
                if all(id in self.detected_ids for id in self.target_ids):
                    if self.state == "SEARCH":
                        self._record_current_movement()
                        self.return_movements = list(reversed(self.movement_history))
                        self.return_index = 0
                        self.state = "RETURN_HOME"
                        self.get_logger().info("ALL MARKERS DETECTED! Beginning return.")
                    elif self.state == "VISUAL_HOMING":
                        self._visual_homing(corners, ids)

                # Normal marker detection during search
                if self.state == "SEARCH":
                    for marker_id in ids.flatten():
                        if marker_id == self.current_target:
                            self.get_logger().info(f"Detected target {marker_id}")
                            self._record_current_movement()
                            self.state = "PAUSE"
                            self.pause_start_time = time.time()
                            self.switch_target()

            # Display detection status
            status_text = f"Detected: {sorted(self.detected_ids)}/{len(self.target_ids)} | State: {self.state}"
            cv2.putText(cv_image, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("ArUco Detection", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Camera error: {str(e)}")

    def _visual_homing(self, corners, ids):
        """Adjust position based on marker positions in camera view"""
        if all(id in self.marker_positions for id in self.target_ids):
            # Calculate average offset from center
            offsets = []
            for marker_id in self.target_ids:
                offsets.append(self.marker_positions[marker_id] - self.camera_center_x)

            avg_offset = np.mean(offsets)

            # If markers are centered, we're home
            if abs(avg_offset) < 20:  # Within 20 pixels of center
                self.state = "STOP"
                self.get_logger().info("Visual homing complete! Stopped at start position.")
            else:
                # Adjust position to center markers
                twist = Twist()
                twist.linear.x = 0.05  # Slow forward/backward
                twist.angular.z = -0.1 * np.sign(avg_offset)  # Turn toward center
                self.cmd_vel_pub.publish(twist)
        else:
            # Not all markers visible - small search pattern
            twist = Twist()
            twist.angular.z = 0.2
            self.cmd_vel_pub.publish(twist)

    def _record_current_movement(self):
        duration = time.time() - self.current_movement_start
        if duration > 0.1:  # Only record meaningful movements
            movement_data = (self.current_linear, self.current_angular, duration)
            self.movement_history.append(movement_data)
        self.current_movement_start = time.time()

    def switch_target(self):
        self.current_target = self.target_ids[
            (self.target_ids.index(self.current_target) + 1) % len(self.target_ids)
            ]
        self.get_logger().info(f"Next target: Marker {self.current_target}")

    def movement_control(self):
        twist = Twist()

        if self.state == "STOP":
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        elif self.state == "RETURN_HOME":
            if self.return_index < len(self.return_movements):
                linear, angular, duration = self.return_movements[self.return_index]
                elapsed = time.time() - self.current_movement_start

                twist.linear.x = -linear
                twist.angular.z = -angular

                if elapsed >= duration:
                    self.return_index += 1
                    self.current_movement_start = time.time()
                    if self.return_index >= len(self.return_movements):
                        self.state = "VISUAL_HOMING"
                        self.homing_start_time = time.time()
                        self.get_logger().info("Path retrace complete. Starting visual homing...")
            else:
                self.state = "VISUAL_HOMING"

        elif self.state == "VISUAL_HOMING":
            # Handled in image_callback via _visual_homing()
            pass

        elif self.state == "PAUSE":
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.current_linear = 0.0
            self.current_angular = 0.0

            if time.time() - self.pause_start_time >= self.pause_duration:
                self.state = "SEARCH"
                self.current_movement_start = time.time()
                self.get_logger().info("Resuming search")

        elif self.state == "SEARCH":
            twist.linear.x = 0.15
            twist.angular.z = 0.3
            self.current_linear = twist.linear.x
            self.current_angular = twist.angular.z

        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoHoming()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
