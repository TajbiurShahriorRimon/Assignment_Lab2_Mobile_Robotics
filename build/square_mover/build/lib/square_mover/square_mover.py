#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class MarkerMapper(Node):
    def __init__(self):
        super().__init__('marker_mapper')
        
        # Camera setup
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        self.bridge = CvBridge()
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            qos_profile
        )
        
        # Movement control
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # ArUco setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.detector_params = aruco.DetectorParameters_create()
        self.target_ids = [1, 2, 3, 4]
        self.marker_positions = {}  # Stores {id: (x, y)} coordinates
        self.centroid = None  # Stores (x, y) of centroid
        
        # Camera calibration (replace with your camera's values!)
        self.camera_matrix = np.array([
            [500, 0, 320],  # fx, 0, cx
            [0, 500, 240],  # 0, fy, cy
            [0, 0, 1]
        ])
        self.dist_coeffs = np.zeros((4, 1))  # No distortion
        
        # State variables
        self.state = "SCAN"
        self.scan_start_time = self.get_clock().now().seconds_nanoseconds()[0]
        self.rotation_duration = 12.0  # Time for 360° at 0.5 rad/s
        
        self.get_logger().info("Marker Mapper Ready!")

    def calculate_centroid(self):
        """Calculate the center point of all detected markers"""
        if len(self.marker_positions) == len(self.target_ids):
            x_coords = [pos[0] for pos in self.marker_positions.values()]
            y_coords = [pos[1] for pos in self.marker_positions.values()]
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            self.centroid = (centroid_x, centroid_y)
            self.get_logger().info(f"Centroid calculated at: ({centroid_x:.2f}, {centroid_y:.2f})")
            return True
        return False

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.detector_params)
            
            if ids is not None:
                # Draw detected markers with IDs
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                
                # Add ID labels
                for i, marker_id in enumerate(ids.flatten()):
                    center = corners[i][0].mean(axis=0)
                    cv2.putText(cv_image, f"ID:{marker_id}", 
                               (int(center[0]), int(center[1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                # Original pose estimation
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, 0.1, self.camera_matrix, self.dist_coeffs
                )
                
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id in self.target_ids:
                        x, y = tvecs[i][0][0], tvecs[i][0][1]
                        self.marker_positions[marker_id] = (x, y)
                        self.get_logger().info(f"Marker {marker_id}: x={x:.2f}m, y={y:.2f}m")
            
            # Calculate and display centroid when all markers are found
            if len(self.marker_positions) == len(self.target_ids) and self.centroid is None:
                self.calculate_centroid()
            
            # Display state and centroid information
            display_text = f"State: {self.state}"
            if self.centroid:
                cx, cy = self.centroid
                display_text += f"\nCentroid: ({cx:.2f}, {cy:.2f})"
                cv2.putText(cv_image, f"Centroid: ({cx:.2f}, {cy:.2f})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            
            cv2.putText(cv_image, display_text, (10,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
            cv2.imshow("Marker View", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Camera error: {str(e)}")

    def control_loop(self):
        twist = Twist()
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        
        if self.state == "SCAN":
            # Rotate at 0.5 rad/s (~30°/s)
            twist.angular.z = 0.3
            
            # Check if full rotation is complete
            if current_time - self.scan_start_time >= self.rotation_duration:
                if len(self.marker_positions) < len(self.target_ids):
                    self.state = "MOVE_FORWARD"
                    self.move_start_time = current_time
                    self.get_logger().warn(f"Missing markers! Detected: {list(self.marker_positions.keys())}")
                else:
                    self.get_logger().info(f"All markers mapped! Positions: {self.marker_positions}")
                    self.calculate_centroid()  # Ensure centroid is calculated
                    self.marker_positions.clear()
                    self.centroid = None  # Reset for next scan
                    self.scan_start_time = current_time
        
        elif self.state == "MOVE_FORWARD":
            # Move forward for 2 seconds (~0.4m at 0.2 m/s)
            twist.linear.x = 0.2
            if current_time - self.move_start_time >= 2.0:
                self.state = "SCAN"
                self.scan_start_time = current_time
        
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = MarkerMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
