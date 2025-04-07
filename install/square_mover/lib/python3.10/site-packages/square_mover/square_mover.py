#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray, Pose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import cv2.aruco
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import time

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        
        # Camera setup
        self.bridge = CvBridge()
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            qos_profile=camera_qos
        )
        
        # ArUco detection
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        
        # For movement control (separate from detection)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.movement_timer = None
        
        # State variables
        self.last_marker_time = 0
        self.stationary_duration = 2.0  # Seconds to pause for detection

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect markers
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            
            if ids is not None:
                self.last_marker_time = time.time()
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                
                for marker_id in ids.flatten():
                    self.get_logger().info(f"Detected Marker ID: {marker_id}")
                    
                    # Start movement after detection
                    if self.movement_timer is None:
                        self.start_movement()
            
            cv2.imshow("ArUco Detection", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Detection error: {str(e)}")

    def start_movement(self):
        """Start moving only after first detection"""
        self.get_logger().info("Starting controlled movement")
        self.movement_timer = self.create_timer(0.1, self.controlled_move)

    def controlled_move(self):
        """Movement that pauses when markers might be visible"""
        twist = Twist()
        
        # Stop if we recently saw a marker
        if time.time() - self.last_marker_time < self.stationary_duration:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        else:
            # Slow, controlled movement
            twist.linear.x = 0.1
            twist.angular.z = 0.2
        
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
