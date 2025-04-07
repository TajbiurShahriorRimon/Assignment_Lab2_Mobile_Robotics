#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import cv2.aruco
import time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class ArucoPauser(Node):
    def __init__(self):
        super().__init__('aruco_pauser')
        
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
        
        # State control
        self.state = "SEARCH"  # "SEARCH" or "PAUSE"
        self.pause_start_time = 0
        self.pause_duration = 2.0  # seconds
        
        self.get_logger().info("Aruco Pauser Ready - Searching for markers")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect markers
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                
                for marker_id in ids.flatten():
                    if marker_id == self.current_target:
                        self.get_logger().info(f"Detected target {marker_id}")
                        self.state = "PAUSE"
                        self.pause_start_time = time.time()
                        self.switch_target()
            
            cv2.imshow("ArUco Detection", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Camera error: {str(e)}")

    def switch_target(self):
        self.current_target = self.target_ids[
            (self.target_ids.index(self.current_target) + 1) % len(self.target_ids)
        ]
        self.get_logger().info(f"Next target: Marker {self.current_target}")

    def movement_control(self):
        twist = Twist()
        
        if self.state == "PAUSE":
            # Stop for 2 seconds
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            
            # Resume searching after pause duration
            if time.time() - self.pause_start_time >= self.pause_duration:
                self.state = "SEARCH"
                self.get_logger().info("Resuming search")
                
        elif self.state == "SEARCH":
            # Normal search movement
            twist.linear.x = 0.15
            twist.angular.z = 0.3
        
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPauser()
    
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
