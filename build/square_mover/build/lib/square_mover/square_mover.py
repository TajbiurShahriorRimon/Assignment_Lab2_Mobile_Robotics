#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import cv2
import time

class SquareMover(Node):
    def __init__(self):
        super().__init__('square_mover')
        
        # 1. Configure camera QoS properly (BEST_EFFORT)
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        
        # 2. Camera subscriber (critical fix)
        self.bridge = CvBridge()
        self.create_subscription(
            Image,
            "/camera/color/image_raw",  # MUST match your camera topic
            self.image_callback,
            qos_profile=camera_qos
        )
        
        # 3. Movement publisher
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
        # 4. Square movement logic
        self.state = "FORWARD"
        self.start_time = time.time()
        self.get_logger().info("Node started with proper QoS config")
        
        # 5. Timer for movement control
        ###self.create_'timer(0.1, self.move_square)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("LIMO Camera Feed", cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Camera error: {str(e)}")

    def move_square(self):
        twist = Twist()
        elapsed = time.time() - self.start_time
        
        if self.state == "FORWARD" and elapsed >= 3.0:  # Move forward for 3 sec
            self.state = "TURN"
            self.start_time = time.time()
            self.get_logger().info("State changed to TURN")
        elif self.state == "TURN" and elapsed >= 1.57:  # Turn for π/2 seconds (~90°)
            self.state = "FORWARD"
            self.start_time = time.time()
            self.get_logger().info("State changed to FORWARD")
        
        twist.linear.x = 0.2 if self.state == "FORWARD" else 0.0
        twist.angular.z = 0.5 if self.state == "TURN" else 0.0
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = SquareMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown signal received")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
