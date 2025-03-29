#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray, Pose
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import cv2
import cv2.aruco
import numpy as np
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import time

class SquareArucoNavigator(Node):
    def __init__(self):
        super().__init__('square_aruco_navigator')
        
        # Camera setup (BEST_EFFORT QoS)
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
        
        # Movement control
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
        # Marker publishing
        self.markers_pub = self.create_publisher(PoseArray, '/detected_markers', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Camera parameters (example values - adjust for your camera)
        self.camera_matrix = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))
        self.marker_size = 0.1  # Physical marker size in meters

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect markers
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)
            
            if ids is not None:
                # Estimate marker poses in 3D space
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs)
                
                # Draw detected markers
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                
                # Process each detected marker
                pose_array = PoseArray()
                pose_array.header = msg.header
                
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id in [1, 2, 3, 4]:  # Only process target IDs
                        # Create proper Pose message with 3D coordinates
                        pose = Pose()
                        pose.position.x = tvecs[i][0][0]  # X in meters
                        pose.position.y = tvecs[i][0][1]  # Y in meters
                        pose.position.z = tvecs[i][0][2]  # Z in meters
                        pose_array.poses.append(pose)
                        
                        self.get_logger().info(
                            f"Marker {marker_id} at: "
                            f"X={tvecs[i][0][0]:.2f}m, "
                            f"Y={tvecs[i][0][1]:.2f}m"
                        )
                
                self.markers_pub.publish(pose_array)
            
            cv2.imshow("ArUco Detection", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Detection error: {str(e)}")

    def move_robot(self):
        twist = Twist()
        twist.linear.x = 0.15
        twist.angular.z = 0.3
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = SquareArucoNavigator()
    #node.create_timer(0.1, node.move_robot)
    
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
