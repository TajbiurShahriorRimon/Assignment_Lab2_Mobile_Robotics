"""
CPSC5207EL02 Intelligent Mobile Robotics

To execute the script: download the Python code and place in the turtlebot3 workspace.
Run the python script directly using: python3 ~/turtlebot3_ws/filename.py

This python code integrates two functionalities: 
1. controlling the Turtlebot3 to move in a rectangular pattern and 
2. displaying live camera feed using OpenCV. 

It demonstrates how to publish velocity commands, subscribe to image topics, 
convert ROS image messages to OpenCV format, and manage simple state transitions 
for robot movement.

- The TurtlebotController class extends Node and integrates both movement control and image 
subscription/display.

- The __init__ method, it sets up a publisher for movement commands, a subscriber for 
camera images, and initializes a timer for periodically updating the robot's state 
(moving forward or turning).

- The update_state method manages the robot's movement by publishing velocity commands 
based on the current state and elapsed time.

- The image_callback method receives image messages from the camera, converts them to OpenCV 
format using cv_bridge, and displays them.

Note:
Ensure your ROS2 environment is correctly set up with all necessary dependencies for cv_bridge, 
OpenCV, and the Turtlebot3 packages. Also, adjust the topic name for the camera images 
if your setup uses a different topic.

Remember to execute this script in an environment where your ROS2 workspace is sourced, 
and all dependencies are installed. This script assumes the Turtlebot3 simulation or a 
real Turtlebot3 is running and publishing camera images to the /camera/image_raw topic.

"""

# Import necessary ROS2 and OpenCV libraries
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math


class TurtlebotController(Node):
    def __init__(self):
        # Initialize the node with the name 'turtlebot_controller'
        super().__init__("turtlebot_controller")

        # Movement control setup
        # Create a publisher for sending velocity commands
        # This publisher will send messages of type Twist to the 'cmd_vel' topic,
        # which is commonly used for controlling robot motion. The queue size of 10
        # ensures that up to 10 messages can be buffered for sending if necessary,
        # managing the flow of commands under varying system loads.
        self.publisher_ = self.create_publisher(Twist, "/cmd_vel", 10)
        # Initial state for the movement logic
        self.state = "move_forward"
        # Record start time for timing the movements
        self.start_time = self.get_clock().now()
        # Define durations for moving forward and turning
        self.duration_move = 2.0  # Move forward for 2 seconds
        self.duration_turn = math.pi / 2 / 0.2  # Time to turn 90 degrees at 0.2 rad/s
        # Create a timer to periodically update the robot's state
        self.timer = self.create_timer(0.1, self.update_state)

        # Configure QoS profile for publishing and subscribing
        # Quality of Service (qos) policies that allow you to tune communication between nodes.
        # QoS policies for subscriber must match the publisher
        # For more info, see: https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # BEST_EFFORT: attempt to deliver samples, but may lose them if the network is not robust.
            durability=DurabilityPolicy.VOLATILE,  # VOLATILE: no attempt is made to persist samples.
            history=HistoryPolicy.KEEP_LAST,  # KEEP_LAST: only store up to N samples, configurable via the queue depth option.
            depth=10,  # a queue size of 10 to buffer messages if they arrive faster than they can be processed
        )

        # Subscribe to the camera topic to receive image messages
        # Create a subscription to listen for messages on the '/camera/image_raw' topic,
        # using the Image message type. The 'image_callback' function is called for each new message
        self.subscription = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            qos_profile=qos_profile,
        )
        self.bridge = (
            CvBridge()
        )  # Initialize a CvBridge to convert ROS images to OpenCV format

    def update_state(self):
        # This method updates the robot's movement state based on elapsed time
        current_time = self.get_clock().now()
        elapsed_time = current_time - self.start_time
        elapsed_seconds = (
            elapsed_time.nanoseconds / 1e9
        )  # Convert nanoseconds to seconds

        msg = Twist()

        # Switch between moving forward and turning based on the state and elapsed time
        if self.state == "move_forward" and elapsed_seconds >= self.duration_move:
            self.state = "turn"
            self.start_time = (
                self.get_clock().now()
            )  # Reset start time for the next state
        elif self.state == "turn" and elapsed_seconds >= self.duration_turn:
            self.state = "move_forward"
            self.start_time = self.get_clock().now()

        # Set the velocity based on the current state
        if self.state == "move_forward":
            msg.linear.x = 0.2
            msg.angular.z = 0.0
        elif self.state == "turn":
            msg.linear.x = 0.0
            msg.angular.z = -0.2  # Clockwise rotation

        self.publisher_.publish(msg)  # Publish the velocity command

    def image_callback(self, msg):
        # This method is called with each new image message from the camera
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error("Failed to convert image: " + str(e))
            return

        # Display the OpenCV image in a window
        cv2.imshow("Camera Image", cv_image)
        cv2.waitKey(1)  # Wait a bit for the window to update


def main(args=None):
    rclpy.init(args=args)  # Initialize ROS2 Python client library
    turtlebot_controller = TurtlebotController()  # Create the Turtlebot controller node
    rclpy.spin(turtlebot_controller)  # Keep the node running and responsive
    # Cleanup before exiting
    turtlebot_controller.destroy_node()
    cv2.destroyAllWindows()  # Close the OpenCV window
    rclpy.shutdown()  # Shutdown ROS2 Python client library


if __name__ == "__main__":
    main()
