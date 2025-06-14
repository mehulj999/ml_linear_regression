#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import time

class HeightPublisher(Node):
    def __init__(self):
        super().__init__('height_publisher')

        # Create publisher for height
        self.height_publisher = self.create_publisher(
            Float64,
            '/height_input',
            10
        )

        # Create subscriber to receive predictions
        self.weight_subscriber = self.create_subscription(
            Float64,
            '/weight_prediction',
            self.weight_callback,
            10
        )

        # Timer to publish test heights
        self.timer = self.create_timer(3.0, self.publish_height)

        # Test heights in cm
        self.test_heights = [150, 160, 170, 175, 180, 185, 190]
        self.current_index = 0

        self.get_logger().info('Height Publisher Node started')
        self.get_logger().info('Publishing test heights and waiting for predictions...')

    def publish_height(self):
        """Publish a test height"""
        if self.current_index < len(self.test_heights):
            height = self.test_heights[self.current_index]

            msg = Float64()
            msg.data = float(height)
            self.height_publisher.publish(msg)

            self.get_logger().info(f'Published height: {height} cm')
            self.current_index += 1
        else:
            self.get_logger().info('All test heights published!')
            self.timer.cancel()

    def weight_callback(self, msg):
        """Callback to receive weight predictions"""
        predicted_weight = msg.data
        self.get_logger().info(f'Received prediction: {predicted_weight:.1f} kg')

def main(args=None):
    rclpy.init(args=args)

    height_publisher = HeightPublisher()

    try:
        rclpy.spin(height_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        height_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
