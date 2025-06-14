#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import time

class HeightPublisher(Node):
    def __init__(self):
        super().__init__('size_publisher')

        # Create publisher for size
        self.size_publisher = self.create_publisher(
            Float64,
            '/size_input',
            10
        )

        # Create subscriber to receive predictions
        self.price_subscriber = self.create_subscription(
            Float64,
            '/house_price_prediction',
            self.price_callback,
            10
        )

        # Timer to publish test
        self.timer = self.create_timer(3.0, self.publish_height)

        # Test  in cm3
        self.test_ = [1150, 1620, 2170, 2750, 3080, 3185, 4190]
        self.current_index = 0

        self.get_logger().info('Size Publisher Node started')
        self.get_logger().info('Publishing test  and waiting for predictions...')

    def publish_height(self):
        """Publish a test size"""
        if self.current_index < len(self.test_):
            size = self.test_[self.current_index]

            msg = Float64()
            msg.data = float(size)
            self.size_publisher.publish(msg)

            self.get_logger().info(f'Published size: {size} cm3')
            self.current_index += 1
        else:
            self.get_logger().info('All test sizes published!')
            self.timer.cancel()

    def price_callback(self, msg):
        """Callback to receive price predictions"""
        predicted_price = msg.data
        self.get_logger().info(f'Received prediction: {predicted_price:.1f} kg')

def main(args=None):
    rclpy.init(args=args)

    size_publisher = HeightPublisher()

    try:
        rclpy.spin(size_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        size_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
