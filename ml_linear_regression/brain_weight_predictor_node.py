#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import Point
import joblib
import os
import numpy as np
from ament_index_python.packages import get_package_share_directory

class BrainWeightPredictorNode(Node):
    def __init__(self):
        super().__init__('brain_weight_predictor_node')

        # Load the trained model
        self.load_model()

        # Create subscriber for size input
        self.size_subscriber = self.create_subscription(
            Float64,
            '/size_input',
            self.size_callback,
            10
        )

        # Create publisher for weight prediction
        self.weight_publisher = self.create_publisher(
            Float64,
            '/brain_weight_prediction',
            10
        )

        # Create a service for one-time predictions (optional)
        self.get_logger().info('Weight Predictor Node has been started')
        self.get_logger().info('Listening for size input on /size_input topic')
        self.get_logger().info('Publishing weight predictions on /brain_weight_prediction topic')

    def load_model(self):
        """Load the joblib model file"""
        try:
            # Try to get the model from the package share directory
            package_name = 'brain_weight_prediction'
            try:
                package_share_directory = get_package_share_directory(package_name)
                model_path = os.path.join(package_share_directory, 'models', 'human_brain_weight_predictor.pkl')
            except:
                # Fallback to local path if package path doesn't work
                model_path = 'human_brain_weight_predictor.pkl'

            # Use joblib.load instead of pickle.load
            self.model = joblib.load(model_path)

            # Debug: Check what type of object was loaded
            self.get_logger().info(f'Model loaded successfully from {model_path}')
            self.get_logger().info(f'Model type: {type(self.model)}')

            # Check if it's a proper model with predict method
            if hasattr(self.model, 'predict'):
                self.get_logger().info('Model has predict method - ready for predictions')

                # Test the model with a sample prediction
                test_input = np.array([[1750]])
                test_prediction = self.model.predict(test_input)
                self.get_logger().info(f'Test prediction: Size 175cm -> Weight {test_prediction[0]:.1f}kg')

            else:
                self.get_logger().error('Loaded object does not have predict method!')
                self.get_logger().error('This suggests the file contains data instead of a trained model')
                raise AttributeError('Model object does not have predict method')

        except FileNotFoundError:
            self.get_logger().error(f'Model file not found at {model_path}')
            self.get_logger().error('Please ensure human_brain_weight_predictor.pkl is in the correct location')
            raise
        except Exception as e:
            if "numpy.core.multiarray" in str(e) or "ARRAY_API" in str(e):
                self.get_logger().error('NumPy compatibility issue detected!')
                self.get_logger().error('Your model was trained with NumPy 1.x but you have NumPy 2.x installed.')
                self.get_logger().error('Solutions:')
                self.get_logger().error('1. Run: pip install "numpy<2.0"')
                self.get_logger().error('2. Or retrain your model with current NumPy version')
            else:
                self.get_logger().error(f'Error loading model: {str(e)}')
            raise

    def size_callback(self, msg):
        """Callback function when size data is received"""
        size = msg.data

        # Validate input
        if size <= 500 or size > 5000:  # Assuming size in cm3
            self.get_logger().warn(f'Invalid size received: {size}')
            return

        try:
            # Make prediction
            # Reshape size for sklearn (needs 2D array)
            size_array = np.array([[size]])
            predicted_weight = self.model.predict(size_array)[0]

            # Create and publish the result
            weight_msg = Float64()
            weight_msg.data = float(predicted_weight)
            self.weight_publisher.publish(weight_msg)

            self.get_logger().info(f'Size: {size:.1f} in -> Predicted Weight: {predicted_weight:.1f} lb')

        except Exception as e:
            self.get_logger().error(f'Error making prediction: {str(e)}')

def main(args=None):
    rclpy.init(args=args)

    try:
        brain_weight_predictor_node = BrainWeightPredictorNode()
        rclpy.spin(brain_weight_predictor_node)
    except Exception as e:
        print(f'Error starting node: {e}')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
