#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64
import joblib
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory


class HousePricePredictorNode(Node):
    def __init__(self):
        super().__init__('house_price_predictor_node')

        # Load the pre-trained model
        self.load_model()

        # Create subscriber for input features
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'house_features',
            self.predict_price_callback,
            10
        )

        # Create publisher for predicted price
        self.publisher = self.create_publisher(
            Float64,
            'predicted_house_price',
            10
        )

        # Feature names for validation
        self.feature_names = [
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
            "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
        ]

        self.get_logger().info('House Price Predictor Node initialized')
        self.get_logger().info(f'Expected features ({len(self.feature_names)}): {self.feature_names}')
        self.get_logger().info('Waiting for house features on topic: /house_features')
        self.get_logger().info('Publishing predictions on topic: /predicted_house_price')

    def load_model(self):
        """Load the pre-trained model from the models directory"""
        try:
            package_share_directory = get_package_share_directory('house_price_prediction')
            model_path = os.path.join(package_share_directory, 'models', 'boston_housing_model.pkl')

            self.get_logger().info(f'Loading model from: {model_path}')
            self.model = joblib.load(model_path)
            self.get_logger().info('Model loaded successfully')

        except FileNotFoundError:
            self.get_logger().error(f'Model file not found at: {model_path}')
            self.get_logger().error('Please ensure boston_housing_model.pkl is in the models/ directory')
            raise
        except Exception as e:
            self.get_logger().error(f'Error loading model: {str(e)}')
            raise

    def predict_price_callback(self, msg):
        """Callback function to handle incoming feature data and make predictions"""
        try:
            # Extract features from the message
            features = np.array(msg.data)

            # Validate input dimensions
            if len(features) != 13:
                self.get_logger().error(
                    f'Expected 13 features, got {len(features)}. '
                    f'Required features: {self.feature_names}'
                )
                return

            # Log received features
            self.get_logger().info(f'Received features: {features}')

            # Reshape for prediction (model expects 2D array)
            features_reshaped = features.reshape(1, -1)

            # Make prediction
            predicted_price = self.model.predict(features_reshaped)[0]

            # Create and publish the prediction message
            prediction_msg = Float64()
            prediction_msg.data = float(predicted_price)

            self.publisher.publish(prediction_msg)

            # Log the prediction
            self.get_logger().info(f'Predicted house price: ${predicted_price:.2f}k')

            # Log feature breakdown for debugging
            feature_log = []
            for i, (name, value) in enumerate(zip(self.feature_names, features)):
                feature_log.append(f'{name}: {value:.3f}')

            self.get_logger().debug('Feature breakdown: ' + ', '.join(feature_log))

        except Exception as e:
            self.get_logger().error(f'Error during prediction: {str(e)}')


def main(args=None):
    rclpy.init(args=args)

    try:
        house_price_predictor = HousePricePredictorNode()
        rclpy.spin(house_price_predictor)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if 'house_price_predictor' in locals():
            house_price_predictor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()