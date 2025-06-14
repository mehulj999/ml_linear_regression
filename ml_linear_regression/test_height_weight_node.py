import rclpy
from rclpy.node import Node
import pandas as pd
import joblib
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

class TestHeightWeightNode(Node):
    def __init__(self):
        super().__init__('test_height_weight_node')

        # Get model path
        package_share = Path(get_package_share_directory('ml_linear_regression'))
        model_path = package_share / 'models' / 'new_height_weight_model.pkl'

        model = joblib.load(model_path)

        self.get_logger().info("✅ Model loaded. Predicting...")

        # MUST match the training format: lowercase column names
        input_df = pd.DataFrame([[170]], columns=['height'])
        prediction = model.predict(input_df)[0]

        self.get_logger().info(f"🎯 Predicted weight for height 170in: {prediction:.2f}lb")

def main():
    rclpy.init()
    TestHeightWeightNode()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
