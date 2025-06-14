import rclpy
from rclpy.node import Node
import pandas as pd
import joblib
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

class TestBrainWeightNode(Node):
    def __init__(self):
        super().__init__('test_brain_weight_node')

        # Load model
        package_share = Path(get_package_share_directory('ml_linear_regression'))
        model_path = package_share / 'models' / 'humanbrain_weightandhead_size_model.pkl'

        model = joblib.load(model_path)

        self.get_logger().info("🧠 Model loaded. Predicting...")

        # Match training: lowercase column name
        input_df = pd.DataFrame([[4000]], columns=['head size(cm^3)'])
        prediction = model.predict(input_df)[0]

        self.get_logger().info(f"🧠 Predicted brain weight for head size 4000 cm³: {prediction:.2f} grams")

def main():
    rclpy.init()
    TestBrainWeightNode()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
