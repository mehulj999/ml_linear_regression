import rclpy
from rclpy.node import Node
import pandas as pd
import joblib
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

class TestBostonHousingNode(Node):
    def __init__(self):
        super().__init__('test_boston_housing_node')

        # Load model
        package_share = Path(get_package_share_directory('ml_linear_regression'))
        model_path = package_share / 'models' / 'boston_housing_model.pkl'

        model = joblib.load(model_path)

        self.get_logger().info("🏠 Model loaded. Predicting...")

        # Use model's expected feature names
        features = model.feature_names_in_

        # Provide dummy values in the same order as feature_names_in_
        # Example dummy input (change values as needed for testing)
        input_data = [[
            0.1,   # crim
            25.0,  # zn
            5.0,   # indus
            0.0,   # chas
            0.5,   # nox
            6.0,   # rm
            60.0,  # age
            4.0,   # dis
            1.0,   # rad
            300.0, # tax
            15.0,  # ptratio
            390.0, # b
            10.0   # lstat
        ]]

        input_df = pd.DataFrame(input_data, columns=features)
        prediction = model.predict(input_df)[0]

        self.get_logger().info(f"🏡 Predicted MEDV (home value): ${prediction:.2f}k")

def main():
    rclpy.init()
    TestBostonHousingNode()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
