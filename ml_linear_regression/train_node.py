import rclpy
from rclpy.node import Node
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from pathlib import Path
from ament_index_python.packages import get_package_share_directory


class TrainNode(Node):
    def __init__(self):
        super().__init__('train_node')
        self.get_logger().info("🚀 Training models...")

        # Define shared and model paths
        package_share = Path(get_package_share_directory('ml_linear_regression'))
        data_path = package_share / 'data'
        model_path = package_share / 'models'
        model_path.mkdir(exist_ok=True)

        # Define datasets and targets
        datasets = [
            {
                "file": "new_height_weight.csv",
                "target": "weight"
            },
            {
                "file": "HumanBrain_WeightandHead_size.csv",
                "target": "brain weight(grams)"
            },
            {
                "file": "boston_housing.csv",
                "target": "medv"
            }
        ]

        for data in datasets:
            try:
                df = pd.read_csv(data_path / data["file"])

                # Clean column names: strip spaces & lowercase
                df.columns = df.columns.str.strip().str.lower()
                target_col = data["target"].strip().lower()

                # Verify target column exists
                if target_col not in df.columns:
                    self.get_logger().error(f"❌ Column '{target_col}' not found in {data['file']}. Skipping...")
                    continue

                # Prepare features and target
                X = df.drop(columns=[target_col])
                y = df[target_col]

                # Train-test split
                X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train model
                model = LinearRegression().fit(X_train, y_train)

                # Save model
                model_name = Path(data["file"]).stem + "_model.pkl"
                joblib.dump(model, model_path / model_name)

                self.get_logger().info(f"✅ Trained and saved: {model_name}")

            except Exception as e:
                self.get_logger().error(f"🔥 Error processing {data['file']}: {e}")

        self.get_logger().info("🎉 All valid models processed.")

def main():
    rclpy.init()
    TrainNode()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
