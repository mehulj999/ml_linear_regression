from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='ml_regression_ros', executable='train_node'),
        Node(package='ml_regression_ros', executable='test_height_weight_node'),
        Node(package='ml_regression_ros', executable='test_brain_weight_node'),
        Node(package='ml_regression_ros', executable='test_boston_housing_node'),
    ])
