from setuptools import setup
import os
from glob import glob

package_name = 'ml_linear_regression'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'data'), glob('data/*.csv')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='ROS 2 package for linear regression using sklearn on multiple datasets.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_node = ml_linear_regression.train_node:main',
            'test_height_weight_node = ml_linear_regression.weight_predictor_node:main',
            'test_brain_weight_node = ml_linear_regression.brain_weight_predictor_node:main',
            'test_boston_housing_node = ml_linear_regression.house_price_predictor_node:main',
        ],
    },
)
