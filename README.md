# ML Linear Regression Package

A ROS2 package implementing linear regression models using scikit-learn for vehicular technology applications. This package demonstrates machine learning fundamentals through three different regression datasets.

## Overview

This package provides training and prediction capabilities for three different linear regression models:
- **Height-Weight Prediction** - Predict weight based on height measurements
- **Brain-Weight Analysis** - Analyze relationship between brain and body weight
- **Boston Housing Prices** - Predict house prices based on various features

The package includes a unified training node and dedicated prediction nodes for each dataset, all communicating via ROS2 topics and services.

## Package Structure

```
ml_linear_regression/
├── ml_linear_regression/
│   ├── train_node.py                    # Unified training node for all models
│   ├── weight_predictor_node.py         # Height-to-weight prediction
│   ├── brain_weight_predictor_node.py   # Brain-weight analysis
│   └── house_price_predictor_node.py    # Boston housing price prediction
├── data/                # Training datasets (.csv files)
├── launch/              # Launch files (.py)
├── setup.py
└── package.xml
```

## Prerequisites

- ROS2 (Humble/Iron/Rolling)
- Python 3.8+
- Required Python packages (automatically installed):
  - scikit-learn
  - pandas
  - joblib
  - matplotlib
  - numpy

## Installation

1. Clone the repository into your ROS2 workspace:
```bash
cd ~/ros2_ws/src
git clone https://github.com/mehulj999/ml_linear_regression.git
```

2. Install dependencies:
```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

3. Build the package:
```bash
colcon build --packages-select ml_linear_regression
source install/setup.bash
```

## Usage

### Training Models

The package uses a unified training node that trains all three regression models:

```bash
ros2 run ml_linear_regression train_node
```

This node will:
- Load all three datasets from the `data/` directory
- Train linear regression models for each dataset
- Save trained models for use by prediction nodes
- Publish training metrics and results

### Prediction Nodes

After training, run the individual prediction nodes:

#### 1. Height-Weight Predictor
Predicts weight based on height input:
```bash
ros2 run ml_linear_regression test_height_weight_node
```

#### 2. Brain-Weight Predictor
Analyzes brain weight to body weight relationships:
```bash
ros2 run ml_linear_regression test_brain_weight_node
```

#### 3. Boston Housing Price Predictor
Predicts house prices based on housing features:
```bash
ros2 run ml_linear_regression test_boston_housing_node
```

## Testing the Models

### Method 1: Using ROS2 Services

Test predictions by calling the appropriate service:

```bash
# Height-Weight Prediction (provide height in cm/inches)
ros2 service call /predict_weight ml_linear_regression_msgs/PredictWeight "height: 175.0"

# Brain-Weight Prediction (provide body weight)
ros2 service call /predict_brain_weight ml_linear_regression_msgs/PredictBrainWeight "body_weight: 70.0"

# Housing Price Prediction (provide housing features array)
ros2 service call /predict_house_price ml_linear_regression_msgs/PredictHousePrice "features: [0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.0900, 1, 296.0, 15.3, 396.90, 4.98]"
```

### Method 2: Using ROS2 Topics

Send prediction requests via topics:

```bash
# List available topics
ros2 topic list

# Example: Send height data for weight prediction
ros2 topic pub /height_input std_msgs/Float64 "data: 175.0"

# Monitor weight predictions
ros2 topic echo /weight_prediction

# Monitor training status
ros2 topic echo /training_status
```

### Method 3: Interactive Testing

Some nodes may provide interactive input prompts when run:

```bash
# Run predictor and follow prompts
ros2 run ml_linear_regression test_height_weight_node
# Enter height when prompted: 175.5
```

## Launch Files

Use launch files to start multiple nodes simultaneously:

```bash
# Train all models
ros2 launch ml_linear_regression train_models.launch.py

# Start all predictors
ros2 launch ml_linear_regression start_predictors.launch.py

# Complete pipeline (training + prediction)
ros2 launch ml_linear_regression full_pipeline.launch.py
```

## Datasets

The package includes three CSV datasets in the `data/` directory:

### 1. Height-Weight Dataset
- **File**: `height_weight.csv`
- **Features**: Height measurements
- **Target**: Weight values
- **Use Case**: Basic linear relationship demonstration

### 2. Brain-Weight Dataset
- **File**: `brain_weight.csv`
- **Features**: Body weight measurements
- **Target**: Brain weight values  
- **Use Case**: Biological relationship analysis

### 3. Boston Housing Dataset
- **File**: `boston_housing.csv`
- **Features**: 13 housing attributes (crime rate, rooms, age, etc.)
- **Target**: Median home values
- **Use Case**: Multi-feature regression analysis

## Model Performance

Each predictor node provides performance metrics:

- **R² Score** - Coefficient of determination
- **Mean Squared Error (MSE)** - Average squared differences
- **Mean Absolute Error (MAE)** - Average absolute differences
- **Root Mean Squared Error (RMSE)** - Square root of MSE

View metrics via topics:
```bash
ros2 topic echo /model_metrics
```

## Visualization

The package may include matplotlib-based visualization:

```bash
# View training plots
ros2 topic echo /training_plots

# Generate prediction vs actual plots
ros2 service call /generate_plots std_srvs/Empty
```

## Configuration

Model parameters can be adjusted by modifying the training node or through ROS2 parameters:

```bash
# Set training parameters
ros2 run ml_linear_regression train_node --ros-args -p test_size:=0.3 -p random_state:=42
```

## Advanced Usage

### Custom Dataset Training

To train on your own dataset:

1. Place CSV file in `data/` directory
2. Modify `train_node.py` to include your dataset
3. Create corresponding predictor node
4. Update `setup.py` entry points

### Model Persistence

Trained models are automatically saved using joblib and can be reloaded:

```python
import joblib
model = joblib.load('path/to/model.pkl')
```

## Debugging

Enable detailed logging:
```bash
ros2 run ml_linear_regression train_node --ros-args --log-level DEBUG
```

Monitor node activity:
```bash
# Check running nodes
ros2 node list

# Inspect node info
ros2 node info /train_node

# View parameter values
ros2 param list /train_node
```

## Troubleshooting

**Dataset not found**: Ensure CSV files are in the `data/` directory and paths are correct.

**Import errors**: Install missing dependencies:
```bash
pip install scikit-learn pandas matplotlib joblib
```

**Model not trained**: Run the training node before starting predictors:
```bash
ros2 run ml_linear_regression train_node
# Wait for training completion, then start predictors
```

**Poor predictions**: Check data quality and consider:
- Feature scaling/normalization
- Outlier removal
- Different train/test splits
- Cross-validation

## Performance Tips

- Use appropriate train/test splits (typically 70/30 or 80/20)
- Consider feature scaling for multi-feature datasets
- Monitor for overfitting with validation curves
- Use cross-validation for robust model evaluation

## Examples

### Quick Start Example
```bash
# Terminal 1: Train models
ros2 run ml_linear_regression train_node

# Terminal 2: Test height-weight prediction
ros2 run ml_linear_regression test_height_weight_node

# Terminal 3: Send test data
ros2 topic pub /height_input std_msgs/Float64 "data: 170.0"
```

### Batch Prediction Example
```bash
# Use launch file for automated testing
ros2 launch ml_linear_regression full_pipeline.launch.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-dataset`)
3. Add your changes
4. Test thoroughly with all three models
5. Submit a pull request

## Future Enhancements

- Add polynomial regression options
- Implement cross-validation
- Add model comparison utilities
- Include feature importance analysis
- Support for custom loss functions

## License

MIT License - see LICENSE file for details

## Maintainer

**Mehul Jain** - mehulj999@hotmail.com

## Assignment Context

This package was developed for Machine Learning Assignment 1 in Vehicular Technology coursework, demonstrating practical implementation of linear regression algorithms in a ROS2 environment.
