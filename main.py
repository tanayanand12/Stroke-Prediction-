from stroke_predictor import StrokePredictor
import pandas as pd

# Initialize predictor
predictor = StrokePredictor()

# Load and preprocess your data
data = pd.read_csv('healthcare-dataset-stroke-data.csv')
processed_data = predictor.preprocess_data(data)

# Split features and target
X = processed_data.iloc[:, :-1]
y = processed_data['stroke']

# Train models
X_test, y_test = predictor.train_models(X, y)

# Save models
predictor.save_models()