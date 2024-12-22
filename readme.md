# Stroke Prediction Model

## Project Overview
This project implements machine learning models to predict stroke risk based on various health indicators and demographic information. The project uses both XGBoost and Random Forest classifiers to create accurate prediction models.

## Dataset
The dataset used is the Healthcare Stroke Dataset, which includes the following features:
- Gender
- Age
- Hypertension
- Heart Disease
- Ever Married
- Work Type
- Residence Type
- Average Glucose Level
- BMI
- Smoking Status

Target variable: Stroke (0 for no stroke, 1 for stroke)

## Project Structure
```
├── stroke_predictor.py                # Main module containing model implementation
├── Stroke-Prediction.ipynb            # Experiment file 
├── requirements.txt                   # Project dependencies
├── main.py                            # Module calling script
├── healthcare-data-stroke-data.csv   # Dataset
├── model.json                        # Saved XGBoost model
├── model.pkl                         # Saved Random Forest model
└── README.md                         # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stroke-prediction.git
cd stroke-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

```python
from stroke_predictor import StrokePredictor

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
```

## Model Features
- Data preprocessing including handling missing values and categorical encoding
- Feature scaling using StandardScaler
- Implementation of both XGBoost and Random Forest classifiers
- Model persistence functionality

## Dependencies
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
