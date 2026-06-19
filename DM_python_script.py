# Train_rf_model.py
# ===============================
# Digital Marketing Campaign Model Training Script
# ===============================
# This script loads the digital marketing dataset, preprocesses it,
# trains a Random Forest model, and saves the model and scaler for later use.
#
# Make sure your dataset file path is correct before running.
# ===============================

# 1. Import required libraries
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from joblib import dump

# Define file paths using Path for cross-platform compatibility
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_PATH = SCRIPT_DIR / 'digital_marketing_campaign_dataset.csv'
MODELS_DIR = SCRIPT_DIR / 'Models'
MODEL_PATH = MODELS_DIR / 'random_forest_marketing_model.joblib'
SCALER_PATH = MODELS_DIR / 'scaler_marketing.joblib'

# Create Models directory if it doesn't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Check if dataset exists
if not DATA_PATH.exists():
    print(f"❌ Error: dataset not found at {DATA_PATH}")
    print("Please ensure the CSV file exists or specify its path, e.g.:")
    print('  python DM_python_script.py "C:\\path\\to\\digital_marketing_campaign_dataset.csv"')
    sys.exit(1)

# Load the dataset
try:
    marketing_data = pd.read_csv(DATA_PATH)
    print("✅ Dataset loaded successfully!")
    print(f"Shape of data: {marketing_data.shape}")
except Exception as e:
    print(f"❌ Error loading dataset: {str(e)}")
    sys.exit(1)

# 2. Display column info (for reference)
print("\nColumns in dataset:")
print(marketing_data.columns.tolist())

# 3. Identify target variable
# You can modify this if your target column name is different (e.g., 'Converted', 'Response', etc.)
target_column = None
for col in marketing_data.columns:
    if col.lower() in ['converted', 'conversion', 'purchased', 'response']:
        target_column = col
        break

if not target_column:
    print("❌ Target column not found automatically. Please update 'target_column' manually.")
    exit(1)

print(f"\n🎯 Target variable detected: {target_column}")

# 4. Split features and target
X = marketing_data.drop(columns=[target_column])
y = marketing_data[target_column]

# Convert text targets to 0/1 if necessary
if y.dtype == 'object':
    y = y.map({'Yes': 1, 'No': 0})
    print("✅ Text targets converted to 0/1")

# 5. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# 6. Preprocess the data
# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'income', 'spend']),
        ('cat', OneHotEncoder(), ['gender', 'city', 'device_type'])],
    remainder='passthrough'
)

# Create the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 7. Train the model
pipeline.fit(X_train, y_train)
print("✅ Model trained successfully!")

# 8. Evaluate the model
y_pred = pipeline.predict(X_test)
print("✅ Model evaluation complete!")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 9. Save the model and scaler
dump(pipeline, MODEL_PATH)
dump(preprocessor, SCALER_PATH)
print("✅ Model and scaler saved successfully!")







