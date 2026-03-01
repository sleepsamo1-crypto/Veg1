import xgboost as xgb
import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv('data.csv')  # adjust the path to your dataset
X = data.drop('target', axis=1)  # features
y = data['target']  # target

# Create DMatrices
dtrain = xgb.DMatrix(X, label=y)

# Specify parameters
gbtree_params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100
}

# Train the model
model = xgb.train(gbtree_params, dtrain)

# Make predictions
predictions = model.predict(dtrain)
print(predictions)