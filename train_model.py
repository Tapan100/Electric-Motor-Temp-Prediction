import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings

warnings.filterwarnings("ignore")

print("\U0001F4E5 Loading dataset...", flush=True)
df = pd.read_csv('dataset/pmsm_temperature_data.csv')
df = df.sample(1000, random_state=42)

print("\n\u2705 Dataset Loaded:", flush=True)
print(df.head(), flush=True)

print("\n\U00002139 Dataset Info:")
print(df.info(), flush=True)

print("\n\U0001F4CA Descriptive Statistics:\n", flush=True)
print(df.describe(), flush=True)

print("\n\U0001F9F9 Preprocessing...", flush=True)
df = df.dropna()

if 'target' in df.columns:
    y = df['target']
    X = df.drop(['target'], axis=1)
elif 'torque' in df.columns:
    y = df['torque']
    X = df.drop(['torque', 'profile_id'], axis=1) if 'profile_id' in df.columns else df.drop(['torque'], axis=1)
else:
    raise ValueError("\u274C Could not find a valid target column like 'target' or 'torque'.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("\u2705 Preprocessing complete.", flush=True)

print("\n\U0001F680 Starting model training...", flush=True)
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'SVM': SVR()
}

best_model = None
best_score = -float('inf')

for name, model in models.items():
    print(f"\n\U0001F527 Training: {name}", flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\U0001F4C8 {name}: R² = {r2:.4f}, MSE = {mse:.4f}", flush=True)
    if r2 > best_score:
        best_score = r2
        best_model = model

os.makedirs('models', exist_ok=True)
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'scaler': scaler}, f)

print(f"\n🏆 Best Model: {type(best_model).__name__} with R² = {best_score:.4f}", flush=True)
print("\u2705 Model saved as models/best_model.pkl", flush=True)
