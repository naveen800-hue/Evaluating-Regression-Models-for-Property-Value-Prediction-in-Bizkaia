# -------------------------------------------
# 🏠 House Price Prediction - Bizkaia Dataset
# -------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ---------------------------
# 1. Load and Inspect Dataset
# ---------------------------

df = pd.read_csv("houses_bizkaia.csv")
print(df.info())

# Drop columns with >70% missing values
drop_cols = ['construct_date', 'garage', 'ground_size', 'heating', 'kitchen', 
             'lift', 'loc_neigh', 'loc_street', 'orientation', 'unfurnished', 'house_id']
df.drop(columns=drop_cols, inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str)  # convert to string
    df[col] = label_encoder.fit_transform(df[col])

# Fill missing values in numeric column
df['m2_useful'] = df['m2_useful'].fillna(df['m2_useful'].median())

# -------------------------
# 2. Split Data for Training
# -------------------------

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=47)

# ----------------------------
# 3. Model Training & Saving
# ----------------------------

def calculate_metrics(model_name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) * 100

    print(f"{model_name} MSE: {mse:.2f}")
    print(f"{model_name} MAE: {mae:.2f}")
    print(f"{model_name} R² Score: {r2:.2f}%")

    # Plot Actual vs Predicted
    plt.figure(figsize=(7,7))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{model_name} Prediction Performance")
    plt.grid()
    plt.show()

# --- Linear Regression ---
lr_model_path = 'LinearRegression_model.pkl'
if os.path.exists(lr_model_path):
    lr = joblib.load(lr_model_path)
    print("✅ Linear Regression model loaded.")
else:
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    joblib.dump(lr, lr_model_path)
    print("✅ Linear Regression model trained and saved.")

lr_preds = lr.predict(X_test)
calculate_metrics("Linear Regression", y_test, lr_preds)

# --- Gradient Boosting ---
gbr_model_path = 'GradientBoosting_model.pkl'
if os.path.exists(gbr_model_path):
    gbr = joblib.load(gbr_model_path)
    print("✅ Gradient Boosting model loaded.")
else:
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train, y_train)
    joblib.dump(gbr, gbr_model_path)
    print("✅ Gradient Boosting model trained and saved.")

gbr_preds = gbr.predict(X_test)
calculate_metrics("Gradient Boosting Regressor", y_test, gbr_preds)

# ---------------------------
# 4. Load and Predict on Test
# ---------------------------

test = pd.read_csv("test.csv")

# Drop same columns as training set
test.drop(columns=drop_cols, inplace=True)

# Encode test set using same approach
for col in test.select_dtypes(include='object').columns:
    test[col] = test[col].astype(str)
    test[col] = label_encoder.fit_transform(test[col])

# Fill missing values
test['m2_useful'] = test['m2_useful'].fillna(df['m2_useful'].median())

# Ensure same columns as training set
missing_cols = [col for col in X_train.columns if col not in test.columns]
for col in missing_cols:
    test[col] = 0  # or better: use training column median if known

# Predict using the Gradient Boosting model
final_predictions = gbr.predict(test)
test['predicted_price'] = final_predictions

print(test[['predicted_price']].head(10))  # display top 10 predictions

# Save predictions if needed
test.to_csv("test_with_predictions.csv", index=False)
