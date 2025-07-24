import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv("greenhouse_gas_emissions.csv")
print("Initial shape:", df.shape)
print("Columns:", df.columns)

# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 7'], errors='ignore')

# Remove rows with any missing values
df = df.dropna()

# Drop non-numeric or irrelevant columns
non_numeric_cols = ['Industry Code', 'Industry Name', 'Substance', 'Unit']
df_cleaned = df.drop(columns=non_numeric_cols, errors='ignore')

# Keep only numeric data
df_cleaned = df_cleaned.select_dtypes(include=[np.number])

# Check if any numeric data left
if df_cleaned.empty:
    raise ValueError("No valid numeric data to train on after cleaning. Please check your dataset.")

# Define features and target
X = df_cleaned.drop(columns=['Supply Chain Emission Factors with Margins'], errors='ignore')
y = df_cleaned['Supply Chain Emission Factors with Margins']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    print(f"\n{name}")
    print("RMSE:", results[name]['RMSE'])
    print("MAE:", results[name]['MAE'])
    print("R²:", results[name]['R2'])

# Tune Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2')
grid_search.fit(X_train, y_train)

# Final Model
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test)

# Plot Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, final_predictions, color='green', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.title("Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Residuals
residuals = y_test - final_predictions
plt.figure(figsize=(6, 4))
sns.histplot(residuals, bins=20, kde=True, color='orange')
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Plot Feature Importance
importances = final_model.feature_importances_
features = X.columns
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()


