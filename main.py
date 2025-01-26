import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
data = pd.read_csv("Housing.csv")

# Data Exploration
print(data.info())
print(data.describe())

# Visualize Target Variable Distribution
sns.histplot(data['price'], kde=True)
plt.title("Price Distribution")
plt.show()

# Handling Categorical Features (Binary Encoding)
boolean_columns = ['mainroad', 'guestroom', 'basement',
                   'hotwaterheating', 'airconditioning', 'prefarea']
for col in boolean_columns:
    data[col] = data[col].map({'yes': 1, 'no': 0})

# One-Hot Encoding for 'furnishingstatus'
data = pd.get_dummies(data, columns=['furnishingstatus'], drop_first=True)

# Define Features and Target
X = data.drop(columns=['price'])
y = data['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Models Initialization
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Gradient Boosting": GradientBoostingRegressor()
}

# Hyperparameter Tuning for Ridge, Lasso, and Gradient Boosting
param_grids = {
    "Ridge": {'alpha': [0.1, 1.0, 10.0]},
    "Lasso": {'alpha': [0.01, 0.1, 1.0]},
    "Gradient Boosting": {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
}

# Training and Validation
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    if name in param_grids:
        grid_search = GridSearchCV(model, param_grids[name], cv=KFold(
            5), scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "Model": best_model,
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    }
    print(f"{name}: MSE = {mse:.2f}, MAE = {mae:.2f}, R2 = {r2:.2f}")

# Feature Importance (For Gradient Boosting)
gb_model = results['Gradient Boosting']['Model']
feature_importances = pd.Series(gb_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()

# Final Results Summary
print("\nFinal Model Performance:")
for name, metrics in results.items():
    print(
        f"{name}: MSE = {metrics['MSE']:.2f}, MAE = {metrics['MAE']:.2f}, R2 = {metrics['R2']:.2f}")
