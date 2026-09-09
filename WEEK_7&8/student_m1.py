import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
# Read dataset
df = pd.read_csv("student_data.csv")

print("Student Data:")
print(df)

# Linear Regression

X = df[["Study_Hours", "Attendance", "Previous_Marks"]]
y = df["Final_Marks"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict final marks
y_pred = model.predict(X_test)

print("\nLinear Regression Predictions:")
print(y_pred)

# Model accuracy
r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)

# Polynomial Regression

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train_poly)

y_pred_poly = poly_model.predict(X_test_poly)

poly_r2 = r2_score(y_test_poly, y_pred_poly)

print("\nPolynomial Regression Predictions:")
print(y_pred_poly)

print("Polynomial Regression R2 Score:", poly_r2)

# Ridge Regression

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)

ridge_r2 = r2_score(y_test, y_pred_ridge)

print("\nRidge Regression Predictions:")
print(y_pred_ridge)

print("Ridge Regression R2 Score:", ridge_r2)
# Lasso Regression

lasso_model = Lasso(alpha=0.1)

lasso_model.fit(X_train, y_train)

y_pred_lasso = lasso_model.predict(X_test)

lasso_r2 = r2_score(y_test, y_pred_lasso)

print("\nLasso Regression Predictions:")
print(y_pred_lasso)

print("Lasso Regression R2 Score:", lasso_r2)
