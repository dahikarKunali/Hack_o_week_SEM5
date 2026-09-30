# Week 13 & 14 - Hack-O-Week
# Ensemble Methods, Bias-Variance, Overfitting, Underfitting and Regularization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# Create a sample classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

# Convert into a DataFrame
data = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(10)])
data["Target"] = y

print("Dataset created successfully!")
print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nData split successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==============================
# 1. BAGGING
# ==============================

bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=50,
    random_state=42
)

# Train the Bagging model
bagging_model.fit(X_train, y_train)

# Make predictions
bagging_pred = bagging_model.predict(X_test)

# Calculate accuracy
bagging_accuracy = accuracy_score(y_test, bagging_pred)

print("\n===== BAGGING RESULTS =====")
print("Bagging Accuracy:", bagging_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, bagging_pred))

# ==============================
# 2. XGBOOST BOOSTING
# ==============================

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric="logloss"
)

# Train the XGBoost model
xgb_model.fit(X_train, y_train)

# Make predictions
xgb_pred = xgb_model.predict(X_test)

# Calculate accuracy
xgb_accuracy = accuracy_score(y_test, xgb_pred)

print("\n===== XGBOOST RESULTS =====")
print("XGBoost Accuracy:", xgb_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, xgb_pred))

# ==============================
# 3. LIGHTGBM BOOSTING
# ==============================

lgbm_model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    verbosity=-1
)

# Train the LightGBM model
lgbm_model.fit(X_train, y_train)

# Make predictions
lgbm_pred = lgbm_model.predict(X_test)

# Calculate accuracy
lgbm_accuracy = accuracy_score(y_test, lgbm_pred)

print("\n===== LIGHTGBM RESULTS =====")
print("LightGBM Accuracy:", lgbm_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, lgbm_pred))

# ==========================================
# 4. BIAS-VARIANCE, OVERFITTING & UNDERFITTING
# ==========================================

depths = [1, 2, 3, 5, 10, None]

train_scores = []
test_scores = []

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    train_scores.append(train_accuracy)
    test_scores.append(test_accuracy)

    print(f"\nDecision Tree Depth: {depth}")
    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"Testing Accuracy: {test_accuracy:.3f}")


# ==========================================
# Plot Training vs Testing Accuracy
# ==========================================

plt.figure(figsize=(10, 6))

depth_labels = ["1", "2", "3", "5", "10", "Unlimited"]

plt.plot(
    depth_labels,
    train_scores,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    depth_labels,
    test_scores,
    marker="o",
    label="Testing Accuracy"
)

plt.xlabel("Decision Tree Depth")
plt.ylabel("Accuracy")
plt.title("Bias-Variance Trade-off")
plt.legend()
plt.grid(True)

plt.show()

# ==========================================
# 5. L1 AND L2 REGULARIZATION
# ==========================================

from sklearn.linear_model import LogisticRegression

# L1 Regularization
l1_model = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=1.0,
    random_state=42
)

l1_model.fit(X_train, y_train)

l1_pred = l1_model.predict(X_test)

l1_accuracy = accuracy_score(y_test, l1_pred)


# L2 Regularization
l2_model = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    C=1.0,
    random_state=42,
    max_iter=1000
)

l2_model.fit(X_train, y_train)

l2_pred = l2_model.predict(X_test)

l2_accuracy = accuracy_score(y_test, l2_pred)


print("\n===== REGULARIZATION RESULTS =====")

print("L1 Regularization Accuracy:", l1_accuracy)
print("L2 Regularization Accuracy:", l2_accuracy)

print("\nL1 Coefficients:")
print(l1_model.coef_)

print("\nL2 Coefficients:")
print(l2_model.coef_)

# ==========================================
# 6. FINAL MODEL COMPARISON
# ==========================================

model_names = [
    "Bagging",
    "XGBoost",
    "LightGBM",
    "L1 Logistic Regression",
    "L2 Logistic Regression"
]

model_accuracies = [
    bagging_accuracy,
    xgb_accuracy,
    lgbm_accuracy,
    l1_accuracy,
    l2_accuracy
]

comparison = pd.DataFrame({
    "Model": model_names,
    "Accuracy": model_accuracies
})

print("\n===== FINAL MODEL COMPARISON =====")
print(comparison.to_string(index=False))


# Plot model comparison
plt.figure(figsize=(10, 6))

plt.bar(model_names, model_accuracies)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Ensemble Methods and Regularization Comparison")
plt.ylim(0, 1)

plt.xticks(rotation=20)
plt.grid(axis="y")

plt.show()

# ==========================================
# 7. SAVE RESULTS
# ==========================================

comparison.to_csv("model_comparison_results.csv", index=False)

print("\nResults saved successfully!")
print("File created: model_comparison_results.csv")