# ============================================================
# WEEK 9 & 10 - HACK-O-WEEK
# MODEL EVALUATION, FEATURE ENGINEERING & SCALING
# AUTOMATED STUDENT PERFORMANCE ANALYSIS
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    accuracy_score
)


# ============================================================
# 1. CREATE REQUIRED FOLDERS
# ============================================================

os.makedirs("results", exist_ok=True)
os.makedirs("graphs", exist_ok=True)


# ============================================================
# 2. CREATE STUDENT PERFORMANCE DATA
# ============================================================

data = {
    "Attendance": [
        85, 72, 91, 65, 78,
        88, 55, 95, 69, 82,
        74, 90, 61, 87, 79,
        93, 68, 76, 84, 58
    ],

    "Assignment_Score": [
        88, 65, 92, 55, 72,
        85, 45, 96, 60, 80,
        70, 94, 50, 89, 75,
        91, 58, 68, 86, 48
    ],

    "Test_Score": [
        85, 60, 95, 50, 70,
        88, 40, 98, 55, 78,
        65, 92, 45, 87, 73,
        94, 52, 62, 84, 43
    ],

    "Study_Hours": [
        7, 4, 8, 3, 5,
        7, 2, 9, 4, 6,
        5, 8, 3, 7, 5,
        9, 3, 4, 7, 2
    ],

    "Previous_Grade": [
        82, 64, 90, 48, 70,
        85, 42, 94, 58, 76,
        68, 91, 45, 86, 72,
        93, 50, 60, 81, 40
    ],

    "Passed": [
        1, 1, 1, 0, 1,
        1, 0, 1, 0, 1,
        1, 1, 0, 1, 1,
        1, 0, 0, 1, 0
    ]
}

df = pd.DataFrame(data)


# ============================================================
# 3. SAVE ORIGINAL DATASET
# ============================================================

df.to_csv(
    "results/student_performance.csv",
    index=False
)


# ============================================================
# 4. INTRODUCE MISSING DATA
# ============================================================

df.loc[3, "Attendance"] = np.nan
df.loc[8, "Study_Hours"] = np.nan
df.loc[12, "Test_Score"] = np.nan


print("\n============================================================")
print("STUDENT PERFORMANCE DATA")
print("============================================================")

print(df)


# ============================================================
# 5. CHECK MISSING DATA
# ============================================================

print("\n============================================================")
print("MISSING DATA")
print("============================================================")

print(df.isnull().sum())


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

# Average score from Assignment and Test scores

df["Average_Score"] = (
    df["Assignment_Score"] +
    df["Test_Score"]
) / 2


# Study efficiency feature

df["Study_Efficiency"] = (
    df["Average_Score"] /
    df["Study_Hours"]
)


# Replace infinite values with NaN

df["Study_Efficiency"] = df[
    "Study_Efficiency"
].replace(
    [np.inf, -np.inf],
    np.nan
)


print("\n============================================================")
print("FEATURE ENGINEERING")
print("============================================================")

print(df)


# ============================================================
# 7. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "Attendance",
    "Assignment_Score",
    "Test_Score",
    "Study_Hours",
    "Previous_Grade",
    "Average_Score",
    "Study_Efficiency"
]

X = df[features]

y = df["Passed"]


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


print("\n============================================================")
print("TRAIN / TEST SPLIT")
print("============================================================")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 9. PREPROCESSING
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

scaler = StandardScaler()


# ============================================================
# 10. RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 11. PIPELINE
# ============================================================

pipeline = Pipeline([
    ("imputer", imputer),
    ("scaler", scaler),
    ("model", model)
])


# ============================================================
# 12. TRAIN MODEL
# ============================================================

pipeline.fit(
    X_train,
    y_train
)


print("\n============================================================")
print("MODEL TRAINING")
print("============================================================")

print(
    "Random Forest model trained successfully."
)


# ============================================================
# 13. PREDICTIONS
# ============================================================

y_pred = pipeline.predict(
    X_test
)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 14. MODEL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# 15. DISPLAY MODEL EVALUATION
# ============================================================

print("\n============================================================")
print("MODEL EVALUATION RESULTS")
print("============================================================")

print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1 Score  : {f1:.2f}")
print(f"ROC-AUC   : {roc_auc:.2f}")


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\n============================================================")
print("CLASSIFICATION REPORT")
print("============================================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Failed",
            "Passed"
        ],
        zero_division=0
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n============================================================")
print("CONFUSION MATRIX")
print("============================================================")

print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Failed",
        "Passed"
    ]
)

disp.plot()

plt.title(
    "Confusion Matrix - Student Performance Model"
)

plt.tight_layout()

# SAVE CONFUSION MATRIX
plt.savefig(
    "graphs/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 18. CROSS-VALIDATION
# ============================================================

print("\n============================================================")
print("5-FOLD CROSS-VALIDATION")
print("============================================================")

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print(
    "Cross-validation scores:"
)

for i, score in enumerate(
    cv_scores,
    start=1
):
    print(
        f"Fold {i}: {score:.2f}"
    )

print(
    f"\nMean Cross-Validation Accuracy: "
    f"{cv_scores.mean():.2f}"
)


# ============================================================
# 19. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.2f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Student Performance Model"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

# SAVE ROC CURVE
plt.savefig(
    "graphs/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 20. FEATURE IMPORTANCE
# ============================================================

trained_model = pipeline.named_steps[
    "model"
]

importance = (
    trained_model.feature_importances_
)


feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})


feature_importance = (
    feature_importance.sort_values(
        by="Importance",
        ascending=False
    )
)


print("\n============================================================")
print("FEATURE IMPORTANCE")
print("============================================================")

print(
    feature_importance
)


plt.figure(
    figsize=(9, 6)
)

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Feature Importance - Student Performance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

# SAVE FEATURE IMPORTANCE GRAPH
plt.savefig(
    "graphs/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 21. SAVE MODEL METRICS
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Mean Cross-Validation Accuracy"
    ],

    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        cv_scores.mean()
    ]
})


metrics.to_csv(
    "results/model_metrics.csv",
    index=False
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("WEEK 9 & 10 PROJECT SUMMARY")
print("============================================================")

print(
    "✓ Missing data handled using median imputation"
)

print(
    "✓ Feature engineering completed"
)

print(
    "✓ Feature scaling completed"
)

print(
    "✓ Train/Test split completed"
)

print(
    "✓ Random Forest model trained"
)

print(
    "✓ Confusion Matrix generated and saved"
)

print(
    "✓ Precision calculated"
)

print(
    "✓ Recall calculated"
)

print(
    "✓ F1 Score calculated"
)

print(
    "✓ ROC-AUC calculated"
)

print(
    "✓ 5-Fold Cross-Validation completed"
)

print(
    "✓ ROC Curve generated and saved"
)

print(
    "✓ Feature Importance generated and saved"
)

print(
    "✓ Model metrics saved to CSV"
)

print("\n============================================================")
print(
    "WEEK 9 & 10 MODEL EVALUATION COMPLETED SUCCESSFULLY!"
)
print("============================================================")
