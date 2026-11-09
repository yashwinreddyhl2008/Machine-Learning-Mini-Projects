
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Phishing_Legitimate_full.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop(["id", "CLASS_LABEL"], axis=1)
y = df["CLASS_LABEL"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Train model
print("Training Random Forest Classifier...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nRANDOM FOREST CLASSIFIER RESULTS")
print("--------------------------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Class 0", "Class 1"],
    yticklabels=["Class 0", "Class 1"]
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(RESULTS_DIR / "confusion_matrix.png")
plt.close()

# Save trained model
joblib.dump(model, RESULTS_DIR / "random_forest_model.pkl")

# Save results
with open(RESULTS_DIR / "baseline_results.txt", "w") as file:
    file.write("PROJECT: Phishing Website Detection\n")
    file.write("MODEL: Random Forest Classifier\n\n")
    file.write("DATASET:\n")
    file.write(f"Total Records: {len(df)}\n")
    file.write(f"Training Samples: {len(X_train)}\n")
    file.write(f"Testing Samples: {len(X_test)}\n")
    file.write(f"Input Features: {X.shape[1]}\n\n")
    file.write("MODEL PERFORMANCE:\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1 Score: {f1:.4f}\n\n")
    file.write("CLASSIFICATION REPORT:\n")
    file.write(classification_report(y_test, y_pred))

print("\nConfusion matrix saved successfully.")
print("Trained model saved successfully.")
print("Results saved successfully.")
