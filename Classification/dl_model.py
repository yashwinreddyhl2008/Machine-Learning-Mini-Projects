
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Set random seeds for reproducibility
tf.keras.utils.set_random_seed(42)
random.seed(42)
np.random.seed(42)

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

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Data Shape:", X_train_scaled.shape)
print("Testing Data Shape:", X_test_scaled.shape)

# Build ANN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display model architecture
model.summary()

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# Train ANN model
print("\nTraining ANN Model...")

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Make predictions
y_probability = model.predict(X_test_scaled)

y_pred = (y_probability >= 0.5).astype(int).ravel()

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Display results
print("\nANN CLASSIFICATION RESULTS")
print("--------------------------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Classification report
report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\nClassification Report:")
print(report)

# Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=["Class 0", "Class 1"],
    yticklabels=["Class 0", "Class 1"]
)

plt.title("ANN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "ann_confusion_matrix.png",
    dpi=300
)

plt.close()

# Save confusion matrix as CSV
cm_df = pd.DataFrame(
    cm,
    index=["Actual Class 0", "Actual Class 1"],
    columns=["Predicted Class 0", "Predicted Class 1"]
)

cm_df.to_csv(RESULTS_DIR / "ann_confusion_matrix.csv")

# Plot training and validation accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("ANN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "ann_accuracy.png",
    dpi=300
)

plt.close()

# Plot training and validation loss
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("ANN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "ann_loss.png",
    dpi=300
)

plt.close()

# Save trained model
model.save(RESULTS_DIR / "phishing_ann.keras")

# Save scaler
joblib.dump(
    scaler,
    RESULTS_DIR / "ann_scaler.pkl"
)

# Save evaluation metrics as CSV
results = pd.DataFrame({
    "Model": ["Artificial Neural Network"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1]
})

results.to_csv(
    RESULTS_DIR / "ann_results.csv",
    index=False
)

# Save detailed results as TXT
with open(
    RESULTS_DIR / "ann_results.txt",
    "w"
) as file:

    file.write("PROJECT: Phishing Website Detection\n")
    file.write("MODEL: Artificial Neural Network\n\n")

    file.write("DATASET INFORMATION:\n")
    file.write(f"Total Records: {len(df)}\n")
    file.write(f"Training Samples: {len(X_train)}\n")
    file.write(f"Testing Samples: {len(X_test)}\n")
    file.write(f"Input Features: {X.shape[1]}\n\n")

    file.write("MODEL ARCHITECTURE:\n")
    file.write(f"Input Layer: {X.shape[1]} features\n")
    file.write("Hidden Layer 1: 128 neurons (ReLU)\n")
    file.write("Dropout: 0.2\n")
    file.write("Hidden Layer 2: 64 neurons (ReLU)\n")
    file.write("Dropout: 0.2\n")
    file.write("Hidden Layer 3: 32 neurons (ReLU)\n")
    file.write("Output Layer: 1 neuron (Sigmoid)\n\n")

    file.write("TRAINING DETAILS:\n")
    file.write("Optimizer: Adam\n")
    file.write("Loss: Binary Crossentropy\n")
    file.write("Maximum Epochs: 100\n")
    file.write("Batch Size: 32\n")
    file.write("Early Stopping: Enabled\n\n")

    file.write("MODEL PERFORMANCE:\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1 Score: {f1:.4f}\n\n")

    file.write("CLASSIFICATION REPORT:\n")
    file.write(report)

    file.write("\nCONFUSION MATRIX:\n")
    file.write(str(cm))

print("\nANN model saved successfully.")
print("Scaler saved successfully.")
print("Evaluation metrics saved successfully.")
print("Confusion matrix saved successfully.")
print("Training graphs saved successfully.")
