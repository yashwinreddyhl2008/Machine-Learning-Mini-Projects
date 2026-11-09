import joblib
import os
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set random seed for reproducibility
tf.keras.utils.set_random_seed(42)

# Load dataset
df = pd.read_csv("Regression/data/real_estate_data.csv")

# Handle unknown property category
df["Property"] = df["Property"].replace("?", pd.NA)

# Select features
numerical_features = [
    "Year",
    "Estimated Value",
    "num_rooms",
    "num_bathrooms",
    "carpet_area",
    "property_tax_rate"
]

categorical_features = [
    "Locality",
    "Property",
    "Residential",
    "Face"
]

X = df[numerical_features + categorical_features]
y = df["Sale Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Numerical preprocessing
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data
X_test_processed = preprocessor.transform(X_test)

# Scale target values
target_scaler = StandardScaler()

y_train_scaled = target_scaler.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
)

y_test_scaled = target_scaler.transform(
    y_test.to_numpy().reshape(-1, 1)
)

# Build Artificial Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(X_train_processed.shape[1],)
    ),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),

    # Output layer for regression
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
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
history = model.fit(
    X_train_processed,
    y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Make predictions
y_pred_scaled = model.predict(X_test_processed)

# Convert predictions back to original price scale
y_pred = target_scaler.inverse_transform(y_pred_scaled)

# Convert actual test values to original scale
y_test_original = y_test.to_numpy().reshape(-1, 1)

# Calculate evaluation metrics
mse = mean_squared_error(y_test_original, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)

# Display performance
print("\nDEEP LEARNING MODEL PERFORMANCE")
print("--------------------------------")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)

# Display preprocessing information
print("\nPREPROCESSING RESULTS")
print("---------------------")
print("Training input shape:", X_train_processed.shape)
print("Testing input shape:", X_test_processed.shape)
print("Training target shape:", y_train_scaled.shape)
print("Testing target shape:", y_test_scaled.shape)

print("\nDATASET INFORMATION")
print("-------------------")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Create results folder
os.makedirs("Regression/results", exist_ok=True)

# Save trained ANN model
model.save("Regression/results/real_estate_ann.keras")

joblib.dump(target_scaler, "Regression/results/target_scaler.pkl")
joblib.dump(preprocessor, "Regression/results/preprocessor.pkl")

# Store evaluation metrics
results = pd.DataFrame({
    "Model": ["Artificial Neural Network"],
    "MSE": [mse],
    "RMSE": [rmse],
    "MAE": [mae],
    "R2 Score": [r2]
})

# Save results as CSV
results.to_csv(
    "Regression/results/ann_results.csv",
    index=False
)

# Save results as text
with open("Regression/results/ann_results.txt", "w") as file:
    file.write("DEEP LEARNING MODEL PERFORMANCE\n")
    file.write("--------------------------------\n")
    file.write(f"MSE: {mse}\n")
    file.write(f"RMSE: {rmse}\n")
    file.write(f"MAE: {mae}\n")
    file.write(f"R2 Score: {r2}\n")

print("\nANN model and evaluation results saved successfully!")
