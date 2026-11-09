import joblib
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    X, y, test_size=0.2, random_state=42
)

# Numerical preprocessing
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Create Random Forest pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nRANDOM FOREST REGRESSION RESULTS")
print("--------------------------------")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)

# Create results folder if it doesn't exist
os.makedirs("Regression/results", exist_ok=True)

joblib.dump(model, "Regression/results/random_forest_model.pkl")
# Store evaluation metrics
results = pd.DataFrame({
    "Model": ["Random Forest Regressor"],
    "MSE": [mse],
    "RMSE": [rmse],
    "MAE": [mae],
    "R2 Score": [r2]
})

# Save results as CSV
results.to_csv("Regression/results/random_forest_results.csv", index=False)

# Save results as text file
with open("Regression/results/random_forest_results.txt", "w") as file:
    file.write("RANDOM FOREST REGRESSION RESULTS\n")
    file.write("--------------------------------\n")
    file.write(f"MSE: {mse}\n")
    file.write(f"RMSE: {rmse}\n")
    file.write(f"MAE: {mae}\n")
    file.write(f"R2 Score: {r2}\n")

print("\nResults saved successfully!")