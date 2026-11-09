# Real Estate Property Price Prediction - Focused Features

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Step 2: Load Dataset
df = pd.read_csv("Regression/data/real_estate_data.csv")

# Step 3: Keep only the most impactful features + target
selected_features = ['Estimated Value', 'Residential', 'num_rooms', 'Property', 'carpet_area']
df_model = df[selected_features + ['Sale Price']].copy()

# Step 4: Handle missing values safely
df_model = df_model.fillna({
    'Estimated Value': df_model['Estimated Value'].median(),
    'carpet_area': df_model['carpet_area'].median(),
    'Property': '?',
    'Residential': 'Detached House'
})

print("Missing values after cleaning:\n", df_model.isnull().sum())
print(df_model.head())

# Step 5: Winsorization for outliers
df_model['Estimated Value'] = winsorize(df_model['Estimated Value'], limits=[0.01, 0.01])
df_model['carpet_area'] = winsorize(df_model['carpet_area'], limits=[0.01, 0.01])

# Step 6: Define features and target
X = df_model.drop('Sale Price', axis=1)
y = df_model['Sale Price']

# Step 7: Separate categorical and numerical features
categorical_features = ['Residential', 'Property']
numerical_features = ['Estimated Value', 'num_rooms', 'carpet_area']

# Step 8: Manual Label Encoding using .map
property_map = {'Single Family': 0, '?': 1, 'Two Family': 2, 'Three Family': 3, 'Four Family': 4}
residential_map = {'Detached House': 0, 'Duplex': 1, 'Triplex': 2, 'Fourplex': 3}

X_cat = pd.DataFrame({
    'Property': X['Property'].map(property_map),
    'Residential': X['Residential'].map(residential_map)
})
print("Categorical features after mapping:\n", X_cat.head())

# Step 9: Scale numerical features
scaler = StandardScaler()
X_num = pd.DataFrame(
    scaler.fit_transform(X[numerical_features]),
    columns=numerical_features,
    index=X.index
)

# Step 10: Combine numerical and categorical features
X_processed = pd.concat([X_num, X_cat], axis=1)

# Step 11: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Step 12: Build Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Step 13: Predictions
y_pred = lr.predict(X_test)

# Step 14: Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nLinear Regression MSE: {mse:.2f}")
print(f"Linear Regression RMSE: {rmse:.3f}")
print(f"Linear Regression R² Score: {r2:.3f}")
print(f"Linear Regression MAE: {mae:.3f}")

# Step 15: Feature Importance
coeff_df = pd.DataFrame({
    'Feature': X_processed.columns,
    'Coefficient': lr.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nTop features affecting Sale Price:")
print(coeff_df)
