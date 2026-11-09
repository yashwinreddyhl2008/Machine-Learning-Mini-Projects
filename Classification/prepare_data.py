
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Phishing_Legitimate_full.csv"

df = pd.read_csv(DATA_PATH)

# Separate input features and target
X = df.drop(["id", "CLASS_LABEL"], axis=1)
y = df["CLASS_LABEL"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display dataset information
print("Dataset Preparation Completed!")

print("\nOriginal Dataset:", df.shape)
print("Input Features:", X.shape)
print("Target:", y.shape)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())
