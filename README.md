# Machine Learning Mini Projects

A collection of two machine learning mini projects developed as part of the academic curriculum at **Government Polytechnic, Channasandra**, Department of Computer Science and Engineering.

This repository demonstrates the application of **Machine Learning and Deep Learning** techniques to real-world prediction and classification problems.

---

## Projects Overview

### 1. Real Estate Property Price Prediction

**Problem Type:** Regression

A machine learning project that predicts property sale prices based on various property-related features.

**Dataset:** `real_estate_data.csv`

**Dataset Details:**
- 10,000 records
- 12 columns
- Target variable: Sale Price
- Features include locality, estimated value, residential status, number of rooms, property type and carpet area.

**Algorithms Implemented:**
- Linear Regression (Baseline)
- Improved Linear Regression
- Random Forest Regressor
- Artificial Neural Network (ANN)

**Best Model: Random Forest Regressor**

| Evaluation Metric | Result |
|---|---:|
| Mean Squared Error (MSE) | 250,819,168,236.43 |
| Root Mean Squared Error (RMSE) | 500,818.50 |
| Mean Absolute Error (MAE) | 184,681.14 |
| R² Score | 0.713514 |

The Random Forest Regressor achieved the best overall performance among the evaluated models.

---

### 2. Phishing Website Detection Using Machine Learning and Deep Learning

**Problem Type:** Classification

A classification project designed to distinguish between phishing and legitimate websites using website-related features.

**Dataset:** `Phishing_Legitimate_full.csv`

**Dataset Details:**
- 10,000 records
- 50 columns
- 48 input features, excluding the ID and target columns
- Target variable: `CLASS_LABEL`
- Balanced dataset with 5,000 records per class.

**Algorithms Implemented:**
- Random Forest Classifier
- Improved Random Forest using GridSearchCV
- Artificial Neural Network (ANN)

**Best Model: Random Forest Classifier**

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 98.50% |
| Precision | 98.89% |
| Recall | 98.10% |
| F1-Score | 98.49% |

**Confusion Matrix:**

| | Predicted Class 0 | Predicted Class 1 |
|---|---:|---:|
| Actual Class 0 | 989 | 11 |
| Actual Class 1 | 19 | 981 |

The Random Forest model achieved 98.50% accuracy on the test dataset.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

## Machine Learning Concepts Applied

- Data Preprocessing
- Handling Missing Values
- Feature Encoding
- Feature Scaling
- Train-Test Splitting
- Regression Analysis
- Classification
- Random Forest
- Linear Regression
- Artificial Neural Networks
- Hyperparameter Tuning using GridSearchCV
- Model Evaluation
- Confusion Matrix

## Repository Structure

```text
Machine-Learning-Mini-Projects/
│
├── Regression/
│   ├── Dataset/
│   ├── results/
│   └── ...
│
├── Classification/
│   ├── Dataset/
│   ├── results/
│   └── ...
│
└── README.md
```

*Note: The folder structure above is a simplified overview. Refer to the individual project folders for the complete files.*

## Project Objectives

1. Understand practical applications of machine learning algorithms.
2. Perform data preprocessing and feature engineering.
3. Implement and compare multiple machine learning models.
4. Explore deep learning using Artificial Neural Networks.
5. Evaluate models using appropriate performance metrics.
6. Identify the best-performing model for each problem.

## Academic Information

| Detail | Information |
|---|---|
| Institution | Government Polytechnic, Channasandra |
| Department | Computer Science and Engineering |
| Academic Year | 2026–2027 |
| Student | Yashwin Reddy H. L. |
| Registration Number | 183CS24046 |
| Project Guide | Mr O K
