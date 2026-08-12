# 🚢 Titanic Survival Prediction – Machine Learning Pipeline

## 📌 Project Overview

This project focuses on building a clean, reusable, and professional Machine Learning workflow for predicting passenger survival using the **Titanic dataset**.

The main objective of this task is to demonstrate how **Scikit-learn Pipelines** can combine data preprocessing and model training into a single workflow. This approach helps maintain consistency between training and testing data and reduces the risk of preprocessing errors and data leakage.

The project includes both a **manual preprocessing approach** and a **Pipeline-based approach** for comparison.

---

## 🎯 Objectives

The main objectives of this project are:

- Build a reusable Machine Learning Pipeline.
- Use `ColumnTransformer` for preprocessing.
- Apply `StandardScaler` to numerical features.
- Apply `OneHotEncoder` to categorical features.
- Combine preprocessing and Logistic Regression into one Pipeline.
- Create at least two engineered features.
- Compare model performance with and without engineered features.
- Compare the manual preprocessing approach with the Pipeline approach.
- Evaluate the model using multiple classification metrics.
- Save and reload the final trained Pipeline using Joblib.

---

## 📊 Dataset

The project uses the **Titanic dataset**, which contains information about passengers aboard the Titanic.

### Target Variable

- `Survived` — Indicates whether the passenger survived.

### Features Used

#### Numerical Features

- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `FamilySize`
- `IsAlone`

#### Categorical Features

- `Pclass`
- `Sex`
- `Embarked`

---

## 🛠️ Technologies & Libraries

The project was developed using Python and the following libraries:

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Scikit-learn Components

- `train_test_split`
- `ColumnTransformer`
- `Pipeline`
- `StandardScaler`
- `OneHotEncoder`
- `SimpleImputer`
- `LogisticRegression`
- Classification metrics

---

## 🔧 Feature Engineering

Two new features were created to improve the representation of passenger information.

### 1. FamilySize

`FamilySize` represents the total number of family members traveling with the passenger.

```python
FamilySize = SibSp + Parch + 1
