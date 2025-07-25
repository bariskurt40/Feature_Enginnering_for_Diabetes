<div align="center">

# FEATURE ENGINEERING FOR DIABETES PREDICTION

### *Transforming Data into Actionable Health Insights*

</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.x-orange?style=for-the-badge&logo=scikit-learn" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/LightGBM-4.x-purple?style=for-the-badge&logo=lightgbm" alt="LightGBM">
  <img src="https://img.shields.io/badge/XGBoost-2.x-green?style=for-the-badge&logo=xgboost" alt="XGBoost">
</p>

<p align="center">
  An end-to-end pipeline demonstrating how advanced feature engineering can dramatically improve machine learning model accuracy.
</p>

---

### 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Expected Output](#-expected-output)
- [Interpreting the Results](#-interpreting-the-results)

---

### 📖 Overview

This project provides a comprehensive workflow for predicting diabetes using the PIMA Indians Diabetes Dataset. It goes beyond simple model training by placing a strong emphasis on a meticulous **Feature Engineering** process. The goal is to demonstrate how data cleaning, outlier management, and the creation of new, insightful variables can lead to a more robust and accurate prediction model.

---

### ✨ Key Features

The core capabilities offered by this project are:

* **🧠 Advanced Feature Engineering:** The script intelligently handles physiologically impossible `0` values, caps outliers, and creates powerful new categorical (`AGE_NEW`, `BMI_NEW`) and interaction (`GLUCOSE_INSULIN`) features.

* **⚙️ Multi-Model Comparison:** It trains, evaluates, and compares a suite of powerful classification algorithms, including `Random Forest`, `LightGBM`, `XGBoost`, and `KNN`, to identify the best-performing model for the task.

* **📊 Robust Preprocessing:** The pipeline uses `RobustScaler` to handle outliers during scaling and employs a mix of `Label Encoding` and `One-Hot Encoding` to prepare categorical data for modeling.

* **📈 Feature Importance Analysis:** It automatically generates visualizations that show which features—both original and newly engineered—are most predictive, providing deep insights into the factors contributing to diabetes.

---

### 🚀 Installation

Follow the steps below to run the project on your local machine.

#### Prerequisites
* Python 3.8+
* Pip Package Manager

#### Steps

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/Feature_Enginnering_for_Diabetes-main.git](https://github.com/your-username/Feature_Enginnering_for_Diabetes-main.git)
    ```

2.  **Navigate to the project directory:**
    ```sh
    cd Feature_Enginnering_for_Diabetes-main
    ```

3.  **Install the required libraries:**
    ```sh
    pip install pandas numpy scikit-learn matplotlib seaborn missingno lightgbm xgboost
    ```

---

### 💻 Usage

The entire workflow is automated within a single script. Ensure the `diabetes.csv` file is located in the `datasets` folder before running.

* **Execute the main script from your terminal:**
    ```sh
    python Diabetes.py
    ```
This command will trigger the complete pipeline: data loading, cleaning, feature engineering, model training, evaluation, and visualization of results.

---

### 📊 Expected Output

Running the script will produce performance metrics and feature importance rankings in your console. Below is a sample of what to expect.

#### Model Performance
The script compares six different models and identifies the one with the highest **accuracy**.

**Model Comparison Results (Sample Output):**

| Model | Accuracy Score |
| :--- | :---: |
| **LGBMClassifier** | **0.88** |
| RandomForestClassifier | 0.87 |
| XGBClassifier | 0.86 |
| KNeighborsClassifier | 0.84 |
| LogisticRegression | 0.82 |
| DecisionTreeClassifier | 0.79 |

A detailed report for the `RandomForestClassifier` is also generated:

* **Accuracy:** 0.87
* **Precision:** 0.85
* **Recall:** 0.88
* **F1 Score:** 0.86
* **AUC:** 0.87

#### Feature Importance
A plot is generated to show which features are most influential in the prediction.

**Feature Importance Ranking (Sample Output):**

| Rank | Feature Name | Importance (Value) |
| :---: | :--- | :---: |
| 1 | `GLUCOSE` | High |
| 2 | `AGE` | High |
| 3 | `BMI` | High |
| 4 | `GLUCOSE_INSULIN` (New) | Medium |
| 5 | `INSULIN` | Medium |
| 6 | `DiabetesPedigreeFunction`| Medium |
| 7 | `BLOODPRESSURE` | Low |
| 8 | `AGE_BMI_seniorOverweight` (New) | Low |
| 9 | `SKINTHICKNESS` | Low |
| 10 | `Pregnancies` | Low |


---

### 🧠 Interpreting the Results

The script's output provides two primary forms of insight:

* **Model Selection:** The direct accuracy comparison helps in choosing the most effective algorithm for this specific, feature-engineered dataset.
* **Domain Insight:** The feature importance plot reveals the key drivers of diabetes prediction. When newly created features (like `GLUCOSE_INSULIN`) rank highly, it validates the success of the feature engineering process and uncovers complex relationships in the data.
