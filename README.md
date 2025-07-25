# Diabetes Prediction Model

This repository contains a Python script for building and evaluating machine learning models to predict the onset of diabetes based on diagnostic measures. The project involves comprehensive data preprocessing, feature engineering, and a comparative analysis of various classification algorithms.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Model Limitations and Potential Biases](#model-limitations-and-potential-biases)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Description

The primary goal of this project is to predict whether a patient has diabetes based on certain diagnostic medical predictor variables. The provided Python script, `Diabetes.py`, performs the following key steps:

1.  **Exploratory Data Analysis (EDA)**: The script begins by exploring the dataset to understand its structure, identify missing values, and analyze the distribution of different features.
2.  **Data Preprocessing**: It handles missing data by replacing zero values in specific columns with the mean of that feature, grouped by the outcome variable. It also addresses outliers by replacing them with calculated threshold values.
3.  **Feature Engineering**: New features are created to potentially improve model performance. This includes creating categorical age and BMI groups, as well as interaction features between different variables.
4.  **Model Training and Comparison**: The script trains and evaluates several machine learning models to find the best-performing one for this prediction task. The models used are:
    * Random Forest
    * LightGBM
    * Decision Tree
    * Logistic Regression
    * XGBoost
    * K-Nearest Neighbors (KNN)
5.  **Feature Importance**: The script also visualizes the importance of different features in the prediction, providing insights into which factors are most influential.

---

## Dataset

This project utilizes the **Pima Indians Diabetes Database**.

* **Source**: The dataset originates from the National Institute of Diabetes and Digestive and Kidney Diseases. It is a well-known dataset in the machine learning community and is available from platforms like Kaggle.

### Features

The dataset includes the following features:

* **Pregnancies**: Number of times pregnant
* **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
* **BloodPressure**: Diastolic blood pressure (mm Hg)
* **SkinThickness**: Triceps skin fold thickness (mm)
* **Insulin**: 2-Hour serum insulin (mu U/ml)
* **BMI**: Body mass index (weight in kg/(height in m)^2)
* **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history.
* **Age**: Age in years
* **Outcome**: The target variable, indicating whether the patient has diabetes (1) or not (0).

### Data Format Example

Here's a sample of what the data looks like in the `.csv` file: Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1


---

## Setup and Installation

To run this project, you'll need Python 3 and several libraries.

### Prerequisites

* Python 3.6 or higher. You can download it from [python.org](https://www.python.org/downloads/).

### Dependencies

You can install all the necessary libraries using `pip`. It is recommended to use a virtual environment.

1.  Create a file named `requirements.txt` and paste the following content into it:

    ```
    numpy
    pandas
    seaborn
    matplotlib
    missingno
    scikit-learn
    lightgbm
    xgboost
    ```

2.  Install the dependencies by running the following command in your terminal:

    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

To run the diabetes prediction script, navigate to the project's root directory in your terminal and execute the `Diabetes.py` script.

### Running the Script

```bash
python Feature_Enginnering_for_Diabetes-main/Diabetes.py
