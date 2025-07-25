FEATURE_ENGINEERING_FOR_DIABETES
Transforming Data into Life-Saving Insights

<p align="center">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/last%2520commit-July%25202025-blue" alt="Last Commit">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/python-100.0%2525-blue" alt="Python">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/languages-1-blue" alt="Languages">
</p>

<p align="center">
Built with the tools and technologies:
</p>

<p align="center">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Markdown-000000%3Fstyle%3Dfor-the-badge%26logo%3Dmarkdown%26logoColor%3Dwhite" alt="Markdown">
<img src="https://www.google.com/search?q=https://img.shields.io/badge/Python-3776AB%3Fstyle%3Dfor-the-badge%26logo%3Dpython%26logoColor%3Dwhite" alt="Python">
</p>

Table of Contents
Overview

Getting Started

Prerequisites

Installation

Project Workflow

Usage

Testing

Overview
Feature_Engineering_for_Diabetes is a comprehensive developer tool that simplifies the process of preparing clinical data for diabetes prediction models. It enables efficient data exploration, visualization, and preprocessing, ensuring your datasets are clean and model-ready.

Why Feature_Engineering_for_Diabetes?
This project aims to facilitate accurate health risk assessments through robust data handling. The core features include:

📝 Data Exploration: Visualize and understand data distributions to identify patterns and anomalies.

🧪 Missing Value Handling: Detect and address gaps in clinical datasets to improve model reliability.

⚙️ Feature Engineering & Scaling: Create impactful new features from existing data and normalize them for optimal model performance.

📊 Model Comparison: Train and evaluate multiple machine learning algorithms (including Random Forest, LGBM, and XGBoost) to identify the best-performing model.

Getting Started
Prerequisites
This project requires the following dependencies:

Programming Language: Python

Package Manager: Conda

Installation
Build Feature_Enginnering_for_Diabetes from the source and install dependencies:

Clone the repository:

> git clone [https://github.com/bariskurt40/Feature_Enginnering_for_Diabetes](https://github.com/bariskurt40/Feature_Enginnering_for_Diabetes)

Navigate to the project directory:

> cd Feature_Enginnering_for_Diabetes

Install the dependencies:
(Note: You will need to create a conda.yml file with the project's dependencies for this command to work.)

Using conda:

> conda env create -f conda.yml

Project Workflow
The script Diabetes.py follows a systematic pipeline to process the data and build predictive models:

Data Preprocessing: Handles missing values (where 0 indicates missing data in columns like Glucose and BMI), caps outliers using quantile-based thresholds, and scales numerical features using RobustScaler.

Feature Engineering: The core of the project involves creating new, insightful features. This includes binning Age, Glucose, and BMI into categorical variables and creating interaction features like GLUCOSE_INSULIN and INSULIN_BMI to capture complex relationships.

Model Training & Evaluation: Trains several classification models (Random Forest, LGBM, XGBoost, Logistic Regression, KNN, Decision Tree) and compares their performance using metrics like Accuracy, Precision, Recall, and F1-Score to find the most effective model.

Usage
Run the project with:

Using conda:

conda activate diabetes_env
python Diabetes.py

Testing
This project can use the pytest test framework. To run a test suite:

Using conda:

conda activate diabetes_env
pytest

<br>

<p align="right"><a href="#top">⬆️ Return</a></p>
