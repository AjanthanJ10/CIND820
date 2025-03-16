# CIND 820: Big Data Analytics Project - Diabetes Prediction

## Overview
This repository contains all the necessary files and documentation for our machine learning project focused on diabetes prediction. The project involves comparing model performance with and without data balancing techniques, specifically SMOTE, and includes hyperparameter tuning using RandomizedSearchCV. We evaluate several models, including Logistic Regression, Random Forest, Decision Tree, CatBoost, Gradient Boosting, and XGBoost. The evaluation metrics used are accuracy, precision, recall, ROC AUC, PR AUC, and confusion matrix.

## Tentative Project Steps 
Data Preprocessing:
* Load and clean dataset
* Feature engineering and selection
* Standardization of numerical features

Training and Evaluation (Unbalanced Dataset)
* Train Logistic Regression, Random Forest, Decision Tree, CatBoost, Gradient Boosting, XGBoost
* Evaluate using accuracy, precision, recall, ROC AUC, PR AUC, and confusion matrix

Applying SMOTE for Data Balancing
* Train and evaluate the same models

Hyperparameter Tuning
* Perform RandomizedSearchCV for optimal hyperparameters
* Train and evaluate models with tuned parameters

Comparison and Analysis
* Compare performance metrics across different settings
* Summarize key insights and findings

## How to Run Project
### Install Depedencies 
! pip install pandas

! pip install numpy

! pip install matplotlib

! pip install seaborn

! pip install sklearn

! pip install xgboost

! pip install catboost

! pip install imbalanced-learn

### Run [InitialCodeplusResult.py>)](InitialCodeplusResult.py)

## CIND 820 - Repository Breakdown
[Diabetes Dataset][def6] This contains the dataset that we will be using to conduct our analysis.

[Exploratory Data Analysis Code][def5] The code we used to generate the EDA Report.

[Exploratory Data Analysis Code HTML][def4] The EDA Report we used to get a understanding of the dataset.

[Initial Code + Result Jupyter Notebook File][def3] The Jupyter Notebook we used to code our machine learning models and evaluation of the different models.

[Initial Code + Result HTML][def2] The HTML document which holds the code plus all the results of the machine learning models.

[Initial Code + Result .py File][def] The .py file of the code.

[README][def] The README file.

[def]: InitialCodeplusResult.py
[def2]: Initial%20Code%20plus%20results.html
[def3]: Initial%20Code%20plus%20results.ipynb
[def4]: EDA_Analysis_Report.html
[def5]: EDA
[def6]: diabetes_012_health_indicators_BRFSS2015.csv
[def7]: README.md