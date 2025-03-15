# %% [markdown]
# # CIND 820: Big Data Analytics Project - Initial Results and the Code

# %% [markdown]
# ## Importing Libraries

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve

# %% [markdown]
# ## Loading Dataset, Creating Features and Target, and Feature Selection

# %%
# Load your dataset
file_path = "/Users/ajanthanjoseph/Documents/GitHub/CIND820/diabetes_012_health_indicators_BRFSS2015.csv"
df = pd.read_csv(file_path)

# Create interaction features
df["BMI_Age"] = df["BMI"] * df["Age"]
df["HighBP_HighChol"] = df["HighBP"] * df["HighChol"]
df["PhysActivity_BMI"] = df["PhysActivity"] * df["BMI"]

# Separate features and target
X = df.drop(columns=["Diabetes_012"])
y = df["Diabetes_012"]

# Apply Chi-Squared test for feature selection
chi2_selector = SelectKBest(score_func=chi2, k='all')
chi2_selector.fit(X, y)
chi2_scores = chi2_selector.scores_

# Apply ANOVA F-test for feature selection
anova_selector = SelectKBest(score_func=f_classif, k='all')
anova_selector.fit(X, y)
anova_scores = anova_selector.scores_

# Create a DataFrame with the feature selection results
feature_selection_results = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': chi2_scores,
    'ANOVA F Score': anova_scores
}).sort_values(by=['Chi2 Score', 'ANOVA F Score'], ascending=False)

# Display Chi-Square and ANOVA scores
print("Chi-Square and ANOVA F-test Scores:")
print(feature_selection_results)

# Identify the 6 lowest scoring features based on the combined scores
lowest_features = feature_selection_results.nsmallest(6, ['Chi2 Score', 'ANOVA F Score'])['Feature'].tolist()

# Display the 6 lowest-scoring features to be dropped
print("\n6 Lowest-Scoring Features to be Dropped:")
print(lowest_features)

# Drop these features from the dataset
X_reduced = X.drop(columns=lowest_features)

# Display the final 15 features used
print("\nFinal 15 Features Used:")
print(X_reduced.columns.tolist())

# Standardize numerical columns
numerical_features = ["BMI", "Age", "BMI_Age", "PhysActivity_BMI"]
scaler = StandardScaler()
X_reduced[numerical_features] = scaler.fit_transform(X_reduced[numerical_features])

# %% [markdown]
# ## Splitting unbalanced dataset into Training and Test Sets

# %%
# Split the original (unbalanced) dataset into training and testing sets
X_train_unbalanced, X_test_unbalanced, y_train_unbalanced, y_test_unbalanced = train_test_split(
    X_reduced, y, test_size=0.3, random_state=42, stratify=y
)

# Display the shapes of the resulting datasets
print("Training and Testing Dataset Shapes (Unbalanced):")
print(f"X_train_unbalanced: {X_train_unbalanced.shape}, y_train_unbalanced: {y_train_unbalanced.shape}")
print(f"X_test_unbalanced: {X_test_unbalanced.shape}, y_test_unbalanced: {y_test_unbalanced.shape}")

# %% [markdown]
# # Unbalanced Data Machine Learning Models + Evaluation of Models

# %% [markdown]
# ## Logistic Regression and Evaluation

# %%
# Logistic Regression
lr_unbalanced = LogisticRegression(max_iter=1000)
lr_unbalanced.fit(X_train_unbalanced, y_train_unbalanced)
y_pred_unbalanced = lr_unbalanced.predict(X_test_unbalanced)

# Evaluation
accuracy = accuracy_score(y_test_unbalanced, y_pred_unbalanced)
precision = precision_score(y_test_unbalanced, y_pred_unbalanced, average='weighted')
recall = recall_score(y_test_unbalanced, y_pred_unbalanced, average='weighted')
roc_auc = roc_auc_score(y_test_unbalanced, lr_unbalanced.predict_proba(X_test_unbalanced), multi_class='ovr')
pr_auc = average_precision_score(y_test_unbalanced, lr_unbalanced.predict_proba(X_test_unbalanced), average='weighted')
conf_matrix = confusion_matrix(y_test_unbalanced, y_pred_unbalanced)

print("Logistic Regression Evaluation (Unbalanced Data):")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# ## Random Forest and Evaluation

# %%
# Random Forest
rf_unbalanced = RandomForestClassifier(random_state=42)
rf_unbalanced.fit(X_train_unbalanced, y_train_unbalanced)
y_pred_unbalanced = rf_unbalanced.predict(X_test_unbalanced)

# Evaluation
accuracy = accuracy_score(y_test_unbalanced, y_pred_unbalanced)
precision = precision_score(y_test_unbalanced, y_pred_unbalanced, average='weighted')
recall = recall_score(y_test_unbalanced, y_pred_unbalanced, average='weighted')
roc_auc = roc_auc_score(y_test_unbalanced, rf_unbalanced.predict_proba(X_test_unbalanced), multi_class='ovr')
pr_auc = average_precision_score(y_test_unbalanced, rf_unbalanced.predict_proba(X_test_unbalanced), average='weighted')
conf_matrix = confusion_matrix(y_test_unbalanced, y_pred_unbalanced)

print("Random Forest Evaluation (Unbalanced Data):")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# ## Decision Tree and Evaluation

# %%
# Decision Tree
dt_unbalanced = DecisionTreeClassifier(random_state=42)
dt_unbalanced.fit(X_train_unbalanced, y_train_unbalanced)
y_pred_unbalanced = dt_unbalanced.predict(X_test_unbalanced)

# Evaluation
accuracy = accuracy_score(y_test_unbalanced, y_pred_unbalanced)
precision = precision_score(y_test_unbalanced, y_pred_unbalanced, average='weighted')
recall = recall_score(y_test_unbalanced, y_pred_unbalanced, average='weighted')
roc_auc = roc_auc_score(y_test_unbalanced, dt_unbalanced.predict_proba(X_test_unbalanced), multi_class='ovr')
pr_auc = average_precision_score(y_test_unbalanced, dt_unbalanced.predict_proba(X_test_unbalanced), average='weighted')
conf_matrix = confusion_matrix(y_test_unbalanced, y_pred_unbalanced)

print("Decision Tree Evaluation (Unbalanced Data):")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# ## Catboost and Evaluation

# %%
# CatBoost
cb_unbalanced = CatBoostClassifier(random_state=42, verbose=0)
cb_unbalanced.fit(X_train_unbalanced, y_train_unbalanced)
y_pred_unbalanced = cb_unbalanced.predict(X_test_unbalanced)

# Evaluation
accuracy = accuracy_score(y_test_unbalanced, y_pred_unbalanced)
precision = precision_score(y_test_unbalanced, y_pred_unbalanced, average='weighted')
recall = recall_score(y_test_unbalanced, y_pred_unbalanced, average='weighted')
roc_auc = roc_auc_score(y_test_unbalanced, cb_unbalanced.predict_proba(X_test_unbalanced), multi_class='ovr')
pr_auc = average_precision_score(y_test_unbalanced, cb_unbalanced.predict_proba(X_test_unbalanced), average='weighted')
conf_matrix = confusion_matrix(y_test_unbalanced, y_pred_unbalanced)

print("CatBoost Evaluation (Unbalanced Data):")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# ## Gradient Boosting and Evaluation

# %%
# Gradient Boosting
gb_unbalanced = GradientBoostingClassifier(random_state=42)
gb_unbalanced.fit(X_train_unbalanced, y_train_unbalanced)
y_pred_unbalanced = gb_unbalanced.predict(X_test_unbalanced)

# Evaluation
accuracy = accuracy_score(y_test_unbalanced, y_pred_unbalanced)
precision = precision_score(y_test_unbalanced, y_pred_unbalanced, average='weighted')
recall = recall_score(y_test_unbalanced, y_pred_unbalanced, average='weighted')
roc_auc = roc_auc_score(y_test_unbalanced, gb_unbalanced.predict_proba(X_test_unbalanced), multi_class='ovr')
pr_auc = average_precision_score(y_test_unbalanced, gb_unbalanced.predict_proba(X_test_unbalanced), average='weighted')
conf_matrix = confusion_matrix(y_test_unbalanced, y_pred_unbalanced)

print("Gradient Boosting Evaluation (Unbalanced Data):")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# ## XGBoost and Evaluation

# %%
# XGBoost
xgb_unbalanced = XGBClassifier(random_state=42)
xgb_unbalanced.fit(X_train_unbalanced, y_train_unbalanced)
y_pred_unbalanced = xgb_unbalanced.predict(X_test_unbalanced)

# Evaluation
accuracy = accuracy_score(y_test_unbalanced, y_pred_unbalanced)
precision = precision_score(y_test_unbalanced, y_pred_unbalanced, average='weighted')
recall = recall_score(y_test_unbalanced, y_pred_unbalanced, average='weighted')
roc_auc = roc_auc_score(y_test_unbalanced, xgb_unbalanced.predict_proba(X_test_unbalanced), multi_class='ovr')
pr_auc = average_precision_score(y_test_unbalanced, xgb_unbalanced.predict_proba(X_test_unbalanced), average='weighted')
conf_matrix = confusion_matrix(y_test_unbalanced, y_pred_unbalanced)

print("XGBoost Evaluation (Unbalanced Data):")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# # Applying SMOTE to balance the target classes

# %% [markdown]
# # Machine Learning Models and Evaluation using the balanced data

# %%
# Apply SMOTE to balance the target classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_reduced, y)

# Check the distribution of the target variable after SMOTE
print(y_resampled.value_counts())

# Split the data into training and testing sets with a 70-30 split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)

# Display the shapes of the resulting datasets
print("Training and Testing Dataset Shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# %% [markdown]
# ## Logistic Regression and Evaluation

# %%
# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, lr.predict_proba(X_test), multi_class='ovr')
pr_auc = average_precision_score(y_test, lr.predict_proba(X_test), average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("Logistic Regression Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# ## Random Forest and Evaluation

# %%
# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test), multi_class='ovr')
pr_auc = average_precision_score(y_test, rf.predict_proba(X_test), average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("Random Forest Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# ## Decision Tree and Evaluation

# %%
# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, dt.predict_proba(X_test), multi_class='ovr')
pr_auc = average_precision_score(y_test, dt.predict_proba(X_test), average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("Decision Tree Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# ## CatBoost and Evaluation

# %%
# CatBoost
cb = CatBoostClassifier(random_state=42, verbose=0)
cb.fit(X_train, y_train)
y_pred = cb.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, cb.predict_proba(X_test), multi_class='ovr')
pr_auc = average_precision_score(y_test, cb.predict_proba(X_test), average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("CatBoost Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# ## Gradient Boosting and Evaluation

# %%
# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, gb.predict_proba(X_test), multi_class='ovr')
pr_auc = average_precision_score(y_test, gb.predict_proba(X_test), average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("Gradient Boosting Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# ## XGBoost and Evaluation

# %%
# XGBoost
xgb = XGBClassifier(random_state=42)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, xgb.predict_proba(X_test), multi_class='ovr')
pr_auc = average_precision_score(y_test, xgb.predict_proba(X_test), average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("XGBoost Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# # Including HyperParameter Boosting using RandomizedSearchCV

# %% [markdown]
#  ## Logistic Regression and Evaluation

# %%
# Logistic Regression with RandomizedSearchCV
param_dist_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

lr_random = RandomizedSearchCV(
    LogisticRegression(max_iter=1000),
    param_distributions=param_dist_lr,
    n_iter=10,  
    cv=3,       
    scoring='accuracy',
    n_jobs=-1,  
    random_state=42
)

lr_random.fit(X_train, y_train)

# Best parameters and evaluation
print("Best Parameters for Logistic Regression:", lr_random.best_params_)
y_pred = lr_random.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, lr_random.predict_proba(X_test), multi_class='ovr')
pr_auc = average_precision_score(y_test, lr_random.predict_proba(X_test), average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("Logistic Regression with RandomizedSearchCV Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# ## Random Forest and Evaluation

# %%
# Random Forest with RandomizedSearchCV
param_dist_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist_rf,
    n_iter=10,  
    cv=3,       
    scoring='accuracy',
    n_jobs=-1,  
    random_state=42
)

rf_random.fit(X_train, y_train)

# Best parameters and evaluation
print("Best Parameters for Random Forest:", rf_random.best_params_)
y_pred = rf_random.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, rf_random.predict_proba(X_test), multi_class='ovr')
pr_auc = average_precision_score(y_test, rf_random.predict_proba(X_test), average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("Random Forest with RandomizedSearchCV Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# ## Decision Tree and Evaluation

# %%
# Decision Tree with RandomizedSearchCV
param_dist_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt_random = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_distributions=param_dist_dt,
    n_iter=10,  
    cv=3,       
    scoring='accuracy',
    n_jobs=-1,  
    random_state=42
)

dt_random.fit(X_train, y_train)

# Best parameters and evaluation
print("Best Parameters for Decision Tree:", dt_random.best_params_)
y_pred = dt_random.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, dt_random.predict_proba(X_test), multi_class='ovr')
pr_auc = average_precision_score(y_test, dt_random.predict_proba(X_test), average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("Decision Tree with RandomizedSearchCV Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# ## CatBoost and Evaluation

# %%
# CatBoost with RandomizedSearchCV
param_dist_cb = {
    'iterations': [100, 200, 300],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2]
}

cb_random = RandomizedSearchCV(
    CatBoostClassifier(random_state=42, verbose=0),
    param_distributions=param_dist_cb,
    n_iter=10,  
    cv=3,       
    scoring='accuracy',
    n_jobs=-1,  
    random_state=42
)

cb_random.fit(X_train, y_train)

# Best parameters and evaluation
print("Best Parameters for CatBoost:", cb_random.best_params_)
y_pred = cb_random.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, cb_random.predict_proba(X_test), multi_class='ovr')
pr_auc = average_precision_score(y_test, cb_random.predict_proba(X_test), average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("CatBoost with RandomizedSearchCV Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# ## Gradient Boosting and Evaluation

# %%
# Gradient Boosting with RandomizedSearchCV
param_dist_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.2],
    'max_depth': [3, 4]
}

gb_random = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions=param_dist_gb,
    n_iter=5,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Fit on the smaller dataset
gb_random.fit(X_train, y_train)

# Best parameters and evaluation
print("Best Parameters for Gradient Boosting:", gb_random.best_params_)
y_pred = gb_random.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, gb_random.predict_proba(X_test), multi_class='ovr')
pr_auc = average_precision_score(y_test, gb_random.predict_proba(X_test), average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("Gradient Boosting with RandomizedSearchCV Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# %% [markdown]
# ## XGBoost and Evaluation

# %%
# XGBoost with RandomizedSearchCV
param_dist_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2]
}

xgb_random = RandomizedSearchCV(
    XGBClassifier(random_state=42),
    param_distributions=param_dist_xgb,
    n_iter=10,  
    cv=3,       
    scoring='accuracy',
    n_jobs=-1,  
    random_state=42
)

xgb_random.fit(X_train, y_train)

# Best parameters and evaluation
print("Best Parameters for XGBoost:", xgb_random.best_params_)
y_pred = xgb_random.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, xgb_random.predict_proba(X_test), multi_class='ovr')
pr_auc = average_precision_score(y_test, xgb_random.predict_proba(X_test), average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print("XGBoost with RandomizedSearchCV Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print(f"PR AUC: {pr_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")


