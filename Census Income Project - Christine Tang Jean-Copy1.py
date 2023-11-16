#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset
file_path = "adult.data"
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

df = pd.read_csv(file_path, names=column_names, na_values=' ?', skipinitialspace=True)

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Encode categorical variables
label_encoder = LabelEncoder()
df_encoded = df_imputed.apply(label_encoder.fit_transform)

# Normalize the data
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)

# Create a binary target variable 'income_over_50k'
df_normalized['income_over_50k'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Split the data into features (X) and target (y)
X = df_normalized.drop(['income', 'income_over_50k'], axis=1)
y = df_normalized['income_over_50k']

# Address class imbalance using train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Classification with Random Forest
clf = RandomForestClassifier(random_state=42, n_jobs=-1)  # n_jobs=-1 for parallel processing
clf.fit(X_train, y_train)

# Predict for the test set and get probability estimates
predictions = clf.predict(X_test)
proba_estimates = clf.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class ('>50K')

# Model Performance Evaluation on the test set
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, proba_estimates)

# Display Classification Metrics
print("\nClassification Metrics (Random Forest):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Error Accuracy and Algorithms with Error (from adult.names)
error_accuracy_c45 = 84.46
error_accuracy_naive_bayes = 83.88
error_accuracy_nbtree = 85.90

# Display Error Accuracy and Algorithms with Error
print("\nError Accuracy and Algorithms with Error:")
print(f"C4.5 Error Accuracy: {error_accuracy_c45:.2f}%")
print(f"Naive Bayes Error Accuracy: {error_accuracy_naive_bayes:.2f}%")
print(f"NBTree Error Accuracy: {error_accuracy_nbtree:.2f}%")

# Add predictions and probability estimates to the original dataframe for exploration
df_test = df.loc[X_test.index].copy()
df_test['predicted_income_over_50k_rf'] = predictions
df_test['probability_over_50k_rf'] = proba_estimates

# Display the dataframe with predictions and probability estimates on the test set
print("\nDataset with Predictions and Probability Estimates (Random Forest - Test Set):")
print(df_test[['income', 'predicted_income_over_50k_rf', 'probability_over_50k_rf']].head())


# In[ ]:




