"""
Imbalanced data - refers t datasets where one class significantly outnumber the others
challenges- bias toward majority class, misleading evaluation metrics, limitied ino for minority class
techniques to handle imbalance data - resample by oversampling or undersampling, algorithmic solutions - class weights, anomaly detection models
evaluation metrics for imballanced data - f1 score, roc-auc, precision recall curve
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# load dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

# explore dataset
print("Dataset info:\n")
print(df.info())
print("\nClass Distribution:\n")
print(df["Class"].value_counts())

# split dataset
X = df.drop(columns=["Class"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

# train random forest
rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)

# predict and evaluate
y_pred = rf_model.predict(X_test)
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
print(f"ROC-AUC: {roc_auc:.4f}")

# apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# display new class distribution
print("\n CLass Distribution after SMOTE: \n")
print(pd.Series(y_resampled).value_counts())

# train rando forest on resampled data
rf_model_smote = RandomForestClassifier(random_state=42)
rf_model_smote.fit(X_train, y_train)

# predict and evalute
y_pred_smote = rf_model_smote.predict(X_test)
print("\n Classification Report(SM):\n")
print(classification_report(y_test, y_pred_smote))

roc_auc_smote = roc_auc_score(y_test, rf_model_smote.predict_proba(X_test)[:, 1])
print(f"ROC-AUC(SM): {roc_auc_smote:.4f}")
