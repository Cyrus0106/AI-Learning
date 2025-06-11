"""
what are parameters - values learned by a ML model during training, adjusted to minimise loss function and optimise predictions
hyperparameters are setting defined before training that influence how the model learns from data, not learned from the data but instead control the learning process
why tune - improve model performane, enhance efficiency, adapt to problem specific needs.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# load data
data = load_breast_cancer()
X, y = data.data, data.target

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

# display dataset info
print("Feature names", data.feature_names)
print("Class names:", data.target_names)

# train random forest with default hyperparameters
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)

# redict and evaluate
y_predict_default = rf_default.predict(X_test)
accuracy_default = accuracy_score(y_test, y_predict_default)

print(f"Default model Accuracy {accuracy_default:.4f}")
print("\nClassfication report\n", classification_report(y_test, y_predict_default))

# train random forest with adjusted hyperparameters
rf_tuned = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)

rf_tuned.fit(X_train, y_train)

# predict and evalueate
y_pred_tuned = rf_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Tuned model Accuracy {accuracy_tuned:.4f}")
print("\nClassfication report\n", classification_report(y_test, y_pred_tuned))
