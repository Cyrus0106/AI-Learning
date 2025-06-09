"""
boosting - ensemble technique that sequentially combines weak learners to form a strong learner
each model tries to correct errors made by previous models
gradient boosting - builds models by minimising a loss function using gradient descent
key params - learning rate(determines contribution of each weak learner, small value reduce overfitting but need more iterations), number of estiators(the number of trees added sequentially
larger values improve learning but increase computation time), regularisation - techniques like limiting treee depth or adding penalties to prevent overfitting
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# split datast
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

# display dataset
print(f"Features: {data.feature_names}")
print(f"Classes: {data.target_names}")

# train gradient boosting model
gd_model = GradientBoostingClassifier(random_state=42)
gd_model.fit(X_train, y_train)

# predict
y_pred_gd = gd_model.predict(X_test)

# Evaluate performance
accuracy_gb = accuracy_score(y_test, y_pred_gd)
print(f"Gradient Boosting Accuracy: {accuracy_gb}")
print(f"\nClassfication Report:\n {classification_report(y_test, y_pred_gd)}")

# define hyperparameter grid
param_grid = {
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
}

# perform grid search
grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=1,
)

grid_search.fit(X_train, y_train)

# display best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_}")

# train random forrest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# predict
y_pred_rf = rf_model.predict(X_test)

# evaluate performance
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random forest accuracy", {accuracy_rf})

