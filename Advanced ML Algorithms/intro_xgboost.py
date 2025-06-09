"""Advanced implementation for gradient boosting algorithm for speed and performance
its improvements are with speed, handling missing data, regularisation, custom loss functions, tree pruning
key hyper parameters are learning rate, number of trees, tree depth and subsample, colsample_bytree and regularisation parameters
"""

import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

# load datasets
data = load_breast_cancer()
X, y = data.data, data.target

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

# display dataset info
print(f"Features: {data.feature_names}")
print(f"Classes: {data.target_names}")

# convert dataset to dmatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# train XGBoost Model
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "eta": 0.1,
}

xgb_model = xgb.train(params, dtrain, num_boost_round=100)

# predict
y_pred = (xgb_model.predict(dtest) > 0.5).astype(int)

# evalaute performance
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy}")
print("\nClassification Report: \n", classification_report(y_test, y_pred))

# define the hyperparameter grid
param_grid = {
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# intialise xgboost classfier
xgb_clf = XGBClassifier(
    user_label_encoder=False, eval_metric="logloss", random_state=42
)

# preform grid search
grid_search = GridSearchCV(
    estimator=xgb_clf, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1
)
grid_search.fit(X_train, y_train)

# display best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accruacy: {grid_search.best_score_}")

# train gradient boosting model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# evaluate gradient boosting performance
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Gradient Boosting Accuracy: {accuracy_gb}")