"""
Grid Search - systematically evaluates all possible combinations of hyperparameter values within a specified grid
+ Evaluates all combinations
- computationally expensive for large grids
- limited to predefined grid
= small parameter spaces
random search - alternative method where hyperparamter combinations are sampled randomly from the specified ranges
+ randomly samples combinations
+ faster for large parameter spaces
+ explores more diverse ranges
= large parameter spaces with time constraint
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# load dataset
data = load_iris()
X, y = data.data, data.target

# split dtaaset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# display dataset info
print("Feature names", data.feature_names)
print("Class names:", data.target_names)

# define hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
}

# initialise grid search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=1,
)

# perform grid search 
grid_search.fit(X_train,y_train)

# evaluate best model
best_grid_model = grid_search.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)
accuracy_grid = accuracy_score(y_test,y_pred_grid)

print(f"Best Hyperparamets (Grid Search): {grid_search.best_params_}")
print(f"Grid Search Accuracy: {accuracy_grid:.4f}")

# define hyperparameter distribution
param_dist = {
    "n_estimators": np.arange(50,200,10),
    "max_depth": [None, 5, 10,15],
    "min_samples_split": [2, 5, 10,20],
}

# initialise random search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# perofrm random search
random_search.fit(X_train,y_train)

# evaluate best model
best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)
accuracy_random = accuracy_score(y_test,y_pred_random)

print(f"Best hyperparameters (Random Search): {random_search.best_params_}")
print(f"Random Search Accuracy: {accuracy_random:.4f}")