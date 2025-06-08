# bagging trains multiple mdels on different subsets of the data by random sampling with replacement - reduces variance, improve robustness
# random forest uilds multiple decision trees using bagging
# random forst key parameters = number of trees(N_estimators -  number of decision trees in the forest), maximum depth(max_depth - limits the depth of each tree to prevent overfitting)
# feature selection(max_feautures - number of features to consider when looking for the best split)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# display the dataset information
print("Features:", data.feature_names)
print("Classes:", data.target_names)

# train random forest
rf_model = RandomForestClassifier(random_state= 42)
rf_model.fit(X_train,y_train)

# predict
y_pred = rf_model.predict(X_test)

# evaluate performance
accuracy = accuracy_score(y_test,y_pred)
print("Random Forest Accuracy:", accuracy)
print("\nClassification Report: \n", classification_report(y_test,y_pred))

# define hyperparameter grid
param_grid = {
    'n_estimators': [50,100,200],
    'max_depth': [None,10,20],
    'max_features': ['sqrt','log2','None']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv =5,
    scoring='accuracy',
    n_jobs=1
)

grid_search.fit(X_train, y_train)

# display the best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best Cross Validation accuracy: {grid_search.best_score_}")
