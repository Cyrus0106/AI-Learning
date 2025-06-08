# ensemble machine learning technique combines the predictions of multiple models to produce a final output
# it reduces variance , reduces bias and improves robustness

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


# load dataset
data = load_iris()
X,y = data.data, data.target

# split dataset
X_train , X_test, y_train, y_test = train_test_split(X, y , random_state=42, train_size=0.2)

# scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train individual models
log_model = LogisticRegression()
dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()

log_model.fit(X_train,y_train)
dt_model.fit(X_train,y_train)
knn_model.fit(X_train,y_train)

# creating voting classifier
ensemble_model = VotingClassifier(
    estimators=[
        ('log_reg', log_model),
        ('decision_tree', dt_model),
        ('knn', knn_model)
    ],
    voting='hard'
)

# train ensemble model
ensemble_model.fit(X_train,y_train)

#predict with ensemble 
y_pred_ensemble = ensemble_model.predict(X_test)


# evaluate individual models
y_pred_log = log_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)

print("Ensemble Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print("Logarithm Accuracy:", accuracy_score(y_test, y_pred_log))
print("Decision Treee Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Knn Accuracy:", accuracy_score(y_test, y_pred_knn))
