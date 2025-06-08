# used to make sure no feature dominates the model because of its magnitude
# min max scaling transforms features to specirfied range typically [0,1]
# standardization(Z score scaling) centers data around 0 and scaled it to have sd of 1

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# load dataset
data = load_iris()
X = pd.DataFrame(data.data,columns=data.feature_names)
y = data.target

# display dataset info
print("Dataset Info:")
print(X.describe())
print("Target Classes: ",data.target_names)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,train_size=0.2)

# train knn classfier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

# predict and evaluate
y_pred = knn.predict(X_test)
print("Accuracy without scaling:", accuracy_score(y_test,y_pred))

# apply min max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# split the dataset
X_train_Scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled,y,random_state=42,train_size=0.2)

# train knn classfier on sclaed data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_Scaled,y_train_scaled)

# predict and evaluate
y_pred_scaled = knn.predict(X_test_scaled)
print("Accuracy with scaling:", accuracy_score(y_test_scaled,y_pred_scaled))

# apply standardisation
scaler = s=StandardScaler()
X_standard = scaler.fit_transform(X)

# split the dataset
X_train_standard, X_test_standard, y_train_standard, y_test_standard = train_test_split(X_standard,y,random_state=42,train_size=0.2)

# train knn classfier on sclaed data
knn_stand = KNeighborsClassifier(n_neighbors=5)
knn_stand.fit(X_train_standard,y_train_standard)

# predict and evaluate
y_pred_std = knn.predict(X_test_standard)
print("Accuracy with Standardisation:", accuracy_score(y_test_standard,y_pred_std))