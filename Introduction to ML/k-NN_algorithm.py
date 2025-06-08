from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# load iris dataset
data = load_iris()
X, y = data.data, data.target

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# experiment with different values of k
for k in range(1,11):
    # initialise k-NN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)

    # predict on test data
    y_pred = knn.predict(X_test)

    # evaluate the performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"k = {k}, Acccuracy = {accuracy:.2f}")

from sklearn.linear_model import LogisticRegression

# train logisitic regression model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train,y_train)

# predict with logisitic reg
y_pred_lr = log_reg.predict(X_test)

# evaluate log regression
accuracy_lr = accuracy_score(y_test,y_pred_lr)
print("Logistic regression accuracy: ",accuracy_lr)