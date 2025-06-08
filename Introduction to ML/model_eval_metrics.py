# model ealuation metrics for regression and classifications
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

# load datasets
data = load_iris()
X, y = data.data, data.target

# initialise classifier
model = RandomForestClassifier(random_state=42)

# perform k fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model,X, y, cv=kf, scoring="accuracy")

# output results
print("Cross validation scores: ", cv_scores)
print("Mean accuracy", cv_scores.mean())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# load dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# train logistic regression
model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)

# predict on test data
y_pred = model.predict(X_test)

# generate the confusion matrix
cm = confusion_matrix(y_test,y_pred)

# display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names) 
disp.plot(cmap="Blues")
plt.show()

# print classficiation report
print("Classification report:\n", classification_report(y_test,y_pred))