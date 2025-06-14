import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report

# generate synthetic dataset
np.random.seed(42)
n_samples = 200
X = np.random.rand(n_samples, 2) * 10
y = (X[:,0] * 1.5 + X[:,1] > 15).astype(int)

# create a dataframe
df = pd.DataFrame(X,columns=['Age','Salary'])
df['Purchase'] = y

# split data
X_train, X_test, y_train, y_test = train_test_split(df[["Age","Salary"]], df["Purchase"], test_size=0.2, random_state=42)

# train logisitic model
model = LogisticRegression()
model.fit(X_train,y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate performance
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
classification = classification_report(y_test,y_pred)

eval = {"Accuracy Score":accuracy,
        "Precision Score": precision,
        "Recall Score": recall,
        "F1 Score": f1,
        "Classification Report": classification}

for i in eval:
    print(f"{i}: {eval[i]}")

#plot decision boundary
x_min, x_max = X[:,0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:,1].min() -1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min,x_max,0.1), np.arange(y_min,y_max,0.1))

# predict probabilities for grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#plot
plt.contour(xx,yy,Z,alpha=0.8,cmap="coolwarm")
plt.scatter(X_test["Age"],X_test["Salary"],c=y_test,edgecolor="k",cmap="coolwarm")
plt.title("Logisitic Regression Decision Boundary")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()