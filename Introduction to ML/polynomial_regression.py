import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# load the dataset
data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.head(5))

# select features (Median Income) and target (Median House Value)
X = df[["MedInc"]]
y = df[["MedHouseVal"]]

# transform the feature to polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# fit polynomial regression model
# model = LinearRegression()
# model.fit(X_poly,y)

# # make predictions
# y_pred = model.predict(X_poly)

# # plot actual vs predicted
# plt.figure(figsize=(10,6))
# plt.scatter(X,y,color="blue",label="Actual Data")
# plt.plot(X,y_pred,color="red",label="Predicted Curve")
# plt.title("Polynomial Regression Model")
# plt.xlabel("Median Income in Cali")
# plt.ylabel("Median House Value in Cali")
# plt.legend()
# plt.show()

# # evaluate model perfomance
# mse = mean_squared_error(X,y_pred)
# r2 = r2_score(X,y_pred)
# print("MSE: ",mse)
# print("R2: ",r2)

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

#split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_poly,y,test_size=0.2,random_state=42)

# ridge regression
ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train,y_train)
ridge_predictions = ridge_model.predict(X_test)

# lasso regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)
lasso_predictions = lasso_model.predict(X_test)

# evaluate the models
ridge_mse = mean_squared_error(y_test, ridge_predictions)
ridge_r2 = r2_score(y_test,ridge_predictions)
lasos_mse = mean_squared_error(y_test, lasso_predictions)
lasso_r2 = r2_score(y_test,lasso_predictions)

print("Ridge MSE: ",ridge_mse)
print("Lasso MSE: ",lasos_mse)
print("Ridge r2: ",ridge_r2)
print("Lasso r2: ",lasso_r2)

# visualise ridge vs lasso
plt.figure(figsize=(10,6))
plt.scatter(X_test[:,0],y_test,color="blue",label="Actual Data",alpha=0.5)
plt.plot(X_test[:,0],ridge_predictions,color="green",label="Ridge Predictions", alpha=0.5)
plt.plot(X_test[:,0],lasso_predictions,color="orange",label="Lasso Predictions", alpha=0.5)
plt.title("Ridge vs Lasso Regression")
plt.xlabel("Median Income(Transformed)")
plt.ylabel("Median House Value in Cali")
plt.legend()
plt.show()