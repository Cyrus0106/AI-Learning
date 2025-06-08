# task 1 perform eda and preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# load data
data = fetch_california_housing(as_frame=True)
df = data.frame

# define features and target
X = df[['MedInc','AveRooms','HouseAge']]
y = df[['MedHouseVal']]

# inspect data
print(df.info())
print(df.describe())

# visualise relationships
sns.pairplot(df, vars=['MedInc','AveRooms','HouseAge','MedHouseVal'])
plt.show()

# check for missing values
print("Missing values \n",df.isnull().sum())

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,train_size=0.2)

# train linear regression model
model = LinearRegression()
model.fit(X_train,y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate performance
mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error: ",mse)