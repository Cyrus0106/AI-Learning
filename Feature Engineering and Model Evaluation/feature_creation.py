# deriving new features from existing features
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# load bike dataset
df = pd.read_csv("datasets/bike_sharing_daily.csv")

# display dataset information
print("Dataset Info:")
print(df.info())

# preview first few rows
print("Dataset preview")
print(df.head(10))

# cnvert dteday to datetime
df['dteday'] = pd.to_datetime(df["dteday"])

# create new features
df['day_of_week'] = df['dteday'].dt.day_name()
df['month'] = df['dteday'].dt.month
df['year'] = df['dteday'].dt.year

# display new features 
print("New features devrived from date columns")
print(df[['dteday', 'day_of_week','month','year']].head(10))

# select feature and target
X= df[['temp']]
y = df['cnt']


#apply polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# DIsplay the transformed feature
print("Original and polynomial features")
print(pd.DataFrame(X_poly,columns=['temp','temp^2']).head())

# split dataset
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_poly_train , X_poly_test  = train_test_split(X_poly,test_size=0.2,random_state=42)

# train and evalute model with original fatures 
model_original = LinearRegression()
model_original.fit(X_train,y_train)
y_pred_original = model_original.predict(X_test)
mse_orginal = mean_squared_error(y_test,y_pred_original)

# train and evaluate model with tranformed features
model_tranformed= LinearRegression()
model_tranformed.fit(X_poly_train,y_train)
y_pred_tranformed = model_tranformed.predict(X_poly_test)
mse_tranformed = mean_squared_error(y_test,y_pred_tranformed)


# copare result
print("MSE Originial", mse_orginal)
print("MSE polynomial", mse_tranformed)