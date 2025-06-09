"""
LightGBM - implementation of gradient boosting designed to handle large dtaaerts and high dimentional data with speed and accuracy
features - histogram based splitting, lead wise tree growth, support GPU training, handling sparse data
Advantages - faster training than xgboost, handles large datasets effectively reduces memory usage with histogram based splitting
When to use LightGBM -  large datasets with numerical features, time sensitive tasks requiring fast training
-------------------------------------------------------------------------------------------------------------------------------------------
Catboost -  gradient boosting library to handle categorical features without the need for preprocessing like on hot encoding
features - native support for categorical data, ordered boosting, robust to overfitting
advantages -  eliminates the need for manual encoding of data, reduces overfitting with boosting techniques, easy to implement for datasets with many categorical features
When to use - datasets with a high proportion of categorical features, applications where overfitting is a concern
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# load titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# select features and target
features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
target = "Survived"

# handle missing values
df.fillna({"Age": df["Age"].median()}, inplace=True)
df.fillna({"Embarked": df["Embarked"].mode()[0]}, inplace=True)

# encode categorical variables
label_encoders = {}
for col in ["Sex", "Embarked"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training data shape {X_train.shape}")
print(f"Test data shape {X_test.shape}")

# train lightgbm model
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

# predict and evaluate
lgb_pred = lgb_model.predict(X_test)
print(f"LightGBM Accuraxcy: {accuracy_score(y_test, lgb_pred):.4f}")


# train catboost model
cat_features = ["Pclass", "Sex", "Embarked"]
cat_model = CatBoostClassifier(cat_features=cat_features, verbose=0)
cat_model.fit(X_train, y_train)

# predict and evaluate
cat_pred = cat_model.predict(X_test)
print(f"Cat Boost Accuracy: {accuracy_score(y_test,cat_pred)}")

# train XGBoost model
xgb_model = XGBClassifier(eval_metrix="logloss")
xgb_model.fit(X_train, y_train)

# predict and evaluate
xgb_pred = xgb_model.predict(X_test)
print(f"Cat Boost Accuracy: {accuracy_score(y_test,xgb_pred)}")

# train catboost without encoding categorical features
cat_model_native = CatBoostClassifier(cat_features=["Sex", "Embarked"], verbose=0)
cat_model_native.fit(X_train, y_train)

# predict and evaluate
cat_pred_native = cat_model_native.predict(X_test)
print(f"CatBoost Native Accuracy: {accuracy_score(y_test,cat_pred_native):.4f}")
