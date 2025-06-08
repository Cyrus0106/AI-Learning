# process of identifying and retaining the most relevant features
# improves model performance, reduces overfitting, enhances interpretability, increase computational efficiency

from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression


# load dataset
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# display dataset info
print(df.head())
print(df.info())

#calculate correlation matrix
correlation_matrix = df.corr()

#plot heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# select features with high correlation to the target variable
correlated_features = correlation_matrix['target'].sort_values(ascending=False)
print("Features most correlated with target:")
print(correlated_features)

# separate featured and targets
X = df.drop(columns=['target'])
y = df['target']

# calculate mutual information
mutual_info = mutual_info_regression(X,y)

# create a dataframe for better visualisation
mi_df = pd.DataFrame({'Feature':X.columns, "Mutual Information":mutual_info})
mi_df = mi_df.sort_values(by="Mutual Information", ascending=False)

print("Mutual Information Scores:")
print(mi_df)


from sklearn.ensemble import RandomForestRegressor
import numpy as np

# train a random forest model

model=RandomForestRegressor(random_state=42)
model.fit(X,y)

# get feature importance
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({'Feature':X.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importance from random forest:")
print(importance_df)

# plot feature importance
plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance from random forest")
plt.show()