# Categorical features are variables that represent categories or labels (like "red", "blue", "green"). These can be converted into numerical form using techniques such as one-hot encoding (which creates a new binary column for each category) or label encoding (which assigns each category a unique integer). This transformation is necessary because most machine learning algorithms require numerical input.

# Numerical features are continuous or discrete numbers (like age or salary). These features often need to be scaled so that they have similar ranges, which helps many algorithms perform better. Common scaling methods include standardization (subtracting the mean and dividing by the standard deviation) and normalization (scaling values to a range, such as 0 to 1).

# Ordinal features are categorical variables with a meaningful order (like "low", "medium", "high"). Ordinal encoding assigns integers to these categories based on their order, preserving the ranking information.

# Log transformation is used to reduce skewness in numerical data. Skewed data can negatively impact model performance, and applying a logarithmic transformation can make the distribution more symmetric and closer to normal.

# Polynomial features involve creating new features by raising existing features to higher powers or multiplying them together. This allows models to capture non-linear relationships between variables and the target, which can improve predictive performance for certain algorithms.

# feature selection - reduces numer of input features to improve model performance

import pandas as pd

# load titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# display dataset information
print("Dataset info: \n")
print(df.info())

# preview the first few rows
print("\n Dataset Preview:\n")
print(df.head(10))

# separate features
categorical_features = df.select_dtypes(include=(["object"])).columns
numerical_features = df.select_dtypes(include=(["int64","float64"])).columns

print("\nCat Features", categorical_features.tolist())
print("\nNumerical Features", numerical_features.tolist())

# display summary of categorical features
print("\n Categorical feature summary: \n")
for col in categorical_features:
    print(f"{col}:\n",df[col].value_counts(),"\n")

#display summary of numercial features
print("\n Numercial feature summary: \n")
print(df[numerical_features].describe())

