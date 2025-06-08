import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# display dataset info
print(df.info())

# preview first few rows
print(df.head(10))

# apply one hot encoding
df_one_hot = pd.get_dummies(df, columns=["Sex","Embarked"], drop_first=True)

# display encoded dataset
print("\n One hot encoded dataset:")
print(df_one_hot.head(10))

# apply label encoding
label_encoder = LabelEncoder()
df['Pclass_encoded'] = label_encoder.fit_transform(df['Pclass'])

# display encoded dataset
print("\n Label encoded dataset:")
print(df[['Pclass','Pclass_encoded']].head())

# apply frequency encoding
df['Ticket_frequency'] = df['Ticket'].map(df['Ticket'].value_counts())

# display frequency encoded feature
print("\n Frequency encoded feature: ")
print(df[['Ticket','Ticket_frequency']].head())

X = df_one_hot.drop(columns=['Survived','Name','Ticket','Cabin','Age'])
y = df['Survived']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# train logsitic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)

# predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy score: ",accuracy_score(y_test,y_pred))