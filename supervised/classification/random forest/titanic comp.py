import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import glob


csv_files = glob.glob(r"C:\kaggle comp\*.csv")

df = [pd.read_csv(file) for file in csv_files]
print(df)
df=pd.read_csv(r"C:\kaggle comp\train.csv")
df.isnull().sum()

df["Age"].fillna(df["Age"].mean(),inplace=True)

scaler=StandardScaler()
df[["Age","Fare"]]=scaler.fit_transform(df[["Age","Fare"]])

df.drop('Cabin', axis=1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)


df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

X=df[["PassengerId","Pclass","Sex","Age","Fare","Embarked_Q","Embarked_S"]]
y=df[["Survived"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()
model.fit(X_train,y_train)

pred=pd.read_csv(r"C:\kaggle comp\test.csv")
pred.drop('Cabin', axis=1, inplace=True)
pred.drop('Ticket', axis=1, inplace=True)
pred["Age"].fillna(pred["Age"].mean(),inplace=True)
scaler=StandardScaler()
pred[["Age","Fare"]]=scaler.fit_transform(pred[["Age","Fare"]])
pred['Sex'] = df['Sex'].map({'male':0, 'female':1})
pred = pd.get_dummies(pred, columns=['Embarked'], drop_first=True)
pred.drop('SibSp', axis=1, inplace=True)
pred.drop('Parch', axis=1, inplace=True)
pred.drop('Name', axis=1, inplace=True)

y_pred=model.predict(pred)
print(y_pred)



# Assuming you have a test DataFrame with PassengerId column
output = pd.DataFrame({'PassengerId': pred['PassengerId'], 'Survived': y_pred})

# Save to CSV
output.to_csv('submission.csv', index=False)

print("CSV file saved successfully!")

