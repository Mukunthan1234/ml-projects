import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("C:\ML\Regression\Concrete_Data_Yeh.csv")
df.describe()
df.isnull().sum()
X=df[["cement","slag","flyash","water","superplasticizer","coarseaggregate","fineaggregate","age"]] 
y=df["csMPa"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print(y_pred)

data=[[1000.34,1443.2,1,2000,3.5,2098,453,30]]
new_pred=model.predict(data)
print(new_pred)
