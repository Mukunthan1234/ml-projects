import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
from sklearn.metrics import r2_score

df=pd.read_csv(r"C:\Users\Mukundhan\OneDrive\Desktop\datasets\Salary_dataset.csv")
df.isnull().sum()
X=df[["YearsExperience"]]
y=df["Salary"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scalar=StandardScaler()
X_scaled_data=scalar.fit_transform(X_train)

X_test_scaled=scalar.transform(X_test)

 

model=LinearRegression()
model.fit(X_scaled_data,y_train)

y_pred=model.predict(X_test_scaled)
print(y_pred)


mse=mean_squared_error(y_test, y_pred)
mae=mean_absolute_error(y_test, y_pred)
rmse=np.sqrt(mse)
score=r2_score(y_test, y_pred)
print(mse)
print(mae)
print(rmse)
print(score)
