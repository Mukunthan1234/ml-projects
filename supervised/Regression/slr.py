import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv(r"C:\Users\Mukundhan\OneDrive\Desktop\dataset\Salary_dataset.csv")
df.isnull().sum()

X=df[["YearsExperience"]]
y=df["Salary"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
plt.scatter(df["YearsExperience"],df["Salary"])
plt.show()
