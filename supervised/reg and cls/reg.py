import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
df=pd.read_csv(r"C:\dataset\financial_regression.csv")
df.columns

df.info()
df.isnull().sum()
df.describe()

#fill all the nan with mean for all the columns 
df = df.fillna(df.mean(numeric_only=True))
df.isnull().sum()

X=df.drop(columns=["date","sp500 close"])
y=df["sp500 close"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


#create a function to evalute model
def evalute_model(true,predicted):
    mae=mean_absolute_error(true,predicted)
    mse=mean_squared_error(true, predicted)
    rmse=np.sqrt(mean_squared_error(true,predicted))
    r2_square=r2_score(true,predicted)
    return mae,mse,rmse,r2_square

#model training
models = {
    
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "support vector machine":SVR(),
   
}
for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)

    model_test_mae , model_test_rmse, model_test_r2, model_test_mse = evalute_model(y_test, y_pred)

    print(list(models.keys())[i])
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    print('='*35)
    print('\n') 