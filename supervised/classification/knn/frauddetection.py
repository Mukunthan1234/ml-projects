import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  

#load the dataset into dataframe
df=pd.read_csv(r"C:\ML\supervised\classification\knn\Synthetic_Financial_datasets_log.csv")

#EDA
df.info()
df.isnull().sum()
df.describe()
missing=(df.isnull().sum()/len(df)*100)
print(missing)
#dist b/t transaction type col
plt.hist(df["type"],bins=10,color="red")
plt.title("distribution over type")
plt.xlable("type")
plt.ylabel("Frequency")
plt.show()


#preprocessing
df["type"]=df["type"].map({"PAYMENT":1,"TRANSFER":2,"CASH_OUT":3,"DEBIT":4,"CASH_IN":5})
df = df.drop(columns=['nameOrig', 'nameDest'])

X=df.drop(columns=["isFraud","isFlaggedFraud"])
y=df["isFraud"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,)
model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)

pred_test=model.predict(X_test)
print(pred_test)
