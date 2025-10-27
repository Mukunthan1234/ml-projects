import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


df=pd.read_csv(r"C:\ML\supervised\classification\decison tree\financial_risk_assessment.csv")

new_df=df.drop(columns=["Education Level","Marital Status","City","State","Country","Marital Status Change","Payment History","Loan Purpose","Gender"])
print(new_df)

new_df["Income"].fillna(df["Income"].mean(),inplace=True)
new_df["Credit Score"].fillna(df["Credit Score"].mean(),inplace=True)
new_df["Loan Amount"].fillna(df["Loan Amount"].mean(),inplace=True)
new_df["Assets Value"].fillna(df["Assets Value"].mean(),inplace=True)
new_df["Number of Dependents"].fillna(df["Number of Dependents"].median(),inplace=True)
new_df["Previous Defaults"].fillna(df["Previous Defaults"].median(),inplace=True)

new_df.isnull().sum()
new_df.describe()
new_df["Risk Rating"]=new_df["Risk Rating"].map({"Low":0,"Medium":1,"High":2})
df.isnull().sum()
new_df["Employment Status"]=df["Employment Status"].map({"Unemployed":0,"Employed":1,"Self-employed":2})

models={"DecisionTree":DecisionTreeClassifier(),"navie_bayes":GaussianNB()}
for i in range(len(list(models))):
    model=list(models.Values())[i]

X=new_df.drop(columns=["Risk Rating"])
y=new_df["Risk Rating"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier(criterion="entropy",random_state=0)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
